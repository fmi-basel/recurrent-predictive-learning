import sys

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from repl.metrics.metrics import participation_ratio
from repl.scripts.utils import dist_reduce_mean, get_criterion_input, log_variable
from repl.scripts.eval_utils import compute_readout_loss


def online_eval(args, 
                model, 
                readout, 
                test_loader, 
                criterion, 
                classifier_criterion,
                regression_criterion,
                device, 
                epoch, 
                model_output_idx,
                multitask=False, 
                num_classes_seq_labels=None, 
                num_classes_dense_labels=None,
                num_regression_targets=None,
                num_regression_dense_targets=None):
    """Evaluate the model on the test set in online manner and log the results
    
    Args:
        args (argparse.Namespace): Command line arguments
        model (RePLModel): Model to evaluate
        readout (nn.Module): Readout network
        test_loader (torch.utils.data.DataLoader): Test data loader
        criterion (nn.Module): loss function used to train RePL model
        classifier_criterion (nn.Module): Classifier loss function
        regression_criterion (nn.Module): Regression loss function
        device (torch.device): Device to use
        epoch (int): Current epoch
        model_output_idx (int): Index of the model output to use
        multitask (bool): Whether the classification task is multilabel
        num_classes_seq_labels (int): Number of classes for sequence labels
        num_classes_dense_labels (int): Number of classes for dense labels (seq2seq)
        num_regression_targets (int): Number of regression targets
        num_regression_dense_targets (int): Number of regression dense targets
    """
    # evaluate on test set
    loss = 0.
    
    if multitask:
        classifier_loss = ({key: 0. for key in num_classes_seq_labels.keys()}, {key: 0. for key in num_classes_dense_labels.keys()})
        classifier_acc1 = ({key: 0. for key in num_classes_seq_labels.keys()}, {key: 0. for key in num_classes_dense_labels.keys()})
        regression_loss = ({key: 0. for key in num_regression_targets.keys()}, {key: 0. for key in num_regression_dense_targets.keys()})
        regression_r2 = ({key: 0. for key in num_regression_targets.keys()}, {key: 0. for key in num_regression_dense_targets.keys()})
        classifier_acc5 = None
    else:
        classifier_loss = 0.
        classifier_acc1 = 0.
        regression_loss = 0.
        regression_r2 = 0.
        classifier_acc5 = 0.
    part_ratio = 0.
    part_ratio_context = 0.
    # Welford online accumulators for z_var and ctx_var (avoids storing all GPU tensors) written by mshalvagal using Claude
    _wf = {k: {'n': 0, 'mean': 0.0, 'M2': 0.0, 'vt_sum': 0.0, 'vt_n': 0}
           for k in ('z', 'ctx')}
    for x, y in tqdm(test_loader, disable=not sys.stdout.isatty()):
        x = x.to(device)
        if multitask:
            y = tuple([yi.to(device) for yi in y])
            target_length = None
        else:
            y = y.to(device)
            # if task is seq2seq classification (e.g. phone), y.shape = (B, L); otherwise y.shape = (B,)
            target_length = y.shape[1] if len(y.shape) == 2 else y.shape[0]
        with torch.no_grad():
            model_output = model(x)  # z, context, pred
            z, context, pred = model_output
            criterion_inputs = get_criterion_input(z, context, pred, y, args.loss, args.online_input, args.prediction_target)
            loss += criterion(*criterion_inputs)

            classifier_loss_, classifier_acc1_, classifier_acc5_, regression_loss_, regression_r2_ = compute_readout_loss(
                data=model_output[model_output_idx], 
                labels=y, 
                readout=readout, 
                classifier_criterion=classifier_criterion, 
                regression_criterion=regression_criterion,
                target_length=target_length, 
                downstream_input=args.online_input,
                dense_prediction=args.dense_prediction,
                single_readout=args.online_single_timestep_readout,
                pred_steps=args.pred_steps,
                task=args.online_task,
                full_spatial_readout=args.online_full_spatial_readout,)
            
            if multitask:
                seq2label_loss, seq2seq_loss = classifier_loss_
                seq2label_acc, seq2seq_acc = classifier_acc1_
                seq2label_regression_loss, seq2seq_regression_loss = regression_loss_
                seq2label_r2, seq2seq_r2 = regression_r2_
                if seq2label_loss:
                    for task_name, loss_val in seq2label_loss.items():
                        classifier_loss[0][task_name] = classifier_loss[0].get(task_name, 0) + loss_val/len(test_loader)
                    for task_name, acc_val in seq2label_acc.items():
                        classifier_acc1[0][task_name] = classifier_acc1[0].get(task_name, 0) + acc_val/len(test_loader)
                if seq2seq_loss:
                    for task_name, loss_val in seq2seq_loss.items():
                        classifier_loss[1][task_name] = classifier_loss[1].get(task_name, 0) + loss_val/len(test_loader)
                    for task_name, acc_val in seq2seq_acc.items():
                        classifier_acc1[1][task_name] = classifier_acc1[1].get(task_name, 0) + acc_val/len(test_loader)
                if seq2label_regression_loss:
                    for task_name, loss_val in seq2label_regression_loss.items():
                        regression_loss[0][task_name] = regression_loss[0].get(task_name, 0) + loss_val/len(test_loader)
                    for task_name, r2_val in seq2label_r2.items():
                        regression_r2[0][task_name] = regression_r2[0].get(task_name, 0) + r2_val/len(test_loader)
                if seq2seq_regression_loss:
                    for task_name, loss_val in seq2seq_regression_loss.items():
                        regression_loss[1][task_name] = regression_loss[1].get(task_name, 0) + loss_val/len(test_loader)
                    for task_name, r2_val in seq2seq_r2.items():
                        regression_r2[1][task_name] = regression_r2[1].get(task_name, 0) + r2_val/len(test_loader)
            else:
                classifier_loss += classifier_loss_/len(test_loader)
                classifier_acc1 += classifier_acc1_/len(test_loader)
                classifier_acc5 += classifier_acc5_/len(test_loader)

            part_ratio += participation_ratio(z) if not args.flatten_enc_output else 0.
            try:
                part_ratio_context += participation_ratio(context)
            except:
                part_ratio_context = -1.
            if len(z.shape) == 5:
                z = z.mean(-1).mean(-1)
            if len(context.shape) == 5:
                context = context.mean(-1).mean(-1)
            for key, t in (('z', z), ('ctx', context)):
                t_f = t.float()
                n_new = t_f.numel()
                mean_new = t_f.mean().item()
                M2_new = t_f.var(unbiased=False).item() * n_new
                # Chan et al. parallel merge
                n_old = _wf[key]['n']
                delta = mean_new - _wf[key]['mean']
                n_total = n_old + n_new
                _wf[key]['mean'] += delta * n_new / n_total
                _wf[key]['M2']   += M2_new + delta**2 * n_old * n_new / n_total
                _wf[key]['n']     = n_total
                # running mean of per-sequence variance over time (dim=1)
                _wf[key]['vt_sum'] += t_f.var(dim=1).sum().item()
                _wf[key]['vt_n']   += t_f.shape[0] * t_f.shape[-1]
    if args.online_task != "seq2seq" and not (args.online_full_spatial_readout and args.flatten_enc_output):
        z_var             = _wf['z']['M2']   / (_wf['z']['n']   - 1)
        avg_var_over_time = _wf['z']['vt_sum']  / _wf['z']['vt_n']
        ctx_var               = _wf['ctx']['M2']  / (_wf['ctx']['n']  - 1)
        avg_var_over_time_ctx = _wf['ctx']['vt_sum'] / _wf['ctx']['vt_n']
    else:
        z_var = None
        avg_var_over_time = None
        ctx_var = None
        avg_var_over_time_ctx = None
    loss /= len(test_loader)
    part_ratio /= len(test_loader)
    part_ratio_context /= len(test_loader)

    # log to wandb and print
    dist.barrier()
    if not args.nolog:
        _ = log_variable(epoch, "Epoch", commit=False)
        _ = log_variable(loss, "Val. loss", commit=False)
        if multitask:
            for task_name in classifier_loss[0].keys():
                _ = log_variable(classifier_loss[0][task_name], f"Classifier val. loss ({task_name})", commit=False)
                _ = log_variable(classifier_acc1[0][task_name], f"Online val. acc. ({task_name})", commit=False)
            for task_name in classifier_loss[1].keys():
                _ = log_variable(classifier_loss[1][task_name], f"Classifier val. loss ({task_name})", commit=False)
                _ = log_variable(classifier_acc1[1][task_name], f"Online val. acc. ({task_name})", commit=False)
            for task_name in regression_loss[0].keys():
                _ = log_variable(regression_loss[0][task_name], f"Regression val. loss ({task_name})", commit=False)
                _ = log_variable(regression_r2[0][task_name], f"Regression val. R2 ({task_name})", commit=False)
            for task_name in regression_loss[1].keys():
                _ = log_variable(regression_loss[1][task_name], f"Regression val. loss ({task_name})", commit=False)
                _ = log_variable(regression_r2[1][task_name], f"Regression val. R2 ({task_name})", commit=False)
        else:
            _ = log_variable(classifier_loss, "Classifier val. loss", commit=False)
            val_acc = log_variable(classifier_acc1, "Online val. acc. ({})".format(args.online_task), commit=False)
        if classifier_acc5 is not None:
            _ = log_variable(classifier_acc5, "Top 5 online val. acc. ({})".format(args.online_task), commit=False)
        _ = log_variable(z_var, "Variance act. (val.)", commit=False)
        _ = log_variable(avg_var_over_time, "Avg. variance act. over time (val.)", commit=False)
        _ = log_variable(ctx_var, "Variance ctx. (val.)", commit=False)
        _ = log_variable(avg_var_over_time_ctx, "Avg. variance ctx. over time (val.)", commit=False)
        part_ratio_log = log_variable(part_ratio, "Part. ratio act. (val.)")
        _ = log_variable(part_ratio_context, "Part. ratio ctx. (val.)")
        if args.loss in ["inv", "ilvic"]:
            _ = log_variable(criterion.pull_loss_val, "Pull loss (val.)", commit=False)
            _ = log_variable(criterion.push_loss_val, "Push loss (val.)", commit=False)
            _ = log_variable(criterion.decorr_loss_val, "Decorrelation loss (val.)", commit=False)
    else:
        val_acc = dist_reduce_mean(classifier_acc1)
        part_ratio_log = dist_reduce_mean(part_ratio)
    
    if dist.get_rank() == 0:
        print("Epoch: {}".format(epoch))
        if multitask:
            for task_name in classifier_loss[0].keys():
                print("Val. acc. ({}): {:.4f}".format(task_name, classifier_acc1[0][task_name]))
            for task_name in classifier_loss[1].keys():
                print("Val. acc. ({}): {:.4f}".format(task_name, classifier_acc1[1][task_name]))
            for task_name in regression_loss[0].keys():
                print("Val. R2 ({}): {:.4f}".format(task_name, regression_r2[0][task_name]))
            for task_name in regression_loss[1].keys():
                print("Val. R2 ({}): {:.4f}".format(task_name, regression_r2[1][task_name]))
        else:
            print("Val. accuracy: {:.4f}".format(val_acc))
        print("Participation ratio: {:.4f}".format(part_ratio_log))


def greedy_online_eval(
        args,
        model, 
        readout, 
        test_loader, 
        criterion, 
        classifier_criterion,
        regression_criterion,
        device, 
        epoch, 
        model_output_idx,
        multitask=False, 
        num_classes_seq_labels=None, 
        num_classes_dense_labels=None,
        num_regression_targets=None,
        num_regression_dense_targets=None):
    
    n_areas = model.n_areas
    loss_per_layer = torch.zeros(n_areas)
    if args.loss == "inv":
        pull_loss_per_layer = torch.zeros(n_areas)
        push_loss_per_layer = torch.zeros(n_areas)
        decorr_loss_per_layer = torch.zeros(n_areas)
    if multitask:
        classifier_loss = [
            ({key: 0. for key in num_classes_seq_labels.keys()}, {key: 0. for key in num_classes_dense_labels.keys()})
            for _ in range(n_areas)
        ]
        classifier_acc1 = [
            ({key: 0. for key in num_classes_seq_labels.keys()}, {key: 0. for key in num_classes_dense_labels.keys()})
            for _ in range(n_areas)
        ]
        regression_loss = [
            ({key: 0. for key in num_regression_targets.keys()}, {key: 0. for key in num_regression_dense_targets.keys()})
            for _ in range(n_areas)
        ]
        regression_r2 = [
            ({key: 0. for key in num_regression_targets.keys()}, {key: 0. for key in num_regression_dense_targets.keys()})
            for _ in range(n_areas)
        ]
        classifier_acc5 = None
    else:
        classifier_loss = torch.zeros(n_areas)
        classifier_acc1 = torch.zeros(n_areas)
        regression_loss = torch.zeros(n_areas)
        regression_r2 = torch.zeros(n_areas)
        classifier_acc5 = torch.zeros(n_areas)
    part_ratio = torch.zeros(n_areas)
    part_ratio_context = torch.zeros(n_areas)
    vars_act = torch.zeros(n_areas)
    vars_ctx = torch.zeros(n_areas)
    for x, y in tqdm(test_loader, disable=not sys.stdout.isatty()):
        x = x.to(device)
        if multitask:
            y = tuple([yi.to(device) for yi in y])
            target_length = None
        else:
            y = y.to(device)
            target_length = y.shape[1] if len(y.shape) == 2 else y.shape[0]
        with torch.no_grad():
            model_output = model(x)
            z_list, ctx_list, pred_list = model_output
            for j in range(n_areas):
                criterion_inputs = get_criterion_input(z_list[j], ctx_list[j], pred_list[j], y, args.loss, args.online_input, args.prediction_target)
                loss_per_layer[j] += criterion(*criterion_inputs).item()
                if args.loss == "inv":
                    pull_loss_per_layer[j] += criterion.pull_loss_val
                    push_loss_per_layer[j] += criterion.push_loss_val
                    decorr_loss_per_layer[j] += criterion.decorr_loss_val

                classifier_loss_, classifier_acc1_, classifier_acc5_, regression_loss_, regression_r2_ = compute_readout_loss(
                    data=model_output[model_output_idx][j], 
                    labels=y, 
                    readout=readout[j], 
                    classifier_criterion=classifier_criterion, 
                    regression_criterion=regression_criterion,
                    target_length=target_length, 
                    downstream_input=args.online_input,
                    dense_prediction=args.dense_prediction,
                    single_readout=args.online_single_timestep_readout,
                    pred_steps=args.pred_steps,
                    task=args.online_task,
                    full_spatial_readout=args.online_full_spatial_readout,)
                
                if multitask:
                    seq2label_loss, seq2seq_loss = classifier_loss_
                    seq2label_acc, seq2seq_acc = classifier_acc1_
                    seq2label_regression_loss, seq2seq_regression_loss = regression_loss_
                    seq2label_r2, seq2seq_r2 = regression_r2_
                    if seq2label_loss:
                        for task_name, loss_val in seq2label_loss.items():
                            classifier_loss[j][0][task_name] += loss_val.item()/len(test_loader)
                        for task_name, acc_val in seq2label_acc.items():
                            classifier_acc1[j][0][task_name] += acc_val.item()/len(test_loader)
                    if seq2seq_loss:
                        for task_name, loss_val in seq2seq_loss.items():
                            classifier_loss[j][1][task_name] += loss_val.item()/len(test_loader)
                        for task_name, acc_val in seq2seq_acc.items():
                            classifier_acc1[j][1][task_name] += acc_val.item()/len(test_loader)
                    if seq2label_regression_loss:
                        for task_name, loss_val in seq2label_regression_loss.items():
                            regression_loss[j][0][task_name] += loss_val.item()/len(test_loader)
                        for task_name, r2_val in seq2label_r2.items():
                            regression_r2[j][0][task_name] += r2_val/len(test_loader)
                    if seq2seq_regression_loss:
                        for task_name, loss_val in seq2seq_regression_loss.items():
                            regression_loss[j][1][task_name] += loss_val.item()/len(test_loader)
                        for task_name, r2_val in seq2seq_r2.items():
                            regression_r2[j][1][task_name] += r2_val/len(test_loader)
                else:
                    classifier_loss[j] += classifier_loss_.item()/len(test_loader)
                    classifier_acc1[j] += classifier_acc1_.item()/len(test_loader)
                    classifier_acc5[j] += classifier_acc5_.item()/len(test_loader)

                part_ratio[j] += participation_ratio(z_list[j]).item() if not args.flatten_area_enc_output else 0.
                part_ratio_context[j] += participation_ratio(ctx_list[j]).item()
                if len(z_list[j].shape) == 5:
                    z_list[j] = z_list[j].mean(-1).mean(-1)
                if len(ctx_list[j].shape) == 5:
                    ctx_list[j] = ctx_list[j].mean(-1).mean(-1)
                vars_act[j] += torch.var(z_list[j]).item() / len(test_loader)
                vars_ctx[j] += torch.var(ctx_list[j]).item() / len(test_loader)
    
    loss_per_layer /= len(test_loader)
    part_ratio /= len(test_loader)
    part_ratio_context /= len(test_loader)

    dist.barrier()
    if not args.nolog:
        _ = log_variable(epoch, "Epoch", commit=False)
        _ = log_variable(loss_per_layer, "Val. loss per layer", commit=False)
        for j in range(n_areas):
            if multitask:
                for task_name in classifier_loss[j][0].keys():
                    _ = log_variable(classifier_loss[j][0][task_name], f"Classifier val. loss ({task_name}), area {j}", commit=False)
                    _ = log_variable(classifier_acc1[j][0][task_name], f"Online val. acc. ({task_name}), area {j}", commit=False)
                for task_name in classifier_loss[j][1].keys():
                    _ = log_variable(classifier_loss[j][1][task_name], f"Classifier val. loss ({task_name}), area {j}", commit=False)
                    _ = log_variable(classifier_acc1[j][1][task_name], f"Online val. acc. ({task_name}), area {j}", commit=False)
                for task_name in regression_loss[j][0].keys():
                    _ = log_variable(regression_loss[j][0][task_name], f"Regression val. loss ({task_name}), area {j}", commit=False)
                    _ = log_variable(regression_r2[j][0][task_name], f"Regression val. R2 ({task_name}), area {j}", commit=False)
                for task_name in regression_loss[j][1].keys():
                    _ = log_variable(regression_loss[j][1][task_name], f"Regression val. loss ({task_name}), area {j}", commit=False)
                    _ = log_variable(regression_r2[j][1][task_name], f"Regression val. R2 ({task_name}), area {j}", commit=False)
            else:
                _ = log_variable(classifier_loss[j], f"Classifier val. loss, area {j}", commit=False)
                val_acc = log_variable(classifier_acc1[j], "Online val. acc. ({}), area {}".format(args.online_task, j), commit=False)
            if classifier_acc5 is not None:
                _ = log_variable(classifier_acc5[j], "Top 5 online val. acc. ({}), area {}".format(args.online_task, j), commit=False)
            _ = log_variable(vars_act[j], f"Variance act. (val.), area {j}", commit=False)
            _ = log_variable(vars_ctx[j], f"Variance ctx. (val.), area {j}", commit=False)
            part_ratio_log = log_variable(part_ratio[j], f"Part. ratio act. (val.), area {j}")
            _ = log_variable(part_ratio_context[j], f"Part. ratio ctx. (val), area {j}")
            if args.loss == "inv":
                _ = log_variable(pull_loss_per_layer[j], f"Pull loss (val.), area {j}", commit=False)
                _ = log_variable(push_loss_per_layer[j], f"Push loss (val.), area {j}", commit=False)
                _ = log_variable(decorr_loss_per_layer[j], f"Decorrelation loss (val.), area {j}", commit=False)
    else:
        val_acc = dist_reduce_mean(classifier_acc1[j])
        part_ratio_log = dist_reduce_mean(part_ratio[j])

    if dist.get_rank() == 0:
        print(f"Epoch: {epoch}")
        for j in range(n_areas):
            print(f"Area {j}")
            if multitask:
                for task_name in classifier_loss[j][0].keys():
                    print("  Val. acc. ({}): {:.4f}".format(task_name, classifier_acc1[j][0][task_name]))
                for task_name in classifier_loss[j][1].keys():
                    print("  Val. acc. ({}): {:.4f}".format(task_name, classifier_acc1[j][1][task_name]))
                for task_name in regression_loss[j][0].keys():
                    print("  Val. R2 ({}): {:.4f}".format(task_name, regression_r2[j][0][task_name]))
                for task_name in regression_loss[j][1].keys():
                    print("  Val. R2 ({}): {:.4f}".format(task_name, regression_r2[j][1][task_name]))
            else:
                print("  Val. accuracy: {:.4f}".format(classifier_acc1[j]))
            print("  Participation ratio: {:.4f}".format(part_ratio[j]))