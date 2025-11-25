import argparse
import torch
import wandb
from tqdm.auto import tqdm

from repl.metrics.metrics import compute_r2
from repl.networks.networks import LinearReadout
from repl.networks.utils import additional_data_process
from repl.scripts.args import add_data_args, add_logging_args, add_offline_eval_args, add_reproducibility_args, add_validation_args
from repl.scripts.eval_utils import offline_linear_regression
from repl.scripts.utils import get_data_specs, prepare_data, seed_everything


def get_multitask_readout(readout_input_dim, num_classes, seq_len=32):
    readout = torch.nn.ModuleList()
    num_classes_seq_labels = num_classes[0]
    if num_classes_seq_labels is not None:
        for task_name in num_classes_seq_labels.keys():
            readout.append(LinearReadout(num_classes=num_classes_seq_labels[task_name], input_dim=readout_input_dim,
                                            seq_len=seq_len, concat_over_time=False, task_name=task_name,
                                            concat_over_batch=False))
    num_classes_dense_labels = num_classes[1]
    if num_classes_dense_labels is not None:
        for task_name in num_classes_dense_labels.keys():
            readout.append(LinearReadout(num_classes=num_classes_dense_labels[task_name], input_dim=readout_input_dim,
                                            seq_len=seq_len, concat_over_time=False, task_name=task_name,
                                            concat_over_batch=True))
    num_seq_regr_targets = num_classes[2]
    if num_seq_regr_targets is not None:
        for task_name in num_seq_regr_targets.keys():
            readout.append(LinearReadout(num_classes=num_seq_regr_targets[task_name], input_dim=readout_input_dim,
                                            seq_len=seq_len, concat_over_time=False, task_name=task_name,
                                            concat_over_batch=False))
    num_dense_regr_targets = num_classes[3]
    if num_dense_regr_targets is not None:
        for task_name in num_dense_regr_targets.keys():
            readout.append(LinearReadout(num_classes=num_dense_regr_targets[task_name], input_dim=readout_input_dim,
                                            seq_len=seq_len, concat_over_time=False, task_name=task_name,
                                            concat_over_batch=True))
    
    return readout


def get_readout_loss_and_metrics(
        data, labels, readout,
        classifier_criterion, regression_criterion, task
):
    multitask = task == "multitask"
    if multitask:
        classification_loss = ({}, {})
        classification_acc = ({}, {})
        regression_loss = ({}, {})
        regression_r2 = ({}, {})
        seq_labels, seq2seq_labels, regr_targets, dense_regr_targets, _ = labels
        num_seq_labels = num_dense_labels = 0
        num_regression_targets = num_regression_dense_targets = 0
        classifier_acc5 = None

        num_seq_labels = seq_labels.shape[-1] # will be 0 if seq_labels is empty
        for i in range(num_seq_labels):
            logits = readout[i](data.detach())
            classification_loss[0][readout[i].task_name] = classifier_criterion(logits, seq_labels[:, i])
            # filter out rows of seq_labels that are -1
            seq_labels_i = seq_labels[:, i]
            logits = logits[seq_labels_i != -1]
            seq_labels_i = seq_labels_i[seq_labels_i != -1]
            classification_acc[0][readout[i].task_name] = (logits.argmax(dim=1) == seq_labels_i).float().mean()
        
        num_dense_labels = seq2seq_labels.shape[-1]
        for i in range(num_dense_labels):
            logits = readout[i + num_seq_labels](data.detach())
            seq2seq_labels_i = seq2seq_labels.flatten(0,1)[:, i]
            classification_loss[1][readout[i + num_seq_labels].task_name] = classifier_criterion(logits, seq2seq_labels_i)
            # filter out rows of seq2seq_labels that are -1
            logits = logits[seq2seq_labels_i != -1]
            seq2seq_labels_i = seq2seq_labels_i[seq2seq_labels_i != -1]
            classification_acc[1][readout[i + num_seq_labels].task_name] = (logits.argmax(dim=1) == seq2seq_labels_i).float().mean()

        num_regression_targets = regr_targets.shape[-1]
        for i in range(num_regression_targets):
            y_pred = readout[i + num_seq_labels + num_dense_labels](data.detach())
            regression_loss[0][readout[i + num_seq_labels + num_dense_labels].task_name] = regression_criterion(y_pred, regr_targets[:, i])
            regression_r2[0][readout[i + num_seq_labels + num_dense_labels].task_name] = compute_r2(regr_targets[:, i], y_pred)

        num_regression_dense_targets = dense_regr_targets.shape[-1]
        for i in range(num_regression_dense_targets):
            y_pred = readout[i + num_seq_labels + num_dense_labels + num_regression_targets](data.detach())
            dense_regr_targets_i = dense_regr_targets.flatten(0,1)[:, i]
            regression_loss[1][readout[i + num_seq_labels + num_dense_labels + num_regression_targets].task_name] = regression_criterion(y_pred, dense_regr_targets_i)
            regression_r2[1][readout[i + num_seq_labels + num_dense_labels + num_regression_targets].task_name] = compute_r2(dense_regr_targets_i, y_pred)
    else:
        y_pred = readout(data.detach())
        classification_loss = classifier_criterion(y_pred, labels)
        y_pred = y_pred[labels != -1]
        labels = labels[labels != -1]
        classification_acc = (y_pred.argmax(dim=1) == labels).float().mean()

        num_classes = y_pred.shape[-1]
        if num_classes < 5:
            classifier_acc5 = 1.0
        else:
            top5 = y_pred.topk(5, dim=1)[1]
            classifier_acc5 = (top5 == labels.unsqueeze(1)).any(dim=1).float().mean()

    return classification_loss, classification_acc, classifier_acc5, regression_loss, regression_r2


def validate_readout(readout, data_loader, 
                     classifier_criterion, regression_criterion,
                     device, epoch, task, 
                     num_classes_seq_labels=None, num_classes_dense_labels=None,
                     num_regression_targets=None, num_regression_dense_targets=None,
                     wandb_log=True):
    readout.eval()
    multitask = task == "multitask"

    if multitask:
        clf_val_loss = ({key: 0. for key in num_classes_seq_labels.keys()}, {key: 0. for key in num_classes_dense_labels.keys()})
        clf_val_acc = ({key: 0. for key in num_classes_seq_labels.keys()}, {key: 0. for key in num_classes_dense_labels.keys()})
        regr_val_loss = ({key: 0. for key in num_regression_targets.keys()}, {key: 0. for key in num_regression_dense_targets.keys()})
        regr_val_r2 = ({key: 0. for key in num_regression_targets.keys()}, {key: 0. for key in num_regression_dense_targets.keys()})
    else:
        clf_val_loss = 0.0
        clf_val_acc = 0.0
        clf_val_acc5 = 0.0

    with torch.no_grad():
        for x, y in tqdm(data_loader):
            x = x.to(device)
            if x.ndim > 3:
                x = x.flatten(start_dim=2)
            if multitask:
                y = tuple([yi.to(device) for yi in y])
            else:
                y = y.to(device)

            classifier_loss, val_acc, val_acc5, regression_loss, val_r2 = get_readout_loss_and_metrics(
                data=x, labels=y, readout=readout,
                classifier_criterion=classifier_criterion,
                regression_criterion=regression_criterion,
                task=task,
            )

            if multitask:
                # classifier_loss is a tuple of four dicts
                if num_classes_seq_labels is not None:
                    for task_name in num_classes_seq_labels.keys():
                        clf_val_loss[0][task_name] += classifier_loss[0][task_name] / len(data_loader)
                        clf_val_acc[0][task_name] += val_acc[0][task_name] / len(data_loader)
                if num_classes_dense_labels is not None:
                    for task_name in num_classes_dense_labels.keys():
                        clf_val_loss[1][task_name] += classifier_loss[1][task_name] / len(data_loader)
                        clf_val_acc[1][task_name] += val_acc[1][task_name] / len(data_loader)
                if num_regression_targets is not None:
                    for task_name in num_regression_targets.keys():
                        regr_val_loss[0][task_name] += regression_loss[0][task_name] / len(data_loader)
                        regr_val_r2[0][task_name] += val_r2[0][task_name] / len(data_loader)
                if num_regression_dense_targets is not None:
                    for task_name in num_regression_dense_targets.keys():
                        regr_val_loss[1][task_name] += regression_loss[1][task_name] / len(data_loader)
                        regr_val_r2[1][task_name] += val_r2[1][task_name] / len(data_loader)
            else:
                clf_val_loss += classifier_loss / len(data_loader)
                clf_val_acc += val_acc / len(data_loader)
                clf_val_acc5 += val_acc5 / len(data_loader) if val_acc5 is not None else 0.0

    if wandb_log:
        wandb.log({"Offline Epoch": epoch}, commit=False)
        if multitask:
            if num_classes_seq_labels is not None:
                for task_name in num_classes_seq_labels.keys():
                    wandb.log({f"Offline val. loss ({task_name})": clf_val_loss[0][task_name]}, commit=False)
                    wandb.log({f"Offline val. acc. ({task_name})": clf_val_acc[0][task_name]}, commit=False)
            if num_classes_dense_labels is not None:
                for task_name in num_classes_dense_labels.keys():
                    wandb.log({f"Offline val. loss ({task_name})": clf_val_loss[1][task_name]}, commit=False)
                    wandb.log({f"Offline val. acc. ({task_name})": clf_val_acc[1][task_name]}, commit=False)
            if num_regression_targets is not None:
                for task_name in num_regression_targets.keys():
                    wandb.log({f"Offline val. loss ({task_name})": regr_val_loss[0][task_name]}, commit=False)
                    wandb.log({f"Offline val. R2 ({task_name})": regr_val_r2[0][task_name]}, commit=False)
            if num_regression_dense_targets is not None:
                for task_name in num_regression_dense_targets.keys():
                    wandb.log({f"Offline val. loss ({task_name})": regr_val_loss[1][task_name]}, commit=False)
                    wandb.log({f"Offline val. R2 ({task_name})": regr_val_r2[1][task_name]}, commit=False)
        else:
            wandb.log({"Offline val. loss": clf_val_loss}, commit=False)
            wandb.log({f"Offline val. acc. ({task})": clf_val_acc}, commit=False)
            wandb.log({f"Top 5 offline val. acc. ({task})": clf_val_acc5}, commit=False)
    
    print(f"Epoch {epoch} validation results:")
    if multitask:
        if num_classes_seq_labels is not None:
            for task_name in num_classes_seq_labels.keys():
                print(f"  {task_name} - Loss: {clf_val_loss[0][task_name]:.4f}, Acc: {clf_val_acc[0][task_name]:.4f}")
        if num_classes_dense_labels is not None:
            for task_name in num_classes_dense_labels.keys():
                print(f"  {task_name} - Loss: {clf_val_loss[1][task_name]:.4f}, Acc: {clf_val_acc[1][task_name]:.4f}")
        if num_regression_targets is not None:
            for task_name in num_regression_targets.keys():
                print(f"  {task_name} - Loss: {regr_val_loss[0][task_name]:.4f}, R2: {regr_val_r2[0][task_name]:.4f}")
        if num_regression_dense_targets is not None:
            for task_name in num_regression_dense_targets.keys():
                print(f"  {task_name} - Loss: {regr_val_loss[1][task_name]:.4f}, R2: {regr_val_r2[1][task_name]:.4f}")
    else:
        print(f"  Loss: {clf_val_loss:.4f}, Acc: {clf_val_acc:.4f}, Top 5 Acc: {clf_val_acc5:.4f}")


def main(args, device):
    input_size, num_classes = get_data_specs(
        dataset=args.dataset,
        target_label=args.offline_task,
        mnist_seqtype=args.mnist_seqtype,
        spritevid_num_sprites=args.spritevid_max_sprites,
        flatten_images=args.flatten_images,
    )
    multitask = args.offline_task == "multitask"

    num_classes_seq_labels = num_classes[0] if multitask else None
    num_classes_dense_labels = num_classes[1] if multitask else None
    num_regression_targets = num_classes[2] if multitask else None
    num_regression_dense_targets = num_classes[3] if multitask else None

    seq_len = args.seq_len

    (train_loader, train_sampler,
     val_loader, val_sampler,
     test_loader, test_sampler) = prepare_data(
        args.dataset,
        args.data_input_dir,
        seq_len=seq_len,
        batch_size=args.offline_batch_size,
        val_batch_size=args.val_batch_size,
        distributed=False,
        num_workers=args.num_workers,
        grayscale=args.grayscale,
        target_label=args.offline_task,
        mnist_seqtype=args.mnist_seqtype,
        spritevid_max_sprites=args.spritevid_max_sprites,
        spritevid_exclude_latent_regions=args.spritevid_exclude_latent_regions,
        spritevid_discretize_latents=args.spritevid_discretize_latents,
        spritevid_noise_type=args.spritevid_noise_type,
        spritevid_noise_level=args.spritevid_noise_level,
        spritevid_frozen_noise=args.spritevid_frozen_noise,
        sprite_noise_on_top=args.sprite_noise_on_top,
        spritevid_grid_enabled=args.spritevid_grid_enabled,
        spritevid_frozen_grid=args.spritevid_frozen_grid,
        spritevid_occlude_n_frames=args.spritevid_occlude_n_frames,
        num_sequences=args.num_sequences,
     )
    
    if args.val_size == 0:
        val_loader = test_loader
        val_sampler = test_sampler

    n_in_channels = 1 if args.grayscale else 3

    if multitask:
        readout = get_multitask_readout(
            readout_input_dim=input_size * n_in_channels,
            num_classes=num_classes,
        )
    else:
        readout = LinearReadout(num_classes=num_classes, input_dim=input_size * n_in_channels,
                         seq_len=seq_len, concat_over_time=False, concat_over_batch=(args.offline_task == "seq2seq"))
        
    if args.use_sklearn_regression:
        linear_regression_output = offline_linear_regression(
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            model=None,
            num_classes=num_classes,
            downstream_input="raw",
            device=device,
        )
        if not args.nolog:
            if num_regression_targets is not None:
                for i in range(len(num_regression_targets)):
                    task_name = readout[i + len(num_classes_seq_labels) + len(num_classes_dense_labels)].task_name
                    _ = wandb.log({f"Offline sklearn regression train R2 ({task_name})": linear_regression_output["seq_r2_score_train"][i]}, commit=False)
                    _ = wandb.log({f"Offline sklearn regression test R2 ({task_name})": linear_regression_output["seq_r2_score_test"][i]}, commit=False)
                    _ = wandb.log({f"Offline sklearn regression train loss ({task_name})": linear_regression_output["seq_regression_loss_train"][i]}, commit=False)
                    _ = wandb.log({f"Offline sklearn regression test loss ({task_name})": linear_regression_output["seq_regression_loss_test"][i]}, commit=True)
            if num_regression_dense_targets is not None:
                for i in range(len(num_regression_dense_targets)):
                    task_name = readout[i + len(num_classes_seq_labels) + len(num_classes_dense_labels) + len(num_regression_targets)].task_name
                    _ = wandb.log({f"Offline sklearn regression dense train R2 ({task_name})": linear_regression_output["dense_r2_score_train"][i]}, commit=False)
                    _ = wandb.log({f"Offline sklearn regression dense test R2 ({task_name})": linear_regression_output["dense_r2_score_test"][i]}, commit=False)
                    _ = wandb.log({f"Offline sklearn regression dense train loss ({task_name})": linear_regression_output["dense_regression_loss_train"][i]}, commit=False)
                    _ = wandb.log({f"Offline sklearn regression dense test loss ({task_name})": linear_regression_output["dense_regression_loss_test"][i]}, commit=True)
    
        if args.save_offline_readout:
            seq_regressor = linear_regression_output["seq_regressor"]
            dense_regressor = linear_regression_output["dense_regressor"]
            # save the sklearn models
            sklearn_path = f"{args.checkpoint_dir}/sklearn_regressors.pt"
            torch.save({
                "seq_regressor": seq_regressor,
                "dense_regressor": dense_regressor,
            }, sklearn_path)

        # delete the regressors to free up memory
        del linear_regression_output["seq_regressor"]
        del linear_regression_output["dense_regressor"]

    classifier_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    regression_criterion = torch.nn.MSELoss()
    if args.offline_optimizer == "adam":
        optimizer = torch.optim.AdamW(
            readout.parameters(), lr=args.offline_lr, weight_decay=args.offline_weight_decay
        )
    elif args.offline_optimizer == "sgd":
        optimizer = torch.optim.SGD(
            readout.parameters(), lr=args.offline_lr, weight_decay=args.offline_weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {args.offline_optimizer} not implemented")

    readout = readout.to(device)
    classifier_criterion = classifier_criterion.to(device)
    regression_criterion = regression_criterion.to(device)

    for epoch in range(args.offline_epochs):
        validate_readout(readout, val_loader, classifier_criterion, regression_criterion,
                         device, epoch, args.offline_task,
                         num_classes_seq_labels=num_classes_seq_labels,
                         num_classes_dense_labels=num_classes_dense_labels,
                         num_regression_targets=num_regression_targets,
                         num_regression_dense_targets=num_regression_dense_targets,
                         wandb_log=not args.nolog)
        
        train_acc = ({}, {}) if multitask else 0.0
        train_r2 = ({}, {}) if multitask else 0.0

        for x, y in tqdm(train_loader):
            readout.train()
            x = x.to(device)
            if x.ndim > 3:
                x = x.flatten(start_dim=2)
            if multitask:
                y = tuple([yi.to(device) for yi in y])
                target_length = None
            else:
                y = y.to(device)
                target_length = y.shape[1] if len(y.shape) == 2 else y.shape[0]

            classifier_loss, train_acc_, _, regression_loss, train_r2_ = get_readout_loss_and_metrics(
                data=x, labels=y, readout=readout,
                classifier_criterion=classifier_criterion,
                regression_criterion=regression_criterion,
                task=args.offline_task,
            )

            loss = 0.
            if multitask:
                # classifier_loss is a tuple of four dicts
                if num_classes_seq_labels is not None:
                    for task_name in num_classes_seq_labels.keys():
                        loss += classifier_loss[0][task_name]
                        train_acc[0][task_name] = train_acc[0].get(task_name, 0.) + train_acc_[0].get(task_name, 0.) / len(train_loader)
                if num_classes_dense_labels is not None:
                    for task_name in num_classes_dense_labels.keys():
                        loss += classifier_loss[1][task_name]
                        train_acc[1][task_name] = train_acc[1].get(task_name, 0.) + train_acc_[1].get(task_name, 0.) / len(train_loader)
                if num_regression_targets is not None:
                    for task_name in num_regression_targets.keys():
                        loss += regression_loss[0][task_name]
                        train_r2[0][task_name] = train_r2[0].get(task_name, 0.) + train_r2_[0].get(task_name, 0.) / len(train_loader)
                if num_regression_dense_targets is not None:
                    for task_name in num_regression_dense_targets.keys():
                        loss += regression_loss[1][task_name]
                        train_r2[1][task_name] = train_r2[1].get(task_name, 0.) + train_r2_[1].get(task_name, 0.) / len(train_loader)
            else:
                loss = classifier_loss
                train_acc += train_acc_ / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not args.nolog:
                wandb.log({"Offline classifier train loss": loss.item()}, commit=True)

        if not args.nolog:
            if multitask:
                if num_classes_seq_labels is not None:
                    for task_name in num_classes_seq_labels.keys():
                        wandb.log({f"Offline classifier train accuracy ({task_name})": train_acc[0][task_name]}, commit=False)
                if num_classes_dense_labels is not None:
                    for task_name in num_classes_dense_labels.keys():
                        wandb.log({f"Offline classifier train accuracy ({task_name})": train_acc[1][task_name]}, commit=False)
                if num_regression_targets is not None:
                    for task_name in num_regression_targets.keys():
                        wandb.log({f"Offline regression train R2 ({task_name})": train_r2[0][task_name]}, commit=False)
                if num_regression_dense_targets is not None:
                    for task_name in num_regression_dense_targets.keys():
                        wandb.log({f"Offline regression train R2 ({task_name})": train_r2[1][task_name]}, commit=False)
            else:
                wandb.log({f"Offline classifier train accuracy ({args.offline_task})": train_acc}, commit=False)

    validate_readout(readout, val_loader, classifier_criterion, regression_criterion,
                     device, epoch, args.offline_task,
                     num_classes_seq_labels=num_classes_seq_labels,
                     num_classes_dense_labels=num_classes_dense_labels,
                     num_regression_targets=num_regression_targets,
                     num_regression_dense_targets=num_regression_dense_targets,
                     wandb_log=not args.nolog)
    
    if args.save_offline_readout:
        readout_path = f"{args.checkpoint_dir}/readout_raw_data_{args.dataset}.pt"
        torch.save(readout.state_dict(), readout_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_reproducibility_args(parser)
    add_logging_args(parser)
    add_data_args(parser)
    add_validation_args(parser)
    add_offline_eval_args(parser)
    args = parser.parse_args()

    seed_everything(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.nolog:
        wandb.init(project="RPL", name=args.experiment_name, config=args)

    main(args, device)