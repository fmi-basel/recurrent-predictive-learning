import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm.auto import tqdm
import wandb

from repl.data.utils import StatsRecorder
from repl.networks.utils import additional_data_process, prepare_model
from repl.scripts.utils import (
    get_data_specs,
    init_distributed, 
    prepare_data, 
    seed_everything,
    dist_reduce_mean, 
    log_variable,
)
from repl.scripts.eval_utils import compute_readout_loss, offline_linear_regression, prepare_readout
from repl.scripts.args import (
    check_args,
    add_data_args, 
    add_ddp_args, 
    add_logging_args, 
    add_model_args, 
    add_offline_eval_args, 
    add_reproducibility_args, 
    add_validation_args, 
)


def validate_offline(args, 
                     model, 
                     readout, 
                     val_loader, 
                     classifier_criterion,
                     regression_criterion, 
                     device, 
                     epoch, 
                     model_output_idx, 
                     readout_input_mean, 
                     readout_input_std, 
                     multitask=False, 
                     num_classes_seq_labels=None, 
                     num_classes_dense_labels=None,
                     num_regression_targets=None,
                     num_regression_dense_targets=None):
    
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
        for x, y in tqdm(val_loader, disable=not sys.stdout.isatty()):
            x = x.to(device)
            if multitask:
                y = tuple([yi.to(device) for yi in y])
                target_length = None
            else:
                y = y.to(device)
                # if task is seq2seq classification (e.g. phone), y.shape = (B, L); otherwise y.shape = (B,)
                target_length = y.shape[1] if len(y.shape) == 2 else y.shape[0]

            readout_input = model(x)[model_output_idx]
            readout_input = (readout_input - readout_input_mean) / (readout_input_std + 1e-8)

            classifier_loss_, classifier_acc1_, classifier_acc5_, regression_loss_, regression_r2_ = compute_readout_loss(
                data=readout_input, 
                labels=y, 
                readout=readout, 
                classifier_criterion=classifier_criterion, 
                regression_criterion=regression_criterion,
                target_length=target_length, 
                downstream_input=args.offline_input,
                dense_prediction=args.dense_prediction,
                single_readout=args.offline_single_timestep_readout,
                pred_steps=args.pred_steps,
                task=args.offline_task,
                full_spatial_readout=args.offline_full_spatial_readout,
            )
            
            if multitask:
                # classifier_loss is a tuple of four dicts
                if num_classes_seq_labels is not None:
                    for task_name in num_classes_seq_labels.keys():
                        clf_val_loss[0][task_name] += classifier_loss_[0][task_name] / len(val_loader)
                        clf_val_acc[0][task_name] += classifier_acc1_[0][task_name] / len(val_loader)
                if num_classes_dense_labels is not None:
                    for task_name in num_classes_dense_labels.keys():
                        clf_val_loss[1][task_name] += classifier_loss_[1][task_name] / len(val_loader)
                        clf_val_acc[1][task_name] += classifier_acc1_[1][task_name] / len(val_loader)
                if num_regression_targets is not None:
                    for task_name in num_regression_targets.keys():
                        regr_val_loss[0][task_name] += regression_loss_[0][task_name] / len(val_loader)
                        regr_val_r2[0][task_name] += regression_r2_[0][task_name] / len(val_loader)
                if num_regression_dense_targets is not None:
                    for task_name in num_regression_dense_targets.keys():
                        regr_val_loss[1][task_name] += regression_loss_[1][task_name] / len(val_loader)
                        regr_val_r2[1][task_name] += regression_r2_[1][task_name] / len(val_loader)
            else:
                clf_val_loss += classifier_loss_ / len(val_loader)
                clf_val_acc += classifier_acc1_ / len(val_loader)
                clf_val_acc5 += classifier_acc5_ / len(val_loader)
    
    dist.barrier()
    if not args.nolog:
        _ = log_variable(epoch, "Offline Epoch", commit=False)
        if multitask:
            for task_name in num_classes_seq_labels.keys():
                _ = log_variable(clf_val_loss[0][task_name], f"Offline val. loss ({task_name})", commit=False)
                _ = log_variable(clf_val_acc[0][task_name], f"Offline val. acc. ({task_name})", commit=False)
            for task_name in num_classes_dense_labels.keys():
                _ = log_variable(clf_val_loss[1][task_name], f"Offline val. loss ({task_name})", commit=False)
                _ = log_variable(clf_val_acc[1][task_name], f"Offline val. acc. ({task_name})", commit=False)
            for task_name in num_regression_targets.keys():
                _ = log_variable(regr_val_loss[0][task_name], f"Offline val. loss ({task_name})", commit=False)
                _ = log_variable(regr_val_r2[0][task_name], f"Offline val. R2 ({task_name})", commit=False)
            for task_name in num_regression_dense_targets.keys():
                _ = log_variable(regr_val_loss[1][task_name], f"Offline val. loss ({task_name})", commit=False)
                _ = log_variable(regr_val_r2[1][task_name], f"Offline val. R2 ({task_name})", commit=False)
        else:
            _ = log_variable(clf_val_loss, "Offline val. loss", commit=False)
            clf_val_acc = log_variable(clf_val_acc, "Offline val. acc. ({})".format(args.offline_task), commit=False)
            _ = log_variable(clf_val_acc5, "Top 5 offline val. acc. ({})".format(args.offline_task), commit=False)
    else:
        if multitask:
            for task_name in num_classes_seq_labels.keys():
                clf_val_acc[0][task_name] = dist_reduce_mean(clf_val_acc[0][task_name])
            for task_name in num_classes_dense_labels.keys():
                clf_val_acc[1][task_name] = dist_reduce_mean(clf_val_acc[1][task_name])
            for task_name in num_regression_targets.keys():
                regr_val_r2[0][task_name] = dist_reduce_mean(regr_val_r2[0][task_name])
            for task_name in num_regression_dense_targets.keys():
                regr_val_r2[1][task_name] = dist_reduce_mean(regr_val_r2[1][task_name])
        else:
            clf_val_acc = dist_reduce_mean(clf_val_acc)

    if dist.get_rank() == 0:
        print("Epoch: {}".format(epoch))
        if multitask:
            for task_name in num_classes_seq_labels.keys():
                print("Val. acc. ({}): {:.4f}".format(task_name, clf_val_acc[0][task_name]))
            for task_name in num_classes_dense_labels.keys():
                print("Val. acc. ({}): {:.4f}".format(task_name, clf_val_acc[1][task_name]))
            for task_name in num_regression_targets.keys():
                print("Val. R2 ({}): {:.4f}".format(task_name, regr_val_r2[0][task_name]))
            for task_name in num_regression_dense_targets.keys():
                print("Val. R2 ({}): {:.4f}".format(task_name, regr_val_r2[1][task_name]))
        else:
            print("Val. acc. ({}): {:.4f}".format(args.offline_task, clf_val_acc))


def get_readout_input_stats(loader, model, model_output_idx, device):
    stats = StatsRecorder(red_dims=(0, 1))
    for x, _ in tqdm(loader, disable=not sys.stdout.isatty()):
        x = x.to(device)
        with torch.no_grad():
            readout_input = model(x)[model_output_idx]
        stats.update(readout_input)
    return stats.mean, stats.std


def offline_eval(args, model, num_classes, model_output_idx, seq_len, device, input_size, checkpoint_dir=None):
    model.eval()
    (train_loader, train_sampler, 
        val_loader, val_sampler,
        test_loader, test_sampler,
    ) = prepare_data(
            args.dataset,
            args.data_input_dir,
            seq_len=seq_len,
            batch_size=args.offline_batch_size,
            val_batch_size=args.offline_batch_size,
            distributed=args.distributed,
            num_workers=args.num_workers,
            grayscale=args.grayscale,
            target_label=args.offline_task,
            mnist_seqtype=args.mnist_seqtype,
            spritevid_max_sprites=args.spritevid_max_sprites,
            spritevid_exclude_latent_regions=False, # we make sure to train the readout on the full latent space
            spritevid_discretize_latents=args.spritevid_discretize_latents,
            spritevid_noise_type=args.spritevid_noise_type,
            spritevid_noise_level=args.spritevid_noise_level,
            spritevid_frozen_noise=args.spritevid_frozen_noise,
            sprite_noise_on_top=args.sprite_noise_on_top,
            spritevid_grid_enabled=args.spritevid_grid_enabled,
            spritevid_frozen_grid=args.spritevid_frozen_grid,
            spritevid_occlude_n_frames=args.spritevid_occlude_n_frames,
            num_sequences=args.num_sequences,
            inter_trial_interval=args.inter_trial_interval,
        )

    multitask = args.offline_task == "multitask"


    num_classes_seq_labels = num_classes[0] if multitask else None
    num_classes_dense_labels = num_classes[1] if multitask else None
    num_regression_targets = num_classes[2] if multitask else None
    num_regression_dense_targets = num_classes[3] if multitask else None


    readout_input_mean_train, readout_input_std_train = get_readout_input_stats(train_loader, model, model_output_idx, device)
    
    readout = prepare_readout(
        task=args.offline_task,
        downstream_input=args.offline_input,
        input_spatial_size=input_size,
        single_timestep_readout=args.offline_single_timestep_readout,
        full_spatial_readout=args.offline_full_spatial_readout,
        num_classes=num_classes,
        seq_len=seq_len,
        evaluate_concat_features=args.evaluate_concat_features,
        enc_output_dim=args.enc_output_dim,
        ctx_dim=args.ctx_dim,
        pred_steps=args.pred_steps,
        dense_prediction=args.dense_prediction,
        prediction_target=args.prediction_target,
        pred_target_dim_override=args.pred_target_dim_override,
        model=model,
    )

    classifier_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    regression_criterion = nn.MSELoss()
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
    if args.distributed:
        readout = nn.parallel.DistributedDataParallel(readout, device_ids=[device], output_device=device)
    classifier_criterion = classifier_criterion.to(device)
    regression_criterion = regression_criterion.to(device)

    if args.offline_task == "multitask":
        if args.use_sklearn_regression:
            dist.barrier()
            print("Using sklearn for regression tasks")
            linear_regression_output = offline_linear_regression(train_dataloader=train_loader,
                                                                  test_dataloader=test_loader,
                                                                  model=model,
                                                                  num_classes=num_classes,
                                                                  downstream_input=args.offline_input,
                                                                  device=device,)
            if not args.nolog:
                for i in range(len(num_regression_targets)):
                    task_name = readout[i + len(num_classes_seq_labels) + len(num_classes_dense_labels)].task_name
                    _ = log_variable(linear_regression_output["seq_r2_score_train"][i], f"Offline sklearn regression train R2 ({task_name})", commit=False)
                    _ = log_variable(linear_regression_output["seq_r2_score_test"][i], f"Offline sklearn regression test R2 ({task_name})", commit=False)
                    _ = log_variable(linear_regression_output["seq_regression_loss_train"][i], f"Offline sklearn regression train loss ({task_name})", commit=False)
                    _ = log_variable(linear_regression_output["seq_regression_loss_test"][i], f"Offline sklearn regression test loss ({task_name})", commit=True)
                for i in range(len(num_regression_dense_targets)):
                    task_name = readout[i + len(num_classes_seq_labels) + len(num_classes_dense_labels) + len(num_regression_targets)].task_name
                    _ = log_variable(linear_regression_output["dense_r2_score_train"][i], f"Offline sklearn regression dense train R2 ({task_name})", commit=False)
                    _ = log_variable(linear_regression_output["dense_r2_score_test"][i], f"Offline sklearn regression dense test R2 ({task_name})", commit=False)
                    _ = log_variable(linear_regression_output["dense_regression_loss_train"][i], f"Offline sklearn regression dense train loss ({task_name})", commit=False)
                    _ = log_variable(linear_regression_output["dense_regression_loss_test"][i], f"Offline sklearn regression dense test loss ({task_name})", commit=True)

    for epoch in range(args.offline_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)

        validate_offline(args, 
                         model=model, 
                         readout=readout, 
                         val_loader=test_loader, 
                         classifier_criterion=classifier_criterion, 
                         regression_criterion=regression_criterion,
                         device=device, 
                         epoch=epoch, 
                         model_output_idx=model_output_idx, 
                         readout_input_mean=readout_input_mean_train,
                         readout_input_std=readout_input_std_train,
                         multitask=multitask, 
                         num_classes_seq_labels=num_classes_seq_labels, 
                         num_classes_dense_labels=num_classes_dense_labels,
                         num_regression_targets=num_regression_targets,
                         num_regression_dense_targets=num_regression_dense_targets)

        train_acc = ({}, {}) if multitask else 0.
        train_r2 = ({}, {}) if multitask else 0.
        for x, y in tqdm(train_loader, disable=not sys.stdout.isatty()):
            x = x.to(device)
            if multitask:
                y = tuple([yi.to(device) for yi in y])
                target_length = None
            else:
                y = y.to(device)
                # if task is seq2seq classification (e.g. phone), y.shape = (B, L); otherwise y.shape = (B,)
                target_length = y.shape[1] if len(y.shape) == 2 else y.shape[0]

            with torch.no_grad():
                readout_input = model(x)[model_output_idx]
            readout_input = (readout_input - readout_input_mean_train) / (readout_input_std_train + 1e-8)

            classifier_loss, train_acc_, _, regression_loss, train_r2 = compute_readout_loss(
                data=readout_input,
                labels=y,
                readout=readout,
                classifier_criterion=classifier_criterion,
                regression_criterion=regression_criterion,
                target_length=target_length,
                downstream_input=args.offline_input,
                dense_prediction=args.dense_prediction,
                single_readout=args.offline_single_timestep_readout,
                pred_steps=args.pred_steps,
                task=args.offline_task,
                full_spatial_readout=args.offline_full_spatial_readout,
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
                        train_r2[0][task_name] = train_r2[0].get(task_name, 0.) + train_r2[0].get(task_name, 0.) / len(train_loader)
                if num_regression_dense_targets is not None:
                    for task_name in num_regression_dense_targets.keys():
                        loss += regression_loss[1][task_name]
                        train_r2[1][task_name] = train_r2[1].get(task_name, 0.) + train_r2[1].get(task_name, 0.) / len(train_loader)
            else:
                loss = classifier_loss
                train_acc += train_acc_ / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not args.nolog:
                dist.barrier()
                _ = log_variable(loss.item(), "Offline classifier train loss")
                
        if not args.nolog:
            dist.barrier()
            if multitask:
                for task_name in num_classes_seq_labels.keys():
                    _ = log_variable(train_acc[0][task_name], f"Offline classifier train accuracy ({task_name})", commit=False)
                for task_name in num_classes_dense_labels.keys():
                    _ = log_variable(train_acc[1][task_name], f"Offline classifier train accuracy ({task_name})", commit=False)
                for task_name in num_regression_targets.keys():
                    _ = log_variable(train_r2[0][task_name], f"Offline regression train R2 ({task_name})", commit=False)
                for task_name in num_regression_dense_targets.keys():
                    _ = log_variable(train_r2[1][task_name], f"Offline regression train R2 ({task_name})", commit=False)
            else:
                _ = log_variable(train_acc, f"Offline classifier train accuracy ({args.offline_task})")
    

    validate_offline(args, 
                     model=model, 
                     readout=readout, 
                     val_loader=test_loader,
                     classifier_criterion=classifier_criterion, 
                     regression_criterion=regression_criterion,
                     device=device, 
                     epoch=epoch+1, 
                     model_output_idx=model_output_idx, 
                     readout_input_mean=readout_input_mean_train,
                     readout_input_std=readout_input_std_train,
                     multitask=multitask, 
                     num_classes_seq_labels=num_classes_seq_labels, 
                     num_classes_dense_labels=num_classes_dense_labels,
                     num_regression_targets=num_regression_targets,
                     num_regression_dense_targets=num_regression_dense_targets)
    
    if dist.get_rank() == 0 and args.save_offline_readout:
        readout_path = os.path.join(checkpoint_dir, f"offline_{args.offline_input}_readout.pt")
        torch.save(readout.state_dict(), readout_path)

        # save the model output mean and std
        readout_stats_path = os.path.join(checkpoint_dir, f"offline_{args.offline_input}_readout_stats.pt")
        torch.save({"mean": readout_input_mean_train, "std": readout_input_std_train}, readout_stats_path)

    model.train()


def main(args, device):
    if args.distributed:
        rank = dist.get_rank()
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else torch.cpu.device_count()
        device_id = rank % device_count
        if args.distribute_data:
            args.batch_size //= device_count
            args.offline_batch_size //= device_count
        device = device_id

    input_size, num_classes = get_data_specs(args.dataset, 
                                             target_label=args.offline_task,
                                             mnist_seqtype=args.mnist_seqtype,
                                             spritevid_num_sprites=args.spritevid_max_sprites,
                                             flatten_images=args.flatten_images,)
    
    seq_len = args.seq_len

    preprocess, postprocess = additional_data_process(args.dataset, args.flatten_enc_output)
    n_in_channels = 1 if args.grayscale else 3

    state_dict = torch.load(args.model_path) if args.model_path is not None else None
    model = prepare_model(
        encoder_kind=args.encoder,
        integrator_kind=args.integrator,
        predictor_kind=args.predictor,
        input_size=input_size,
        n_in_channels=n_in_channels,
        enc_dim=args.enc_output_dim,
        ctx_dim=args.ctx_dim,
        ctx_n_layers=args.ctx_n_layers,
        pred_n_hidden_layers=args.pred_n_hidden_layers,
        pred_hidden_dim=args.pred_hidden_dim,
        pred_steps=args.pred_steps,
        dense_prediction=args.dense_prediction,
        prediction_target=args.prediction_target,
        pred_target_dim_override=args.pred_target_dim_override,
        use_bn_enc=args.use_bn,
        enc_n_layers=args.enc_n_layers,
        enc_kernel_size=args.enc_kernel_size,
        enc_stride=args.enc_stride,
        enc_padding=args.enc_padding,
        enc_pool_size=args.enc_pool_size,
        resnet_n_channels=args.resnet_n_channels,
        resnet_n_blocks=args.resnet_n_blocks,
        resnet_pool=args.resnet_pool,
        resnet_stem_kernel_size=args.resnet_stem_kernel_size,
        preprocess=preprocess,
        postprocess=postprocess,
        state_dict=state_dict,
        return_full_features=args.offline_full_spatial_readout,
        flatten_enc_output=args.flatten_enc_output,
    )
    # if state_dict is not None:
    #     model.load_state_dict(state_dict)
    
    model = model.to(device)
    if args.distributed:
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)

    if args.offline_input == "enc":
        model_output_idx = 0
    elif args.offline_input == "ctx":
        model_output_idx = 1
    else:
        model_output_idx = 2
    
    # take args.model_path
    checkpoint_dir = os.path.dirname(args.model_path)

    offline_eval(args, model, num_classes, model_output_idx, seq_len, device, input_size, checkpoint_dir=checkpoint_dir)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline evaluation script")
    add_reproducibility_args(parser)
    add_logging_args(parser)
    add_data_args(parser)
    add_model_args(parser)
    add_validation_args(parser)
    add_offline_eval_args(parser)
    add_ddp_args(parser)
    args = parser.parse_args()

    if args.offline_task is None:
        raise ValueError("Please specify the task for offline evaluation")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = init_distributed(args.dist_backend, args.dist_url, args.local_rank)

    if args.distributed:
        if dist.get_rank() == 0 and not args.nolog:
            wandb.init(project="RPL", name=args.experiment_name, config=args)
    elif not args.nolog:
        wandb.init(project="RPL", name=args.experiment_name, config=args)
    
    check_args(args)
    main(args, device)
