import argparse
import os
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
from repl.scripts.eval_utils import compute_readout_loss, offline_greedy_linear_regression, prepare_readout
from repl.scripts.args import (
    check_args,
    add_data_args, 
    add_ddp_args, 
    add_logging_args, 
    add_greedy_model_args,
    add_offline_eval_args, 
    add_reproducibility_args, 
    add_validation_args, 
)


def greedy_offline_eval(args, model, num_classes, model_output_idx, seq_len, device, input_size, checkpoint_dir=None):
    """Offline evaluation for greedy (multi-area) models
    
    Args:
        args: Command line arguments
        model: Greedy model with multiple areas
        num_classes: Number of classes for classification tasks
        model_output_idx: Which model output to use (0=enc, 1=ctx, 2=pred)
        seq_len: Length of input sequences
        device: Device to run evaluation on
        input_size: Size of input images
        checkpoint_dir: Directory to save readout models
    """
    model.eval()
    n_areas = model.n_areas
    
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
        )

    multitask = args.offline_task == "multitask"

    num_classes_seq_labels = num_classes[0] if multitask else None
    num_classes_dense_labels = num_classes[1] if multitask else None
    num_regression_targets = num_classes[2] if multitask else None
    num_regression_dense_targets = num_classes[3] if multitask else None

    # Get stats for each area separately
    readout_input_mean_train = []
    readout_input_std_train = []
    
    area_stats = [StatsRecorder(red_dims=(0, 1)) for _ in range(n_areas)]
    for x, _ in tqdm(train_loader, desc=f"Computing stats"):
        x = x.to(device)
        with torch.no_grad():
            readout_inputs = model(x)[model_output_idx]
        for j in range(n_areas):
            area_stats[j].update(readout_inputs[j])
    for j in range(n_areas):
        readout_input_mean_train.append(area_stats[j].mean)
        readout_input_std_train.append(area_stats[j].std)
    
    # Prepare readouts for each area
    readout = prepare_readout(
        task=args.offline_task,
        downstream_input=args.offline_input,
        input_spatial_size=input_size,
        single_timestep_readout=args.offline_single_timestep_readout,
        full_spatial_readout=args.offline_full_spatial_readout,
        num_classes=num_classes,
        seq_len=seq_len,
        evaluate_concat_features=args.evaluate_concat_features,
        enc_output_dim=args.area_enc_dims,
        ctx_dim=args.area_ctx_dims,
        pred_steps=args.pred_steps,
        dense_prediction=args.dense_prediction,
        prediction_target=args.prediction_target,
        pred_target_dim_override=args.pred_target_dim_override,
        model=model,
        n_areas=n_areas,
    )
    readout = readout.to(device)

    classifier_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    regression_criterion = nn.MSELoss()
    
    # Create optimizer for each area's readout
    params_readout = []
    for name, param in readout.named_parameters():
        if param.requires_grad:
            params_readout.append(param)
    optimizer = torch.optim.Adam(params_readout, lr=args.offline_lr, weight_decay=args.offline_weight_decay)

    classifier_criterion = classifier_criterion.to(device)
    regression_criterion = regression_criterion.to(device)

    if args.offline_task == "multitask":
        if args.use_sklearn_regression:
            dist.barrier()
            print("Using sklearn regression for offline evaluation")
            linear_regression_output = offline_greedy_linear_regression(
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                model=model,
                num_classes=num_classes,
                downstream_input=args.offline_input,
                device=device,
            )

        if not args.nolog:
            for area in range(n_areas):
                for i in range(len(num_regression_targets)):
                    task_name = readout[area][i + len(num_classes_seq_labels) + len(num_classes_dense_labels)].task_name
                    _ = log_variable(
                        linear_regression_output["seq_r2_score_train"][area][i],
                        f"Offline sklearn regression train R2 ({task_name}), area {area}",
                        commit=False
                    )
                    _ = log_variable(
                        linear_regression_output["seq_r2_score_test"][area][i],
                        f"Offline sklearn regression test R2 ({task_name}), area {area}",
                        commit=False
                    )
                    _ = log_variable(
                        linear_regression_output["seq_regression_loss_train"][area][i],
                        f"Offline sklearn regression train loss ({task_name}), area {area}",
                        commit=False
                    )
                    _ = log_variable(
                        linear_regression_output["seq_regression_loss_test"][area][i],
                        f"Offline sklearn regression test loss ({task_name}), area {area}",
                        commit=True
                    )
                for i in range(len(num_regression_dense_targets)):
                    task_name = readout[area][i + len(num_classes_seq_labels) + len(num_classes_dense_labels) + len(num_regression_targets)].task_name
                    _ = log_variable(
                        linear_regression_output["dense_r2_score_train"][area][i],
                        f"Offline sklearn regression dense train R2 ({task_name}), area {area}",
                        commit=False
                    )
                    _ = log_variable(
                        linear_regression_output["dense_r2_score_test"][area][i],
                        f"Offline sklearn regression dense test R2 ({task_name}), area {area}",
                        commit=False
                    )
                    _ = log_variable(
                        linear_regression_output["dense_regression_loss_train"][area][i],
                        f"Offline sklearn regression dense train loss ({task_name}), area {area}",
                        commit=False
                    )
                    _ = log_variable(
                        linear_regression_output["dense_regression_loss_test"][area][i],
                        f"Offline sklearn regression dense test loss ({task_name}), area {area}",
                        commit=True
                    )

    # Training loop
    for epoch in range(args.offline_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)

        # Validate
        validate_greedy_offline(
            args, 
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
            num_regression_dense_targets=num_regression_dense_targets
        )
        
        for x, y in tqdm(train_loader):
            x = x.to(device)
            if multitask:
                y = tuple([yi.to(device) for yi in y])
                target_length = None
            else:
                y = y.to(device)
                # if task is seq2seq classification (e.g. phone), y.shape = (B, L); otherwise y.shape = (B,)
                target_length = y.shape[1] if len(y.shape) == 2 else y.shape[0]

            with torch.no_grad():
                readout_inputs = model(x)[model_output_idx]
            
            total_loss = 0.
            clf_loss_values = []
            reg_loss_values = []
            train_acc = []
            train_r2 = []
            loss_values = [0.0 for _ in range(n_areas)]
            for j in range(n_areas):
                readout_input = (readout_inputs[j] - readout_input_mean_train[j]) / (readout_input_std_train[j] + 1e-8)

                classifier_loss, train_acc_, _, regression_loss, train_r2_ = compute_readout_loss(
                    data=readout_input,
                    labels=y,
                    readout=readout[j],
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
                    clf_loss_values.append(classifier_loss)
                    reg_loss_values.append(regression_loss)
                    train_acc.append(train_acc_)
                    train_r2.append(train_r2_)
                    if num_classes_seq_labels is not None:
                        for task_name in num_classes_seq_labels.keys():
                            total_loss += classifier_loss[0][task_name]
                            loss_values[j] += classifier_loss[0][task_name].item()
                    if num_classes_dense_labels is not None:
                        for task_name in num_classes_dense_labels.keys():
                            total_loss += classifier_loss[1][task_name]
                            loss_values[j] += classifier_loss[1][task_name].item()
                    if num_regression_targets is not None:
                        for task_name in num_regression_targets.keys():
                            total_loss += regression_loss[0][task_name]
                            loss_values[j] += regression_loss[0][task_name].item()
                    if num_regression_dense_targets is not None:
                        for task_name in num_regression_dense_targets.keys():
                            total_loss += regression_loss[1][task_name]
                            loss_values[j] += regression_loss[1][task_name].item()
                else:
                    total_loss += classifier_loss
                    clf_loss_values.append(classifier_loss.item())
                    reg_loss_values.append(regression_loss.item())
                    train_acc.append(train_acc_)
                    train_r2.append(train_r2_)

                if not args.nolog:
                    dist.barrier()
                    _ = log_variable(loss_values[j], f"Offline classifier train loss, area {j}")

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        
        # Log training metrics
        if not args.nolog:
            dist.barrier()
            for j in range(n_areas):
                if multitask:
                    for task_name in num_classes_seq_labels.keys():
                        _ = log_variable(train_acc[j][0][task_name], f"Offline classifier train accuracy ({task_name}), area {j}", commit=False)
                    for task_name in num_classes_dense_labels.keys():
                        _ = log_variable(train_acc[j][1][task_name], f"Offline classifier train accuracy ({task_name}), area {j}", commit=False)
                    for task_name in num_regression_targets.keys():
                        _ = log_variable(train_r2[j][0][task_name], f"Offline regression train R2 ({task_name}), area {j}", commit=False)
                    for task_name in num_regression_dense_targets.keys():
                        _ = log_variable(train_r2[j][1][task_name], f"Offline regression train R2 ({task_name}), area {j}", commit=False)
                else:
                    _ = log_variable(train_acc[j], f"Offline classifier train accuracy ({args.offline_task}), area {j}")
    
    # Final validation
    validate_greedy_offline(
        args, 
        model=model, 
        readout=readout, 
        val_loader=test_loader,
        classifier_criterion=classifier_criterion, 
        regression_criterion=regression_criterion,
        device=device, 
        epoch=args.offline_epochs, 
        model_output_idx=model_output_idx, 
        readout_input_mean=readout_input_mean_train,
        readout_input_std=readout_input_std_train,
        multitask=multitask, 
        num_classes_seq_labels=num_classes_seq_labels, 
        num_classes_dense_labels=num_classes_dense_labels,
        num_regression_targets=num_regression_targets,
        num_regression_dense_targets=num_regression_dense_targets
    )
    
    # Save models if needed
    if dist.get_rank() == 0 and args.save_offline_readout:
        torch.save(readout.state_dict(), os.path.join(checkpoint_dir, f"offline_{args.offline_input}_readout.pt"))
        for j in range(n_areas):
            # save the model output mean and std
            readout_stats_path = os.path.join(checkpoint_dir, f"offline_{args.offline_input}_readout_stats_area{j}.pt")
            torch.save({"mean": readout_input_mean_train[j], "std": readout_input_std_train[j]}, readout_stats_path)

    model.train()


def validate_greedy_offline(args, 
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
    """Validate offline readouts for greedy models with multiple areas
    
    Args:
        args: Command line arguments
        model: Greedy model with multiple areas
        readout: List of readout networks, one for each area
        val_loader: Validation data loader
        classifier_criterion: Loss function for classification tasks
        regression_criterion: Loss function for regression tasks
        device: Device to run validation on
        epoch: Current training epoch
        model_output_idx: Which model output to use (0=enc, 1=ctx, 2=pred)
        readout_input_mean: Mean of readout inputs for normalization, per area
        readout_input_std: Std of readout inputs for normalization, per area
        multitask: Whether the classification task is multiclass
        num_classes_seq_labels: Number of classes for sequence labels
        num_classes_dense_labels: Number of classes for dense labels (seq2seq)
        num_regression_targets: Number of regression targets
        num_regression_dense_targets: Number of regression dense targets
    """
    n_areas = model.n_areas
    
    if multitask:
        clf_val_loss = [
            ({key: 0. for key in num_classes_seq_labels.keys()}, {key: 0. for key in num_classes_dense_labels.keys()})
            for _ in range(n_areas)
        ]
        clf_val_acc = [
            ({key: 0. for key in num_classes_seq_labels.keys()}, {key: 0. for key in num_classes_dense_labels.keys()})
            for _ in range(n_areas)
        ]
        regr_val_loss = [
            ({key: 0. for key in num_regression_targets.keys()}, {key: 0. for key in num_regression_dense_targets.keys()})
            for _ in range(n_areas)
        ]
        regr_val_r2 = [
            ({key: 0. for key in num_regression_targets.keys()}, {key: 0. for key in num_regression_dense_targets.keys()})
            for _ in range(n_areas)
        ]
    else:
        clf_val_loss = [0.0] * n_areas
        clf_val_acc = [0.0] * n_areas
        clf_val_acc5 = [0.0] * n_areas
    
    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x = x.to(device)
            if multitask:
                y = tuple([yi.to(device) for yi in y])
                target_length = None
            else:
                y = y.to(device)
                # if task is seq2seq classification (e.g. phone), y.shape = (B, L); otherwise y.shape = (B,)
                target_length = y.shape[1] if len(y.shape) == 2 else y.shape[0]

            readout_inputs = model(x)[model_output_idx]
            
            for j in range(n_areas):
                readout_input = (readout_inputs[j] - readout_input_mean[j]) / (readout_input_std[j] + 1e-8)

                classifier_loss_, classifier_acc1_, classifier_acc5_, regression_loss_, regression_r2_ = compute_readout_loss(
                    data=readout_input, 
                    labels=y, 
                    readout=readout[j], 
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
                    # classifier_loss is a tuple of dicts
                    if num_classes_seq_labels is not None:
                        for task_name in num_classes_seq_labels.keys():
                            clf_val_loss[j][0][task_name] += classifier_loss_[0][task_name] / len(val_loader)
                            clf_val_acc[j][0][task_name] += classifier_acc1_[0][task_name] / len(val_loader)
                    if num_classes_dense_labels is not None:
                        for task_name in num_classes_dense_labels.keys():
                            clf_val_loss[j][1][task_name] += classifier_loss_[1][task_name] / len(val_loader)
                            clf_val_acc[j][1][task_name] += classifier_acc1_[1][task_name] / len(val_loader)
                    if num_regression_targets is not None:
                        for task_name in num_regression_targets.keys():
                            regr_val_loss[j][0][task_name] += regression_loss_[0][task_name] / len(val_loader)
                            regr_val_r2[j][0][task_name] += regression_r2_[0][task_name] / len(val_loader)
                    if num_regression_dense_targets is not None:
                        for task_name in num_regression_dense_targets.keys():
                            regr_val_loss[j][1][task_name] += regression_loss_[1][task_name] / len(val_loader)
                            regr_val_r2[j][1][task_name] += regression_r2_[1][task_name] / len(val_loader)
                else:
                    clf_val_loss[j] += classifier_loss_ / len(val_loader)
                    clf_val_acc[j] += classifier_acc1_ / len(val_loader)
                    clf_val_acc5[j] += classifier_acc5_ / len(val_loader)

    # Log validation results
    dist.barrier()
    if not args.nolog:
        _ = log_variable(epoch, "Offline Epoch", commit=False)
        for j in range(n_areas):
            if multitask:
                for task_name in num_classes_seq_labels.keys():
                    _ = log_variable(clf_val_loss[j][0][task_name], f"Offline val. loss ({task_name}), area {j}", commit=False)
                    _ = log_variable(clf_val_acc[j][0][task_name], f"Offline val. acc. ({task_name}), area {j}", commit=False)
                for task_name in num_classes_dense_labels.keys():
                    _ = log_variable(clf_val_loss[j][1][task_name], f"Offline val. loss ({task_name}), area {j}", commit=False)
                    _ = log_variable(clf_val_acc[j][1][task_name], f"Offline val. acc. ({task_name}), area {j}", commit=False)
                for task_name in num_regression_targets.keys():
                    _ = log_variable(regr_val_loss[j][0][task_name], f"Offline val. loss ({task_name}), area {j}", commit=False)
                    _ = log_variable(regr_val_r2[j][0][task_name], f"Offline val. R2 ({task_name}), area {j}", commit=False)
                for task_name in num_regression_dense_targets.keys():
                    _ = log_variable(regr_val_loss[j][1][task_name], f"Offline val. loss ({task_name}), area {j}", commit=False)
                    _ = log_variable(regr_val_r2[j][1][task_name], f"Offline val. R2 ({task_name}), area {j}", commit=False)
            else:
                _ = log_variable(clf_val_loss[j], f"Offline val. loss, area {j}", commit=False)
                _ = log_variable(clf_val_acc[j], f"Offline val. acc. ({args.offline_task}), area {j}", commit=False)
                _ = log_variable(clf_val_acc5[j], f"Top 5 offline val. acc. ({args.offline_task}), area {j}", commit=False)
    else:
        for j in range(n_areas):
            if multitask:
                for task_name in num_classes_seq_labels.keys():
                    clf_val_acc[j][0][task_name] = dist_reduce_mean(clf_val_acc[j][0][task_name])
                for task_name in num_classes_dense_labels.keys():
                    clf_val_acc[j][1][task_name] = dist_reduce_mean(clf_val_acc[j][1][task_name])
                for task_name in num_regression_targets.keys():
                    regr_val_r2[j][0][task_name] = dist_reduce_mean(regr_val_r2[j][0][task_name])
                for task_name in num_regression_dense_targets.keys():
                    regr_val_r2[j][1][task_name] = dist_reduce_mean(regr_val_r2[j][1][task_name])
            else:
                clf_val_acc[j] = dist_reduce_mean(clf_val_acc[j])

    # Print validation results
    if dist.get_rank() == 0:
        print(f"Epoch: {epoch}")
        for j in range(n_areas):
            print(f"Area {j}")
            if multitask:
                for task_name in num_classes_seq_labels.keys():
                    print(f"  Val. acc. ({task_name}): {clf_val_acc[j][0][task_name]:.4f}")
                for task_name in num_classes_dense_labels.keys():
                    print(f"  Val. acc. ({task_name}): {clf_val_acc[j][1][task_name]:.4f}")
                for task_name in num_regression_targets.keys():
                    print(f"  Val. R2 ({task_name}): {regr_val_r2[j][0][task_name]:.4f}")
                for task_name in num_regression_dense_targets.keys():
                    print(f"  Val. R2 ({task_name}): {regr_val_r2[j][1][task_name]:.4f}")
            else:
                print(f"  Val. acc. ({args.offline_task}): {clf_val_acc[j]:.4f}")


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

    preprocess, postprocess = additional_data_process(args.dataset, args.flatten_area_enc_output)
    n_in_channels = 1 if args.grayscale else 3

    state_dict = torch.load(args.model_path) if args.model_path is not None else None

    model = prepare_model(
        encoder_kind=args.area_encoders_kind,
        integrator_kind=args.area_integrators_kind,
        predictor_kind=args.area_predictors_kind,
        input_size=input_size,
        n_in_channels=n_in_channels,
        enc_dim=args.area_enc_dims,
        ctx_dim=args.area_ctx_dims,
        ctx_n_layers=args.area_ctx_n_layers,
        pred_n_hidden_layers=args.area_pred_n_hidden_layers,
        pred_hidden_dim=args.area_pred_hidden_dims,
        pred_steps=args.pred_steps,
        dense_prediction=args.dense_prediction,
        prediction_target=args.prediction_target,
        pred_target_dim_override=args.pred_target_dim_override,
        use_bn_enc=args.area_enc_bn,
        enc_n_layers=args.area_enc_n_layers,
        enc_kernel_size=args.area_enc_kernel_sizes,
        enc_stride=args.area_enc_strides,
        enc_padding=args.area_enc_paddings,
        enc_pool_size=args.area_enc_pool_sizes,
        preprocess=preprocess,
        postprocess=postprocess,
        state_dict=state_dict,
        return_full_features=args.offline_full_spatial_readout,
        flatten_enc_output=args.flatten_area_enc_output,
        n_areas=args.n_areas,
        frozen_areas=[True] * args.n_areas,
    )
    
    model = model.to(device)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)

    if args.offline_input == "enc":
        model_output_idx = 0
    elif args.offline_input == "ctx":
        model_output_idx = 1
    else:
        model_output_idx = 2
    
    # take args.model_path
    checkpoint_dir = os.path.dirname(args.model_path)

    greedy_offline_eval(args, model, num_classes, model_output_idx, seq_len, device, input_size, checkpoint_dir=checkpoint_dir)


    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline evaluation script")
    add_reproducibility_args(parser)
    add_logging_args(parser)
    add_data_args(parser)
    add_greedy_model_args(parser)
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
            wandb.init(project="hRPL", name=args.experiment_name, config=args)
    elif not args.nolog:
        wandb.init(project="hRPL", name=args.experiment_name, config=args)
    
    # check_args(args)
    main(args, device)
