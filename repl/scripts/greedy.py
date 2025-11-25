import argparse
from copy import deepcopy
import os

import torch
import torch.distributed as dist
from tqdm import tqdm
import wandb

from repl.networks.utils import additional_data_process, prepare_model
from repl.scripts.args import (
    add_criterion_args, 
    add_data_args,
    add_ddp_args, 
    add_greedy_model_args, 
    add_logging_args, 
    add_offline_eval_args, 
    add_online_eval_args, 
    add_optimization_args, 
    add_reproducibility_args, 
    add_training_args, 
    add_validation_args, 
)
from repl.scripts.eval_utils import compute_readout_loss, prepare_readout
from repl.scripts.online_eval import greedy_online_eval
from repl.scripts.offline_greedy_eval import greedy_offline_eval as offline_eval
from repl.scripts.utils import (
    get_criterion_input,
    get_data_specs,
    init_distributed,
    log_variable,
    prepare_criterion,
    prepare_data,
    prepare_model_optimization, 
    seed_everything
)


def main(args, device):
    if args.distributed:
        raise NotImplementedError("Distributed training is not supported for greedy models.")
    
    multitask = args.online_task == "multitask"
    input_size, num_classes = get_data_specs(dataset=args.dataset,
                                             target_label=args.online_task,
                                             mnist_seqtype=args.mnist_seqtype,
                                             spritevid_num_sprites=args.spritevid_max_sprites,
                                             flatten_images=args.flatten_images,)
    
    num_classes_seq_labels = num_classes[0] if multitask else None
    num_classes_dense_labels = num_classes[1] if multitask else None
    num_regression_targets = num_classes[2] if multitask else None
    num_regression_dense_targets = num_classes[3] if multitask else None

    num_classes_seq_labels = num_classes[0] if multitask else None
    num_classes_dense_labels = num_classes[1] if multitask else None

    seq_len = args.seq_len

    (train_loader, train_sampler, 
        val_loader, val_sampler,
        test_loader, test_sampler,
    ) = prepare_data(
            args.dataset,
            args.data_input_dir,
            seq_len=seq_len,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            distributed=args.distributed,
            num_workers=args.num_workers,
            grayscale=args.grayscale,
            target_label=args.online_task,
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
    
    # if args.val_size is 0, then we use test set for validation
    if args.val_size == 0:
        val_loader = test_loader
        val_sampler = test_sampler

    preprocess, postprocess = additional_data_process(args.dataset, args.flatten_area_enc_output)
    n_in_channels = 1 if args.grayscale else 3

    if args.freeze:
        # not needed but will make random baselines slightly more efficient + will make it consistent with frozen_areas arg
        frozen_areas = [True] * args.n_areas
    elif args.frozen_areas is None:
        frozen_areas = [False] * args.n_areas
    else:
        frozen_areas = [True if i in args.frozen_areas else False for i in range(args.n_areas)]

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
        return_full_features=args.online_full_spatial_readout,
        flatten_enc_output=args.flatten_area_enc_output,
        n_areas=args.n_areas,
        frozen_areas=frozen_areas,
    )

    readout = prepare_readout(
        task=args.online_task,
        downstream_input=args.online_input,
        input_spatial_size=input_size,
        single_timestep_readout=args.online_single_timestep_readout,
        full_spatial_readout=args.online_full_spatial_readout,
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
        n_areas=args.n_areas,
    )

    criterion = prepare_criterion(
        loss=args.loss,
        pred_steps=args.pred_steps,
        regularize_over=args.regularize_over,
        discount_factor=args.discount_factor,
        pull_coef=args.pull_coef,
        push_coef=args.push_coef,
        decorr_coef=args.decorr_coef,
        readout=deepcopy(readout),  # readout is passed to criterion for supervised training and it is a copy of downstream readout
        classification_task=args.online_task,
        dense_prediction=args.dense_prediction,
        pred_loss_type=args.pred_loss_type,
        full_spatial_readout=args.online_full_spatial_readout,
        no_sg=args.no_sg,
    )

    classifier_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    regression_criterion = torch.nn.MSELoss()

    if args.online_input == "enc":
        model_output_idx = 0
    elif args.online_input == "ctx":
        model_output_idx = 1
    else:
        model_output_idx = 2

    optimizer, scheduler = prepare_model_optimization(
        model=model, readout=readout, optimizer=args.optimizer, use_scheduler=args.use_scheduler, epochs=args.epochs,
        lr=args.lr, weight_decay=args.weight_decay, pred_lr_mult=args.pred_lr_mult, online_lr=args.online_lr,
        online_weight_decay=args.online_weight_decay, criterion=criterion
    )

    model = model.to(device)
    readout = readout.to(device)

    criterion = criterion.to(device)
    classifier_criterion = classifier_criterion.to(device)
    regression_criterion = regression_criterion.to(device)

    # set up checkpointing
    if args.checkpoint_dir is not None:
        checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
        if not os.path.exists(checkpoint_dir) and dist.get_rank() == 0:
            os.makedirs(checkpoint_dir)
    else:
        args.checkpoint_every = 0

    # vector dims for logging norms
    vector_dims = -1

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)

        if args.use_scheduler:
            scheduler.step(epoch)

        # # save checkpoint
        if dist.get_rank() == 0 and args.checkpoint_every > 0:
            if args.checkpoint_every == 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{epoch}.pt"))
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, f"optimizer_{epoch}.pt"))

        # evaluation
        greedy_online_eval(
            args,
            model=model,
            readout=readout,
            test_loader=val_loader,
            criterion=criterion,
            classifier_criterion=classifier_criterion,
            regression_criterion=regression_criterion,
            device=device,
            epoch=epoch,
            model_output_idx=model_output_idx,
            multitask=multitask,
            num_classes_seq_labels=num_classes_seq_labels,
            num_classes_dense_labels=num_classes_dense_labels,
            num_regression_targets=num_regression_targets,
            num_regression_dense_targets=num_regression_dense_targets,
        )

        # train
        for i, (x, y) in enumerate(tqdm(train_loader)):
            x = x.to(device) # (B, C, H, W)
            if multitask:
                y = tuple([yi.to(device) for yi in y])
                target_length = None
            else:
                y = y.to(device)
                # if task is seq2seq classification (e.g. phone), y.shape = (B, L); otherwise y.shape = (B,)
                target_length = y.shape[1] if len(y.shape) == 2 else y.shape[0]
            
            # one forward pass through the model with loss calculation
            loss_values = torch.zeros(args.n_areas)
            if args.loss == "inv":
                pull_loss_values = torch.zeros(args.n_areas)
                push_loss_values = torch.zeros(args.n_areas)
                decorr_loss_values = torch.zeros(args.n_areas)
            total_loss = 0.0
            if args.freeze:
                with torch.no_grad():
                    model_outputs = model(x)
            else:
                model_outputs = model(x)  # (z, context, pred)
                z_list, ctx_list, pred_list = model_outputs
                for j in range(args.n_areas):
                    criterion_inputs = get_criterion_input(z_list[j], ctx_list[j], pred_list[j], y, args.loss, args.online_input, args.prediction_target)
                    loss = criterion(*criterion_inputs)
                    loss_values[j] = loss.item()
                    if args.loss == "inv":
                        pull_loss_values[j] = criterion.pull_loss_val
                        push_loss_values[j] = criterion.push_loss_val
                        decorr_loss_values[j] = criterion.decorr_loss_val
                    if not frozen_areas[j]:
                        total_loss += loss

            # readout pass
            clf_loss_values = []
            reg_loss_values = []
            for j in range(args.n_areas):
                clf_loss, _, _, reg_loss, _ = compute_readout_loss(
                    data=model_outputs[model_output_idx][j],
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
                    full_spatial_readout=args.online_full_spatial_readout,
                )
                if multitask:
                    clf_loss_values.append(clf_loss)
                    reg_loss_values.append(reg_loss)
                    if num_classes_seq_labels is not None:
                        for task_name in num_classes_seq_labels.keys():
                            total_loss += clf_loss[0][task_name]
                    if num_classes_dense_labels is not None:
                        for task_name in num_classes_dense_labels.keys():
                            total_loss += clf_loss[1][task_name]
                    if num_regression_targets is not None:
                        for task_name in num_regression_targets.keys():
                            total_loss += reg_loss[0][task_name]
                    if num_regression_dense_targets is not None:
                        for task_name in num_regression_dense_targets.keys():
                            total_loss += reg_loss[1][task_name]
                else:
                    clf_loss_values.append(clf_loss.item())
                    total_loss += clf_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # log to wandb
            if not args.nolog:
                dist.barrier()
                z_list, ctx_list, pred_list = model_outputs
                for j in range(args.n_areas):
                    log_variable(loss_values[j], f"Train loss, area {j}", commit=False)
                    if multitask:
                        if num_classes_seq_labels is not None:
                            for task_name in num_classes_seq_labels.keys():
                                log_variable(clf_loss_values[j][0][task_name], f"Train clf loss, area {j}, {task_name}", commit=False)
                        if num_classes_dense_labels is not None:
                            for task_name in num_classes_dense_labels.keys():
                                log_variable(clf_loss_values[j][1][task_name], f"Train clf loss, area {j}, {task_name}", commit=False)
                        if num_regression_targets is not None:
                            for task_name in num_regression_targets.keys():
                                log_variable(reg_loss_values[j][0][task_name], f"Train reg loss, area {j}, {task_name}", commit=False)
                        if num_regression_dense_targets is not None:
                            for task_name in num_regression_dense_targets.keys():
                                log_variable(reg_loss_values[j][1][task_name], f"Train reg loss, area {j}, {task_name}", commit=False)
                    else:
                        log_variable(clf_loss_values[j], f"Train clf loss, area {j}", commit=False)
                    log_variable(torch.linalg.vector_norm(z_list[j], dim=vector_dims).mean().item(), f"Act. norm, area {j}", commit=False)
                    log_variable(torch.linalg.vector_norm(ctx_list[j], dim=vector_dims).mean().item(), f"Ctx. norm, area {j}", commit=False)
                    log_variable(torch.linalg.vector_norm(pred_list[j], dim=vector_dims).mean().item(), f"Pred. norm, area {j}", commit=False)
                    log_variable(torch.var(z_list[j]).item(), f"Act. var, area {j}")
                    if args.loss == "inv":
                        log_variable(pull_loss_values[j], f"Pull loss, area {j}", commit=False)
                        log_variable(push_loss_values[j], f"Push loss, area {j}", commit=False)
                        log_variable(decorr_loss_values[j], f"Decorr loss, area {j}", commit=False)
            
    # save final model
    if dist.get_rank() == 0 and args.checkpoint_dir is not None:
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_final.pt"))
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, f"optimizer_final.pt"))
        if args.save_online_readout:
            torch.save(readout.state_dict(), os.path.join(checkpoint_dir, f"online_{args.online_input}_readout.pt"))

    # final evaluation
    greedy_online_eval(
        args,
        model=model,
        readout=readout,
        test_loader=val_loader,
        criterion=criterion,
        classifier_criterion=classifier_criterion,
        regression_criterion=regression_criterion,
        device=device,
        epoch=args.epochs,
        model_output_idx=model_output_idx,
        multitask=multitask,
        num_classes_seq_labels=num_classes_seq_labels,
        num_classes_dense_labels=num_classes_dense_labels,
        num_regression_targets=num_regression_targets,
        num_regression_dense_targets=num_regression_dense_targets,
    )

    # offline evaluation
    if args.offline_task is not None:
        if args.offline_input == "enc":
            offline_model_output_idx = 0
        elif args.offline_input == "ctx":
            offline_model_output_idx = 1
        else:
            offline_model_output_idx = 2
        if args.dataset == "libri":
            if args.offline_task == "seq2label":
                offline_num_classes = 251
            elif args.offline_task == "seq2seq":
                offline_num_classes = 41
            offline_eval(args, model, offline_num_classes, offline_model_output_idx, seq_len, device, input_size=input_size, checkpoint_dir=checkpoint_dir)
        elif args.dataset == "shakespeare":
            offline_eval(args, model, num_classes, offline_model_output_idx, seq_len, device, input_size=input_size, checkpoint_dir=checkpoint_dir)
        else:
            offline_eval(args, model, num_classes, offline_model_output_idx, seq_len, device, input_size=input_size, checkpoint_dir=checkpoint_dir)
    
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_reproducibility_args(parser)
    add_logging_args(parser)
    add_data_args(parser)
    add_greedy_model_args(parser)
    add_criterion_args(parser)
    add_training_args(parser)
    add_optimization_args(parser)
    add_validation_args(parser)
    add_online_eval_args(parser)
    add_offline_eval_args(parser)
    add_ddp_args(parser)
    args = parser.parse_args()

    seed_everything(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = init_distributed(args.dist_backend, args.dist_url, args.local_rank)

    if args.distributed:
        if dist.get_rank() == 0 and not args.nolog:
            wandb.init(project="hRPL", name=args.experiment_name, config=args)
    elif not args.nolog:
        wandb.init(project="hRPL", name=args.experiment_name, config=args)

    main(args, device)
