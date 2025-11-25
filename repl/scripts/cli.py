import argparse
from copy import deepcopy
import os

import torch
import torch.distributed as dist
import wandb
from tqdm.auto import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from repl.networks.utils import additional_data_process, prepare_model
from repl.scripts.offline_eval import offline_eval
from repl.scripts.online_eval import online_eval
from repl.scripts.utils import (
    get_criterion_input,
    get_data_specs,
    init_distributed,
    prepare_model_optimization,
    seed_everything,
    prepare_data,
    prepare_criterion,
    log_variable,
)
from repl.scripts.eval_utils import compute_readout_loss, prepare_readout
from repl.scripts.args import (
    add_data_args,
    add_ddp_args,
    add_logging_args,
    add_criterion_args,
    add_model_args,
    add_optimization_args,
    add_reproducibility_args,
    add_training_args,
    add_validation_args,
    add_online_eval_args,
    add_offline_eval_args,
    check_args,
)


def main(args, device):
    if args.distributed:
        rank = dist.get_rank()
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else torch.cpu.device_count()
        device_id = rank % device_count
        if args.distribute_data:
            args.batch_size //= device_count
            args.val_batch_size //= device_count
        device = device_id
    
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
            inter_trial_interval=args.inter_trial_interval,
        )

    # if args.val_size is 0, then we use test set for validation
    if args.val_size == 0:
        val_loader = test_loader
        val_sampler = test_sampler

    preprocess, postprocess = additional_data_process(args.dataset, args.flatten_enc_output)
    n_in_channels = 1 if args.grayscale else 3

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
        preprocess=preprocess,
        postprocess=postprocess,
        return_full_features=args.online_full_spatial_readout,
        flatten_enc_output=args.flatten_enc_output,
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
        enc_output_dim=args.enc_output_dim,
        ctx_dim=args.ctx_dim,
        pred_steps=args.pred_steps,
        dense_prediction=args.dense_prediction,
        prediction_target=args.prediction_target,
        pred_target_dim_override=args.pred_target_dim_override,
        model=model,
    )
            
    criterion = prepare_criterion(
        loss=args.loss,
        pred_steps=args.pred_steps,
        regularize_over=args.regularize_over,
        discount_factor=args.discount_factor,
        pull_coef=args.pull_coef,
        push_coef=args.push_coef,
        decorr_coef=args.decorr_coef,
        readout=deepcopy(readout),  # readout is passed to criterion for supervised training and it is a copy of downstream readout which is trained separately
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
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        readout = torch.nn.SyncBatchNorm.convert_sync_batchnorm(readout)
        model = DDP(model, device_ids=[device], output_device=device)
        readout = DDP(readout, device_ids=[device], output_device=device)
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

    # train loop
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)

        if args.use_scheduler:
            scheduler.step(epoch)

        # # save checkpoint
        if args.checkpoint_every > 0:
            if dist.get_rank() == 0 and epoch % args.checkpoint_every == 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{epoch}.pt"))
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, f"optimizer_{epoch}.pt"))

            # dist.barrier()
            # map_location = {'cuda:%d' % 0: 'cuda:%d' % device_id}
            # model.load_state_dict(
            #     torch.load(os.path.join(checkpoint_dir, f"model_{epoch}.pt"), map_location=map_location)
            # )
            # optimizer.load_state_dict(
            #     torch.load(os.path.join(checkpoint_dir, f"optimizer_{epoch}.pt"), map_location=map_location)
            # )
        # initial evaluation
        online_eval(args, 
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
                    num_regression_dense_targets=num_regression_dense_targets)

        # train
        for i, (x, y) in enumerate(tqdm(train_loader)):
            # move to device
            x = x.to(device)
            if multitask:
                y = tuple([yi.to(device) for yi in y])
                target_length = None
            else:
                y = y.to(device)
                # if task is seq2seq classification (e.g. phone), y.shape = (B, L); otherwise y.shape = (B,)
                target_length = y.shape[1] if len(y.shape) == 2 else y.shape[0]
            
            # forward pass
            if args.freeze:  # freeze model weights, no grad computation, only train readout
                with torch.no_grad():
                    model_output = model(x)
                    loss_val = 0.0  # no val. loss from model
                    loss = 0.0  # no train loss from model
            else:
                model_output = model(x)  # (z, context, pred)
                criterion_inputs = get_criterion_input(*model_output, y, args.loss, args.online_input, args.prediction_target)
                loss = criterion(*criterion_inputs)
                loss_val = loss.item()
            
            classifier_loss, _, _, regression_loss, _ = compute_readout_loss(
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
                full_spatial_readout=args.online_full_spatial_readout,
            )
            readout_loss_val = 0.0
            if multitask:
                # classifier_loss is also a tuple of two dictionaries (seq_labels, dense_labels)
                if num_classes_seq_labels is not None:
                    for task_name in num_classes_seq_labels.keys():
                        readout_loss_val += classifier_loss[0][task_name]
                if num_classes_dense_labels is not None:
                    for task_name in num_classes_dense_labels.keys():
                        readout_loss_val += classifier_loss[1][task_name]
                if num_regression_targets is not None:
                    for task_name in num_regression_targets.keys():
                        readout_loss_val += regression_loss[0][task_name]
                if num_regression_dense_targets is not None:
                    for task_name in num_regression_dense_targets.keys():
                        readout_loss_val += regression_loss[1][task_name]
            else:
                readout_loss_val = classifier_loss
            loss += readout_loss_val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # log to wandb
            if not args.nolog:
                dist.barrier()
                z, context, pred = model_output
                log_variable(loss_val, "Train loss", commit=False)
                if multitask:
                    if num_classes_seq_labels is not None:
                        for task_name in num_classes_seq_labels.keys():
                            log_variable(classifier_loss[0][task_name].item(), f"Classifier train loss {task_name}", commit=False)
                    if num_classes_dense_labels is not None:
                        for task_name in num_classes_dense_labels.keys():
                            log_variable(classifier_loss[1][task_name].item(), f"Classifier train loss {task_name}", commit=False)
                    if num_regression_targets is not None:
                        for task_name in num_regression_targets.keys():
                            log_variable(regression_loss[0][task_name].item(), f"Regression train loss {task_name}", commit=False)
                    if num_regression_dense_targets is not None:
                        for task_name in num_regression_dense_targets.keys():
                            log_variable(regression_loss[1][task_name].item(), f"Regression train loss {task_name}", commit=False)
                else:
                    log_variable(classifier_loss.item(), "Classifier train loss", commit=False)
                log_variable(torch.linalg.vector_norm(z, dim=vector_dims).mean(), "Act. norm", commit=False)
                log_variable(torch.linalg.vector_norm(context, dim=vector_dims).mean(), "Context norm", commit=False)
                log_variable(torch.linalg.vector_norm(pred, dim=vector_dims).mean(), "Pred. norm", commit=False)
                log_variable(torch.var(z), "Variance act.")
                if args.loss == "inv":
                    log_variable(criterion.pull_loss_val, "Pull loss (train)", commit=False)
                    log_variable(criterion.push_loss_val, "Push loss (train)", commit=False)
                    log_variable(criterion.decorr_loss_val, "Decorr loss (train)", commit=False)
    
    # save final model
    if dist.get_rank() == 0 and args.checkpoint_dir is not None:
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_final.pt"))
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, f"optimizer_final.pt"))
        if args.save_online_readout:
            torch.save(readout.state_dict(), os.path.join(checkpoint_dir, f"online_{args.online_input}_readout.pt"))

    # final evaluation
    online_eval(args, 
                model=model, 
                readout=readout, 
                test_loader=val_loader, 
                criterion=criterion, 
                classifier_criterion=classifier_criterion, 
                regression_criterion=regression_criterion,
                device=device, 
                epoch=epoch+1, 
                model_output_idx=model_output_idx, 
                multitask=multitask, 
                num_classes_seq_labels=num_classes_seq_labels, 
                num_classes_dense_labels=num_classes_dense_labels,
                num_regression_targets=num_regression_targets,
                num_regression_dense_targets=num_regression_dense_targets)
    
    # get index for model output to use for offline evaluation
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
        else:
            offline_eval(args, model, num_classes, offline_model_output_idx, seq_len, device, input_size=input_size, checkpoint_dir=checkpoint_dir)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_reproducibility_args(parser)
    add_logging_args(parser)
    add_data_args(parser)
    add_model_args(parser)
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
            wandb.init(project="RPL", name=args.experiment_name, config=args)
    elif not args.nolog:
        wandb.init(project="RPL", name=args.experiment_name, config=args)
    
    check_args(args)
    main(args, device)
