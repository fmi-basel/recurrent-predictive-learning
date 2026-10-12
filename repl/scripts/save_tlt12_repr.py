"""
This script extracts latent representations from a trained model on the TLT12 
dataset and saves them along with their corresponding labels. The final 
representations and labels are saved in a .npz file format for later use.
The file contains the following arrays:
- X_train: Latent representations for the training set.
- y_train_clf: Classification labels for the training set.
- y_train_regr: Regression labels for the training set.
- X_val: Latent representations for the validation set.
- y_val_clf: Classification labels for the validation set.
- y_val_regr: Regression labels for the validation set.
- X_test: Latent representations for the test set.
- y_test_clf: Classification labels for the test set.
- y_test_regr: Regression labels for the test set.
- X_ood_test: Latent representations for the out-of-distribution test set.
- y_ood_test_clf: Classification labels for the out-of-distribution test set.
- y_ood_test_regr: Regression labels for the out-of-distribution test set.
"""
import torch
import numpy as np
from repl.data.image_dataloader import TLT12DataLoader
from repl.scripts.utils import get_data_specs
from repl.networks.utils import additional_data_process
from repl.networks.utils import setup_encoder, setup_integrator, setup_predictor
from repl.networks.repl import RePLModel
from repl.scripts.args import (
    add_encoder_args,
    add_context_integrator_args,
    add_predictor_args,
)
import os
import argparse
from tqdm.auto import tqdm
from copy import deepcopy


def get_repr(model, x, repr_idx, dense_labels=True, time_avg=True):
    """
    Extracts the latent representation from the model output.
    
    Args:
        model: The trained model.
        x: The input data.
        repr_idx: The index of the model output to extract. (0: encoder, 1: integrator, 2: predictor)
        dense_labels: Whether to flatten the labels.
        time_avg: Whether to average the representations over time.

    Returns:
        The extracted latent representation.
    """
    with torch.no_grad():
        out = model(x)
        input_repr = out[repr_idx].detach()
        if dense_labels:
            input_repr = input_repr.flatten(0, 1)
        elif time_avg:
            input_repr = input_repr.mean(dim=1)
        else:
            input_repr = input_repr[:, -1, :]
    return input_repr


def get_synset_labels_per_timestep(y):
    """
    Add labels for synset classification to each timestep of the sequence.

    Args:
        y: The tuple of labels returned by TLT12DataLoader.

    Returns:
        A tensor of shape (batch_size * seq_len, 1) containing the synset labels.
    """
    synset_labels = y[0][:, 0]
    seq_len = y[3].shape[1]
    synset_labels_per_timestep = synset_labels.unsqueeze(1).repeat(1, seq_len)
    return synset_labels_per_timestep.view(-1, 1)


def get_labels(y, is_regr=True, dense_labels=True, do_abs=False, indices=None):
    """
    Returns the labels for the task.

    Args:
        y: The tuple of labels returned by TLT12DataLoader.
        is_regr: Whether to return regression labels or classification labels.
        dense_labels: Whether to return labels for each timestep or for the entire sequence.
        do_abs: Whether to take the absolute value of the labels.
        indices: The indices of the labels to return. By default, all labels are returned.

    Returns:
        A tensor of shape (batch_size * seq_len, num_labels) or (batch_size, num_labels) containing the labels.
    """
    if is_regr:
        y_true = y[3].flatten(0, 1) if dense_labels else y[2]
        if do_abs:
            y_true = torch.abs(y_true)
    else:
        if dense_labels:
            y_true = y[1].flatten(0, 1)
            synset_labels = get_synset_labels_per_timestep(y)
            y_true = torch.cat([y_true, synset_labels], dim=-1)
        else:
            y_true = y[0]
    if indices is not None:
        y_true = y_true[:, indices]
    return y_true


def get_latent_representations_and_labels(model, loader, repr_idx=1, 
                                          device="cuda",
                                          dense_labels=True, do_abs=False, 
                                          time_avg=True, indices=None):
    """
    Extracts latent representations and corresponding labels from the model for a given data loader.

    Args:
        model: The model to extract representations from.
        loader: The data loader for a data split.
        repr_idx: The index of the model output to extract from. (0: encoder, 1: integrator, 2: predictor)
        device: The device to run the model on.
        dense_labels: Whether to return labels for each timestep or for the entire sequence.
        do_abs: Whether to take the absolute value of the labels.
        time_avg: Whether to average the representations over time.
        indices: The indices of the labels to return. By default, all labels are returned.

    Returns:
        A tuple of (representations, labels, num_classification_labels) where:
            - representations is a numpy array of shape (num_samples, representation_dim)
            - labels is a numpy array of shape (num_samples, num_labels)
            - num_classification_labels is an integer representing the number of classification tasks (used to separate classification labels from regression labels)
    """
    representations = []
    labels = []
    for X, y in tqdm(loader, desc="Extracting representations"):
        X = X.to(device)
        input_repr = get_repr(model, X, repr_idx, dense_labels, time_avg)
        y_true_clf = get_labels(y, is_regr=False, dense_labels=dense_labels, 
                            do_abs=do_abs, indices=indices)
        y_true_regr = get_labels(y, is_regr=True, dense_labels=dense_labels, 
                            do_abs=do_abs, indices=indices)[:, :10]
        y_true = torch.cat([y_true_clf, y_true_regr], dim=-1)
        n_clf = y_true_clf.shape[-1]
        representations.append(input_repr.cpu().numpy())
        labels.append(y_true.cpu().numpy())
    representations = np.concatenate(representations, axis=0)
    labels = np.concatenate(labels, axis=0)
    return representations, labels, n_clf


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='datasets/TLT12/shards', help='Path to the dataset directory.')
    parser.add_argument('--model_dir', type=str, default='checkpoints', help='Path to the directory containing the trained model.')
    parser.add_argument('--save_dir', type=str, default='tlt12_reprs', help='Path to the directory where the extracted representations will be saved.')
    parser.add_argument('--model_name', type=str, default=None, help='Name of the trained model to load. If None, a randomly initialized model will be used.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loading.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for running the model.')
    parser.add_argument('--repr_file_name', type=str, default="representations.npz", help='Name of the file to save the extracted representations and labels. Use METHODNAME_COUNTER.npz for the evaluation scripts where METHODNAME is the name of the method (e.g., RPL, sup, etc.) and COUNTER is a number to count different models.')
    parser.add_argument("--repr_idx", type=int, default=1, help="Index of the model output to extract from (0: encoder, 1: integrator, 2: predictor).")
    add_encoder_args(parser)
    add_context_integrator_args(parser)
    add_predictor_args(parser)
    args = parser.parse_args()

    # set paths
    DATA_DIR = args.dataset_dir
    SAVE_DIR = args.save_dir
    if args.model_name is not None:
        MODEL_DIR = os.path.join(args.model_dir, args.model_name)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Prepare data
    input_size, num_classes = get_data_specs("tlt12", "multitask")
    dataloader = TLT12DataLoader(data_dir=DATA_DIR,
                                    num_workers=args.num_workers)
    dataloader.train_subset._normalize_latents = False
    dataloader.val_traversal_dataset._normalize_latents = False
    dataloader.test_traversal_dataset._normalize_latents = False
    train_loader, _ = dataloader.get_supervised_train(batch_size=args.batch_size)
    val_loader, _ = dataloader.get_supervised_validation(batch_size=args.batch_size)
    test_loader, _ = dataloader.get_test(batch_size=args.batch_size)
    ood_test_loader, _ = dataloader.get_ood_test(batch_size=args.batch_size)
    preprocess, postprocess = additional_data_process("tlt12", True)

    # create model
    encoder = setup_encoder(
        kind=args.encoder,
        input_size=input_size,
        n_in_channels=3,
        latent_size=args.enc_output_dim,
        use_bn=args.use_bn,
        enc_n_layers=args.enc_n_layers,
        resnet_n_channels=args.resnet_n_channels,
        resnet_n_blocks=args.resnet_n_blocks,
        resnet_pool=args.resnet_pool,
        resnet_stem_kernel_size=args.resnet_stem_kernel_size,
    )
    integrator = setup_integrator(kind=args.integrator,
                                  input_size=args.enc_output_dim,
                                  latent_size=args.ctx_dim)
    predictor = setup_predictor(kind=args.predictor,
                                ctx_size=args.ctx_dim,
                                enc_size=args.enc_output_dim,
                                n_layers=args.pred_n_hidden_layers,
                                hidden_size=args.pred_hidden_dim,
                                pred_steps=args.pred_steps,
                                dense_prediction=args.dense_prediction,
                                pred_act_fn=args.pred_act_fn,
                                prediction_target=args.prediction_target,)
    model = RePLModel(encoder=encoder,
                      integrator=integrator,
                      predictor=predictor,
                      preprocess=preprocess,
                      postprocess=postprocess)

    # load model weights
    print(f"model architecture: {model}")
    if args.model_name is not None:  # if None, randomly model
        model_pre_load = deepcopy(model)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_final.pt")))
        model_post_load = deepcopy(model)
        # check if the model weights were loaded correctly
        for pre_param, post_param in zip(model_pre_load.parameters(), model_post_load.parameters()):
            if pre_param.data.equal(post_param.data):
                print(f"Parameter {pre_param.name} did not change after loading weights.")
    model.to(args.device)

    # get latent representations and labels
    X_train, y_train, n_clf = get_latent_representations_and_labels(
        model=model,
        loader=train_loader,
        repr_idx=args.repr_idx,
        device=args.device,
        dense_labels=True,
        do_abs=False,
        time_avg=False,
    )
    y_train_clf = y_train[:, :n_clf]
    y_train_regr = y_train[:, n_clf:]

    X_val, y_val, _ = get_latent_representations_and_labels(
        model=model,
        loader=val_loader,
        repr_idx=args.repr_idx,
        device=args.device,
        dense_labels=True,
        do_abs=False,
        time_avg=False,
    )
    y_val_clf = y_val[:, :n_clf]
    y_val_regr = y_val[:, n_clf:]

    X_test, y_test, _ = get_latent_representations_and_labels(
        model=model,
        loader=test_loader,
        repr_idx=args.repr_idx,
        device=args.device,
        dense_labels=True,
        do_abs=False,
        time_avg=False,
    )
    y_test_clf = y_test[:, :n_clf]
    y_test_regr = y_test[:, n_clf:]
    
    X_ood_test, y_ood_test, _ = get_latent_representations_and_labels(
        model=model,
        loader=ood_test_loader,
        repr_idx=args.repr_idx,
        device=args.device,
        dense_labels=True,
        do_abs=False,
        time_avg=False,
    )
    y_ood_test_clf = y_ood_test[:, :n_clf]
    y_ood_test_regr = y_ood_test[:, n_clf:]

    prefix = ""
    if args.repr_idx == 0:
        prefix = "enc_"
    elif args.repr_idx == 2:
        prefix = "pred_"
    np.savez(
        os.path.join(SAVE_DIR, f"{prefix}{args.repr_file_name}.npz"),
        X_train=X_train,
        y_train_clf=y_train_clf,
        y_train_regr=y_train_regr,
        X_val=X_val,
        y_val_clf=y_val_clf,
        y_val_regr=y_val_regr,
        X_test=X_test,
        y_test_clf=y_test_clf,
        y_test_regr=y_test_regr,
        X_ood_test=X_ood_test,
        y_ood_test_clf=y_ood_test_clf,
        y_ood_test_regr=y_ood_test_regr,
    )

if __name__ == "__main__":
    __main__()
