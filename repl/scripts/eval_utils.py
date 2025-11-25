import numpy as np
import torch
from repl.networks.networks import LinearReadout, IdentityIntegrator
from repl.networks.utils import get_predictor_output_dim
from repl.metrics.metrics import compute_r2


def compute_readout(data, target_length, readout, readout_input, task=None, dense_prediction=False, single_readout=False, pred_steps=None, full_spatial_readout=False):
    """Compute the logits for the downstream task

    Args:
        data (torch.Tensor): The data tensor
        target_length (int): The target length (used for phone classification)
        readout (nn.Module): The readout network
        readout_input (str): The model's component from which the data tensor comes. One of ["enc", "ctx", "pred"]
        task (str): The downstream task. One of ["phone", "speaker"]
        dense_prediction (bool): Whether to predict dense output, i.e. make predictions for each time step up to pred_steps
        single_readout (bool): Whether to use a single timestep readout (for predictor only)
        pred_steps (int): The number of prediction steps
        full_spatial_readout (bool): Whether to use full spatial readout (for supervised loss)

    Returns:
        torch.Tensor: The logits
    """
    if len(data.shape) == 5:
        if full_spatial_readout:
            data = data.view(data.shape[0], data.shape[1], -1)
        else:
            data = data.mean(-1).mean(-1)

    if task == "seq2seq":
        if "pred" in readout_input and dense_prediction and single_readout:
                pred_dim = data.shape[-1]
                logits = readout(data[:, -target_length:, :pred_dim//pred_steps].detach())
        else:
            logits = readout(data[:, -target_length:].detach())
    else:
        logits = readout(data.detach())
    return logits

def compute_readout_loss(data, 
                         labels, 
                         readout, 
                         classifier_criterion,
                         regression_criterion, 
                         target_length, 
                         downstream_input, 
                         dense_prediction, 
                         single_readout, 
                         pred_steps, 
                         task, 
                         full_spatial_readout):
    """ Compute the readout loss and accuracy for the downstream task

    Args:
        data (torch.Tensor): The data tensor
        labels (torch.Tensor): The labels tensor
        readout (nn.Module): The readout network
        classifier_criterion (nn.Module): The classification loss criterion
        regression_criterion (nn.Module): The regression loss criterion
        target_length (int): The target length (used for phone classification)
        downstream_input (str): The model's component from which the data tensor comes. One of ["enc", "ctx", "pred"]
        dense_prediction (bool): Whether to predict dense output, i.e. make predictions for each time step up to pred_steps
        single_readout (bool): Whether to use a single timestep readout (for predictor only)
        pred_steps (int): The number of prediction steps
        task (str): The downstream task. One of ["seq2label", "seq2seq", "multitask"]
        full_spatial_readout (bool): Whether to use full spatial readout (for supervised loss)

    Returns:
        torch.Tensor: The classification loss
        torch.Tensor: The classification accuracy
        torch.Tensor: The top-5 classification
        torch.Tensor: The regression loss
        torch.Tensor: The regression R2
    """
    multitask = task == "multitask"
    if multitask:
        classification_loss = ({}, {})
        classification_accuracy = ({}, {})
        regression_loss = ({}, {})
        regression_r2 = ({}, {})
        seq_labels, seq2seq_labels, regr_targets, dense_regr_targets, _ = labels
        num_seq_labels = num_dense_labels = 0
        num_regression_targets = num_regression_dense_targets = 0
        classifier_acc5 = None

        num_seq_labels = seq_labels.shape[-1] # will be 0 if seq_labels is empty
        for i in range(num_seq_labels):
            logits = compute_readout(
                data=data,
                target_length=target_length,
                readout=readout[i],
                readout_input=downstream_input,
                task="seq2label",
                dense_prediction=dense_prediction,
                single_readout=single_readout,
                pred_steps=pred_steps,
                full_spatial_readout=full_spatial_readout,
            )
            classification_loss[0][readout[i].task_name] = classifier_criterion(logits, seq_labels[:, i])
            # filter out rows of seq_labels that are -1
            seq_labels_i = seq_labels[:, i]
            logits = logits[seq_labels_i != -1]
            seq_labels_i = seq_labels_i[seq_labels_i != -1]
            classification_accuracy[0][readout[i].task_name] = (logits.argmax(dim=1) == seq_labels_i).float().mean()
        
        num_dense_labels = seq2seq_labels.shape[-1] # will be 0 if seq2seq_labels is empty
        target_length = seq2seq_labels.shape[1]
        for i in range(num_dense_labels):
            logits = compute_readout(
                data=data,
                target_length=target_length,
                readout=readout[num_seq_labels + i],
                readout_input=downstream_input,
                task="seq2seq",
                dense_prediction=dense_prediction,
                single_readout=single_readout,
                pred_steps=pred_steps,
                full_spatial_readout=full_spatial_readout,
            )
            labels = seq2seq_labels[:, :, i].view(-1)
            classification_loss[1][readout[num_seq_labels + i].task_name] = classifier_criterion(logits, labels)
            # filter out rows of seq2seq_labels that are -1
            logits = logits[labels != -1]
            labels = labels[labels != -1]
            classification_accuracy[1][readout[num_seq_labels + i].task_name] = (logits.argmax(dim=1) == labels).float().mean()

        num_regression_targets = regr_targets.shape[-1] # will be 0 if regr_targets is empty
        for i in range(num_regression_targets):
            pred = compute_readout(
                data=data,
                target_length=target_length,
                readout=readout[num_seq_labels + num_dense_labels + i],
                readout_input=downstream_input,
                task="seq2label",
                dense_prediction=dense_prediction,
                single_readout=single_readout,
                pred_steps=pred_steps,
                full_spatial_readout=full_spatial_readout,
            )
            regression_loss[0][readout[num_seq_labels + num_dense_labels + i].task_name] = regression_criterion(pred.squeeze(), regr_targets[:, i].view(-1))
            regression_r2[0][readout[num_seq_labels + num_dense_labels + i].task_name] = compute_r2(regr_targets[:, i].view(-1), pred.squeeze()) 

        num_regression_dense_targets = dense_regr_targets.shape[-1] # will be 0 if dense_regr_targets is empty
        for i in range(num_regression_dense_targets):
            pred = compute_readout(
                data=data,
                target_length=target_length,
                readout=readout[num_seq_labels + num_dense_labels + num_regression_targets + i],
                readout_input=downstream_input,
                task="seq2seq",
                dense_prediction=dense_prediction,
                single_readout=single_readout,
                pred_steps=pred_steps,
                full_spatial_readout=full_spatial_readout,
            )
            targets = dense_regr_targets[:, :, i].view(-1)
            regression_loss[1][readout[num_seq_labels + num_dense_labels + num_regression_targets + i].task_name] = regression_criterion(pred.squeeze(), targets)
            regression_r2[1][readout[num_seq_labels + num_dense_labels + num_regression_targets + i].task_name] = compute_r2(targets, pred.squeeze())
    else:
        logits = compute_readout(
            data=data,
            target_length=target_length,
            readout=readout,
            readout_input=downstream_input,
            task=task,
            dense_prediction=dense_prediction,
            single_readout=single_readout,
            pred_steps=pred_steps,
            full_spatial_readout=full_spatial_readout,
        )
        labels = labels.view(-1)
        if task == "seq2seq":
            # essential for LibriSpeech phone classification
            labels = labels[:logits.shape[0]]
        classification_loss = classifier_criterion(logits, labels)
        # filter out rows of labels that are -1
        logits = logits[labels != -1]
        labels = labels[labels != -1]
        classification_accuracy = (logits.argmax(dim=1) == labels).float().mean()

        num_classes = logits.shape[1]
        if num_classes < 5:
            classifier_acc5 = 1.0
        else:
            classifier_acc5 = (logits.topk(5, dim=1).indices == labels.unsqueeze(1)).float().max(dim=1).values.mean()

        regression_loss = 0.0
        regression_r2 = 0.0

    return classification_loss, classification_accuracy, classifier_acc5, regression_loss, regression_r2


def get_readout_input_dim(downstream_input,
                          enc_output_dim=None,
                          ctx_dim=None,
                          pred_steps=None,
                          dense_prediction=False,
                          prediction_target=None,
                          full_spatial_readout=False,
                          model=None,
                          single_readout=False,
                          input_spatial_size=0,
                          pred_target_dim_override=0,
                          ):
    """Get the input dimension for the readout network

    Args:
        downstream_input (str): The input to the downstream task. One of ["enc", "ctx", "pred"]
        args (argparse.Namespace): Command line arguments
        full_spatial_readout (bool): Whether to use full spatial readout
        model (RePLModel): The model
        single_readout (bool): Whether to use a single timestep readout (for predictor only)

    Returns:
        int: The input dimension
    """
    if downstream_input == "enc" or (downstream_input=="ctx" and isinstance(model.integrator, IdentityIntegrator)):
        num_features = enc_output_dim
        if full_spatial_readout:
            num_features *= model.encoder.get_output_spatial_shape(input_spatial_size) ** 2
    elif downstream_input == "ctx":
        num_features = ctx_dim
        if full_spatial_readout:
            encoder_out_spatial_size = model.encoder.get_output_spatial_shape(input_spatial_size)
            num_features *= model.integrator.get_output_spatial_shape(encoder_out_spatial_size) ** 2
    elif downstream_input == "pred":
        num_features = get_predictor_output_dim(ctx_dim, 
                                                enc_output_dim, 
                                                pred_steps, 
                                                dense_prediction, 
                                                prediction_target,
                                                single_readout=single_readout,
                                                pred_target_dim_override=pred_target_dim_override)
        if full_spatial_readout:
            NotImplementedError("Full spatial readout not implemented for predictor")
    else:
        return None
    return num_features


def prepare_readout(
        task=None,
        downstream_input=None,
        input_spatial_size=None,
        single_timestep_readout=False,
        full_spatial_readout=False,
        num_classes=None,
        seq_len=None,
        evaluate_concat_features=False,
        enc_output_dim=None,
        ctx_dim=None,
        pred_steps=None,
        dense_prediction=False,
        prediction_target=None,
        pred_target_dim_override=0,
        model=None,
        n_areas=None,
):
    """Prepare the readout network
    
    Args:
        task (str): The downstream task. One of ["phone", "speaker"]
        downstream_input (str): The input to the downstream task. One of ["enc", "ctx", "pred"]
        input_spatial_size (int): The input spatial size
        single_timestep_readout (bool): Whether to use a single timestep readout
        full_spatial_readout (bool): Whether to use full spatial readout
        num_classes (int): The number of classes
        seq_len (int): The sequence length
        evaluate_concat_features (bool): Whether to evaluate the concatenated features
        enc_output_dim (int): The encoder output dimension
        ctx_dim (int): The context dimension
        pred_steps (int): The number of prediction steps
        dense_prediction (bool): Whether to predict dense output
        prediction_target (str): The prediction target
        pred_target_dim_override (int): The prediction target dimension override (only used for inv loss)
        model (RePLModel): The model
        n_areas (int): The number of areas (for hierarchical models)

    Returns:
        nn.Module: The readout network
    """
    if n_areas is None:  # normal RePL model with deep encoder
        num_features = get_readout_input_dim(
            downstream_input=downstream_input, 
            enc_output_dim=enc_output_dim,
            ctx_dim=ctx_dim,
            pred_steps=pred_steps,
            dense_prediction=dense_prediction,
            prediction_target=prediction_target,
            single_readout=single_timestep_readout,
            full_spatial_readout=full_spatial_readout,
            model=model,
            input_spatial_size=input_spatial_size,
            pred_target_dim_override=pred_target_dim_override
        )
        if num_features is None:
            raise ValueError(f"Invalid online eval input: {downstream_input}")
        
        multitask = task == "multitask"
        
        if not multitask:
            concat_over_batch = task == "seq2seq"
            readout = LinearReadout(num_classes=num_classes, input_dim=num_features,
                                    seq_len=seq_len, concat_over_time=evaluate_concat_features,
                                    concat_over_batch=concat_over_batch)
        else:
            readout = torch.nn.ModuleList()
            # num_classes is a tuple of dictionaries with keys as task names (first dict for seq labels, second dict for dense labels)
            num_classes_seq_labels = num_classes[0]
            if num_classes_seq_labels is not None:
                for task_name in num_classes_seq_labels.keys():
                    readout.append(LinearReadout(num_classes=num_classes_seq_labels[task_name], input_dim=num_features,
                                                seq_len=seq_len, concat_over_time=False, task_name=task_name,
                                                concat_over_batch=False))
            num_classes_dense_labels = num_classes[1]
            if num_classes_dense_labels is not None:
                for task_name in num_classes_dense_labels.keys():
                    readout.append(LinearReadout(num_classes=num_classes_dense_labels[task_name], input_dim=num_features,
                                                seq_len=seq_len, concat_over_time=False, task_name=task_name,
                                                concat_over_batch=True))
            num_regression_targets = num_classes[2]
            if num_regression_targets is not None:
                for task_name in num_regression_targets.keys():
                    readout.append(LinearReadout(num_classes=1, input_dim=num_features,
                                                seq_len=seq_len, concat_over_time=False, task_name=task_name,
                                                concat_over_batch=False))
            num_regression_dense_targets = num_classes[3]
            if num_regression_dense_targets is not None:
                for task_name in num_regression_dense_targets.keys():
                    readout.append(LinearReadout(num_classes=1, input_dim=num_features,
                                                seq_len=seq_len, concat_over_time=False, task_name=task_name,
                                                concat_over_batch=True))
        return readout
    else:
        return prepare_greedy_readout(
            task=task,
            downstream_input=downstream_input,
            input_spatial_size=input_spatial_size,
            single_timestep_readout=single_timestep_readout,
            full_spatial_readout=full_spatial_readout,
            num_classes=num_classes,
            seq_len=seq_len,
            evaluate_concat_features=evaluate_concat_features,
            enc_output_dim=enc_output_dim,
            ctx_dim=ctx_dim,
            pred_steps=pred_steps,
            dense_prediction=dense_prediction,
            prediction_target=prediction_target,
            pred_target_dim_override=pred_target_dim_override,
            model=model,
            n_areas=n_areas
        )


def prepare_greedy_readout(
        task=None,
        downstream_input=None,
        input_spatial_size=None,
        single_timestep_readout=False,
        full_spatial_readout=False,
        num_classes=None,
        seq_len=None,
        evaluate_concat_features=False,
        enc_output_dim=None,
        ctx_dim=None,
        pred_steps=None,
        dense_prediction=False,
        prediction_target=None,
        pred_target_dim_override=0,
        model=None,
        n_areas=None,
):
    """Prepare the readout network for each area in a hierarchical RePL model.
    
    Args:
        task (str): The downstream task. One of ["phone", "speaker"]
        downstream_input (str): The input to the downstream task. One of ["enc", "ctx", "pred"]
        input_spatial_size (int): The input spatial size
        single_timestep_readout (bool): Whether to use a single timestep readout
        full_spatial_readout (bool): Whether to use full spatial readout
        num_classes (int): The number of classes
        seq_len (int): The sequence length
        evaluate_concat_features (bool): Whether to evaluate the concatenated features
        enc_output_dim (list): The encoder output dimension for each area
        ctx_dim (list): The context dimension for each area
        pred_steps (int): The number of prediction steps
        dense_prediction (bool): Whether to predict dense output
        prediction_target (str): The prediction target
        pred_target_dim_override (int): The prediction target dimension override (only used for inv loss)
        model (HierarchicalRePLModel): The model
        n_areas (int): The number of areas

    Returns:
        nn.Module: The readout network for each area
    """
    if len(enc_output_dim) == 1:
        enc_output_dim = enc_output_dim * n_areas
    if len(ctx_dim) == 1:
        ctx_dim = ctx_dim * n_areas
    input_spatial_size_i = input_spatial_size
    readout = torch.nn.ModuleList()
    for i in range(n_areas):
        if downstream_input == "enc" or (downstream_input=="ctx" and isinstance(model.areas[i].integrator, IdentityIntegrator)):
            num_features = enc_output_dim[i]
            if full_spatial_readout:
                encoder_out_spatial_size = model.areas[i].encoder.get_output_spatial_shape(input_spatial_size_i) ** 2
                num_features *= encoder_out_spatial_size
        elif downstream_input == "ctx":
            num_features = ctx_dim[i]
            if full_spatial_readout:
                encoder_out_spatial_size = model.areas[i].encoder.get_output_spatial_shape(input_spatial_size_i)
                num_features *= model.areas[i].integrator.get_output_spatial_shape(encoder_out_spatial_size) ** 2
        elif downstream_input == "pred":
            num_features = get_predictor_output_dim(ctx_dim[i], 
                                                    enc_output_dim[i], 
                                                    pred_steps, 
                                                    dense_prediction, 
                                                    prediction_target,
                                                    single_readout=single_timestep_readout,
                                                    pred_target_dim_override=pred_target_dim_override)
            if full_spatial_readout:
                NotImplementedError("Full spatial readout not implemented for predictor")
        else:
            num_features = None

        # input_spatial_size_i = encoder_out_spatial_size
        if num_features is None:
            raise ValueError(f"Invalid online eval input: {downstream_input}")

        multitask = task == "multitask"

        if not multitask:
            concat_over_batch = task == "seq2seq"
            readout.append(LinearReadout(num_classes=num_classes, input_dim=num_features,
                                        seq_len=seq_len, concat_over_time=evaluate_concat_features,
                                        concat_over_batch=concat_over_batch))
        else:
            readout_i = torch.nn.ModuleList()
            # num_classes is a tuple of dictionaries with keys as task names (first dict for seq labels, second dict for dense labels)
            num_classes_seq_labels = num_classes[0]
            if num_classes_seq_labels is not None:
                for task_name in num_classes_seq_labels.keys():
                    readout_i.append(LinearReadout(num_classes=num_classes_seq_labels[task_name], input_dim=num_features,
                                                seq_len=seq_len, concat_over_time=False, task_name=task_name,
                                                concat_over_batch=False))
            num_classes_dense_labels = num_classes[1]
            if num_classes_dense_labels is not None:
                for task_name in num_classes_dense_labels.keys():
                    readout_i.append(LinearReadout(num_classes=num_classes_dense_labels[task_name], input_dim=num_features,
                                                seq_len=seq_len, concat_over_time=False, task_name=task_name,
                                                concat_over_batch=True))
            num_classes_regr = num_classes[2]
            if num_classes_regr is not None:
                for task_name in num_classes_regr.keys():
                    readout_i.append(LinearReadout(num_classes=num_classes_regr[task_name], input_dim=num_features,
                                                seq_len=seq_len, concat_over_time=False, task_name=task_name,
                                                concat_over_batch=False))
            num_classes_dense_regr = num_classes[3]
            if num_classes_dense_regr is not None:
                for task_name in num_classes_dense_regr.keys():
                    readout_i.append(LinearReadout(num_classes=num_classes_dense_regr[task_name], input_dim=num_features,
                                                seq_len=seq_len, concat_over_time=False, task_name=task_name,
                                                concat_over_batch=True))
            readout.append(readout_i)
    return readout


def get_repl_representation(dataloader, model=None, downstream_input="ctx", label_types=(0,1,2,3), device="cpu"):
    """Get the model representation for the downstream tasks (multitask only)

    Args:
        dataloader (DataLoader): The dataloader
        model (RePLModel): The model
        downstream_input (str): The model component to use for the downstream task. One of ["raw", "enc", "ctx", "pred"]
        label_types (tuple): The label types to return (cf. num_classes)
        device (str): The device to use

    Returns:
        data (np.ndarray): The model representation
        labels (list of np.ndarray): The labels for each label type
    """
    data = []
    labels = [[] for _ in range(len(label_types))]  # Initialize labels for each type
    for i, (batch_data, batch_labels) in enumerate(dataloader):
        batch_data = batch_data.to(device)
        if downstream_input == "raw" or model is None:
            data.append(batch_data.cpu().numpy().reshape(batch_data.shape[0], batch_data.shape[1], -1))
        else:
            with torch.no_grad():
                z, ctx, pred = model(batch_data)

            if downstream_input == "enc":
                data.append(z.cpu().numpy())
            elif downstream_input == "ctx":
                data.append(ctx.cpu().numpy())
            elif downstream_input == "pred":
                data.append(pred.cpu().numpy())
            else:
                raise ValueError(f"Invalid downstream input: {downstream_input}")

        for j, label_type in enumerate(label_types):
            if label_type < len(batch_labels):
                labels[j].append(batch_labels[label_type].cpu().numpy().squeeze())

    data = np.concatenate(data, axis=0)
    labels = [np.concatenate(label, axis=0) for label in labels]
    return data, labels


def get_greedy_repl_representation(dataloader, model, downstream_input="ctx", label_types=(0,1,2,3), device="cpu"):
    """Get the model representation for the downstream tasks (multitask only) for each area in a hierarchical RePL model
    
    Args:
        dataloader (DataLoader): The dataloader
        model (HierarchicalRePLModel): The model
        downstream_input (str): The model component to use for the downstream task. One of ["raw", "enc", "ctx", "pred"]
        label_types (tuple): The label types to return (cf. num_classes)
        device (str): The device to use

    Returns:
        data (list of np.ndarray): The model representation for each area
        labels (list of np.ndarray): The labels for each label type
    """
    data = [[] for _ in range(model.n_areas)]  # Initialize data for each area
    labels = [[] for _ in range(len(label_types))]  # Initialize labels for each type
    for i, (batch_data, batch_labels) in enumerate(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            z_list, ctx_list, pred_list = model(batch_data)
        
        for j in range(model.n_areas):
            if downstream_input == "enc":
                data[j].append(z_list[j].cpu().numpy())
            elif downstream_input == "ctx":
                data[j].append(ctx_list[j].cpu().numpy())
            elif downstream_input == "pred":
                data[j].append(pred_list[j].cpu().numpy())
            else:
                raise ValueError(f"Invalid downstream input: {downstream_input}")
        
        for j, label_type in enumerate(label_types):
            if label_type < len(batch_labels):
                labels[j].append(batch_labels[label_type].cpu().numpy().squeeze())

    data = [np.concatenate(area_data, axis=0) for area_data in data]
    labels = [np.concatenate(label, axis=0) for label in labels]
    return data, labels


def get_linear_regression_metrics(x, y_true, regressor):
    """Compute the regression loss and R2 score for linear regression
    
    Args:
        x (np.ndarray): The input data
        y_true (np.ndarray): The true labels
        regressor (LinearRegression): The linear regressor object

    Returns:
        tuple: The regression loss and R2 score per task
    """
    from sklearn.metrics import r2_score
    y_pred = regressor.predict(x)
    regression_loss = np.mean((y_pred - y_true) ** 2, axis=0)  # Mean Squared Error per label
    r2_score_value = r2_score(y_true, y_pred, multioutput='raw_values')  # R2 score per label
    return regression_loss, r2_score_value


def offline_linear_regression(
        train_dataloader,
        test_dataloader,
        model=None,
        num_classes=None,
        downstream_input="ctx",
        device="cpu"
):
    """Perform offline linear regression on the model's output

    Args:
        train_dataloader (DataLoader): The training dataloader
        test_dataloader (DataLoader): The testing dataloader
        model (RePLModel): The model
        num_classes (tuple): The number of classes for each task
        downstream_input (str): The model component to use for the downstream task. One of ["raw", "enc", "ctx", "pred"]
        device (str): The device to use

    Returns:
        dict: The regression losses and R2 scores for dense and seq labels
            - dense_regression_loss_train (np.ndarray): The regression loss for dense labels (seq2seq tasks) on the training set
            - dense_r2_score_train (np.ndarray): The R2 score for dense labels (seq2seq tasks) on the training set
            - dense_regression_loss_test (np.ndarray): The regression loss for dense labels (seq2seq tasks) on the test set
            - dense_r2_score_test (np.ndarray): The R2 score for dense labels (seq2seq tasks) on the test set
            - seq_regression_loss_train (np.ndarray): The regression loss for sequence labels (seq2label tasks) on the training set
            - seq_r2_score_train (np.ndarray): The R2 score for sequence labels (seq2label tasks) on the training set
            - seq_regression_loss_test (np.ndarray): The regression loss for sequence labels (seq2label tasks) on the test set
            - seq_r2_score_test (np.ndarray): The R2 score for sequence labels (seq2label tasks) on the test set
            - dense_regressor (LinearRegression): The linear regressor for seq2seq tasks
            - seq_regressor (LinearRegression): The linear regressor for seq2label tasks
    """
    from sklearn.linear_model import LinearRegression


    n_dense_labels = len(num_classes[3]) if num_classes is not None and len(num_classes) > 3 else 0
    n_seq_labels = len(num_classes[2]) if num_classes is not None and len(num_classes) > 0 else 0

    assert n_dense_labels > 0 or n_seq_labels > 0, "At least one dense or seq label is required for offline linear regression"

    readout_train_data, (readout_train_seq_labels, readout_train_dense_labels) = get_repl_representation(
        train_dataloader,
        model,
        downstream_input=downstream_input,
        label_types=(2, 3),  # seq labels and dense labels
        device=device
    )
    readout_test_data, (readout_test_seq_labels, readout_test_dense_labels) = get_repl_representation(
        test_dataloader,
        model,
        downstream_input=downstream_input,
        label_types=(2, 3),  # seq labels and dense labels
        device=device
    )

    # normalize the data (only for more interpretable weights)
    train_mean = readout_train_data.mean(axis=(0,1))
    train_std = readout_train_data.std(axis=(0,1))
    readout_train_data = (readout_train_data - train_mean) / train_std
    readout_test_data = (readout_test_data - train_mean) / train_std

    n_features = readout_train_data.shape[2]

    dense_regression_loss_train, dense_regression_loss_test = None, None
    dense_r2_score_train, dense_r2_score_test = None, None
    seq_regression_loss_train, seq_regression_loss_test = None, None
    seq_r2_score_train, seq_r2_score_test = None, None

    dense_regressor = LinearRegression()
    seq_regressor = LinearRegression()

    if n_dense_labels > 0:
        dense_regressor.fit(readout_train_data.reshape((-1, n_features)), readout_train_dense_labels.reshape((-1, n_dense_labels)))
        dense_regression_loss_train, dense_r2_score_train = get_linear_regression_metrics(
            readout_train_data.reshape((-1, n_features)),
            readout_train_dense_labels.reshape((-1, n_dense_labels)),
            dense_regressor
        )
        dense_regression_loss_test, dense_r2_score_test = get_linear_regression_metrics(
            readout_test_data.reshape((-1, n_features)),
            readout_test_dense_labels.reshape((-1, n_dense_labels)),
            dense_regressor
        )

    if n_seq_labels > 0:
        seq_regressor.fit(readout_train_data.mean(axis=1), readout_train_seq_labels)  # mean over time for seq labels
        seq_regression_loss_train, seq_r2_score_train = get_linear_regression_metrics(
            readout_train_data.mean(axis=1),
            readout_train_seq_labels,
            seq_regressor
        )
        seq_regression_loss_test, seq_r2_score_test = get_linear_regression_metrics(
            readout_test_data.mean(axis=1),
            readout_test_seq_labels,
            seq_regressor
        )
    
    return {
        "dense_regression_loss_train": dense_regression_loss_train,
        "dense_r2_score_train": dense_r2_score_train,
        "dense_regression_loss_test": dense_regression_loss_test,
        "dense_r2_score_test": dense_r2_score_test,
        "seq_regression_loss_train": seq_regression_loss_train,
        "seq_r2_score_train": seq_r2_score_train,
        "seq_regression_loss_test": seq_regression_loss_test,
        "seq_r2_score_test": seq_r2_score_test,
        "dense_regressor": dense_regressor,
        "seq_regressor": seq_regressor
    }


def offline_greedy_linear_regression(
        train_dataloader,
        test_dataloader,
        model,
        num_classes=None,
        downstream_input="ctx",
        device="cpu",
):
    """Perform offline linear regression on the model's output for a specific area

    Args:
        train_dataloader (DataLoader): The training dataloader
        test_dataloader (DataLoader): The testing dataloader
        model (RePLModel): The model
        num_classes (tuple): The number of classes for each task
        downstream_input (str): The model component to use for the downstream task. One of ["raw", "enc", "ctx", "pred"]
        device (str): The device to use

    Returns:
        dict: The regression losses and R2 scores for dense and seq labels
            - dense_regression_loss_train (np.ndarray): The regression loss for dense labels (seq2seq tasks) on the training set
            - dense_r2_score_train (np.ndarray): The R2 score for dense labels (seq2seq tasks) on the training set
            - dense_regression_loss_test (np.ndarray): The regression loss for dense labels (seq2seq tasks) on the test set
            - dense_r2_score_test (np.ndarray): The R2 score for dense labels (seq2seq tasks) on the test set
            - seq_regression_loss_train (np.ndarray): The regression loss for sequence labels (seq2label tasks) on the training set
            - seq_r2_score_train (np.ndarray): The R2 score for sequence labels (seq2label tasks) on the training set
            - seq_regression_loss_test (np.ndarray): The regression loss for sequence labels (seq2label tasks) on the test set
            - seq_r2_score_test (np.ndarray): The R2 score for sequence labels (seq2label tasks) on the test set
            - dense_regressor (list of LinearRegression): The linear regressor for seq2seq tasks for each area
            - seq_regressor (list of LinearRegression): The linear regressor for seq2label tasks for each area
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    n_dense_labels = len(num_classes[3]) if num_classes is not None and len(num_classes) > 3 else 0
    n_seq_labels = len(num_classes[2]) if num_classes is not None and len(num_classes) > 0 else 0

    assert n_dense_labels > 0 or n_seq_labels > 0, "At least one dense or seq label is required for offline linear regression"

    readout_train_data, (readout_train_seq_labels, readout_train_dense_labels) = get_greedy_repl_representation(
        train_dataloader,
        model,
        downstream_input=downstream_input,
        label_types=(2, 3),  # seq labels and dense labels
        device=device
    )
    readout_test_data, (readout_test_seq_labels, readout_test_dense_labels) = get_greedy_repl_representation(
        test_dataloader,
        model,
        downstream_input=downstream_input,
        label_types=(2, 3),  # seq labels and dense labels
        device=device
    )
    # normalize the data (only for more interpretable weights)
    train_mean = [area_data.mean(axis=(0, 1)) for area_data in readout_train_data]
    train_std = [area_data.std(axis=(0, 1)) for area_data in readout_train_data]
    readout_train_data = [((area_data - mean) / std) for area_data, mean, std in zip(readout_train_data, train_mean, train_std)]
    readout_test_data = [((area_data - mean) / std) for area_data, mean, std in zip(readout_test_data, train_mean, train_std)]

    n_features = readout_train_data[0].shape[2]
    dense_regression_loss_train, dense_regression_loss_test = None, None
    dense_r2_score_train, dense_r2_score_test = None, None
    seq_regression_loss_train, seq_regression_loss_test = None, None
    seq_r2_score_train, seq_r2_score_test = None, None
    dense_regressor = [LinearRegression() for _ in range(model.n_areas)]
    seq_regressor = [LinearRegression() for _ in range(model.n_areas)]

    if n_dense_labels > 0:
        dense_regression_loss_train = np.zeros((model.n_areas, n_dense_labels))
        dense_r2_score_train = np.zeros((model.n_areas, n_dense_labels))
        dense_regression_loss_test = np.zeros((model.n_areas, n_dense_labels))
        dense_r2_score_test = np.zeros((model.n_areas, n_dense_labels))

        for i in range(model.n_areas):
            dense_regressor[i].fit(readout_train_data[i].reshape((-1, n_features)), readout_train_dense_labels.reshape((-1, n_dense_labels)))
            dense_regression_loss_train[i], dense_r2_score_train[i] = get_linear_regression_metrics(
                readout_train_data[i].reshape((-1, n_features)),
                readout_train_dense_labels.reshape((-1, n_dense_labels)),
                dense_regressor[i]
            )
            dense_regression_loss_test[i], dense_r2_score_test[i] = get_linear_regression_metrics(
                readout_test_data[i].reshape((-1, n_features)),
                readout_test_dense_labels.reshape((-1, n_dense_labels)),
                dense_regressor[i]
            )

    if n_seq_labels > 0:
        seq_regression_loss_train = np.zeros((model.n_areas, n_seq_labels))
        seq_r2_score_train = np.zeros((model.n_areas, n_seq_labels))
        seq_regression_loss_test = np.zeros((model.n_areas, n_seq_labels))
        seq_r2_score_test = np.zeros((model.n_areas, n_seq_labels))

        for i in range(model.n_areas):
            seq_regressor[i].fit(readout_train_data[i].mean(axis=1), readout_train_seq_labels)
            seq_regression_loss_train[i], seq_r2_score_train[i] = get_linear_regression_metrics(
                readout_train_data[i].mean(axis=1),
                readout_train_seq_labels,
                seq_regressor[i]
            )
            seq_regression_loss_test[i], seq_r2_score_test[i] = get_linear_regression_metrics(
                readout_test_data[i].mean(axis=1),
                readout_test_seq_labels,
                seq_regressor[i]
            )

    return {
        "dense_regression_loss_train": dense_regression_loss_train,
        "dense_r2_score_train": dense_r2_score_train,
        "dense_regression_loss_test": dense_regression_loss_test,
        "dense_r2_score_test": dense_r2_score_test,
        "seq_regression_loss_train": seq_regression_loss_train,
        "seq_r2_score_train": seq_r2_score_train,
        "seq_regression_loss_test": seq_regression_loss_test,
        "seq_r2_score_test": seq_r2_score_test,
        "dense_regressor": dense_regressor,
        "seq_regressor": seq_regressor
    }
