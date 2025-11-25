import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from tqdm.notebook import tqdm


def compute_pca_projection(representations, n_components=3):
    """
    Compute PCA projection for a single representation vector.
    
    Args:
        representations: numpy array of shape (n_samples, n_features)
        n_components: number of PCA components to compute (default: 3)
        
    Returns:
        dict: Contains PCA object and projected representations
            - 'pca': fitted sklearn PCA object
            - 'projections': projected representations (n_samples, n_components)
            - 'explained_variance_ratio': explained variance ratios for components
    """
    from sklearn.decomposition import PCA


    # Fit PCA
    pca = PCA(n_components=n_components)
    projections = pca.fit_transform(representations)
    
    return {
        'pca': pca,
        'projections': projections,
        'explained_variance_ratio': pca.explained_variance_ratio_
    }


def compute_umap_projection(representations, n_components=3, n_neighbors=5, min_dist=0.5, metric='euclidean', densmap=False):
    """
    Compute UMAP projection with optimized parameters.
    
    Args:
        representations: numpy array of shape (n_samples, n_features)
        n_components: number of UMAP components to compute (default: 3)
        n_neighbors: number of neighbors to consider for UMAP (default: 5)
        min_dist: minimum distance between points in UMAP (default: 0.5)
        
    Returns:
        dict: Contains UMAP object and projected representations
    """
    import umap.umap_ as umap
    import numpy as np
    from sklearn.decomposition import PCA
    
    print("Computing UMAP projection...")
    
    # Optional: Preprocessing with PCA if dimensions are very high
    n_samples, n_features = representations.shape
    if n_features > 100:
        print(f"Applying PCA preprocessing to reduce dimensions from {n_features} to 100")
        pca = PCA(n_components=100)
        representations = pca.fit_transform(representations)
    
    # Optional: subsample data for faster computation
    if n_samples > 100000:
        print("Subsampling data for faster computation")
        idx = np.random.choice(n_samples, 100000, replace=False)
        representations_ = representations[idx]
    else:
        representations_ = representations

    # Fit UMAP with optimized parameters
    umap_obj = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors, 
        min_dist=min_dist,
        n_jobs=-1,  # Use all available cores
        low_memory=False,  # Faster but uses more memory
        n_epochs=200,  # Reduce from default (usually 500-1000)
        init='random',  # Faster than 'spectral'
        verbose=True,
        metric=metric,
        densmap=densmap
    )
    if densmap:
        projections = umap_obj.fit(representations_).embedding_
    else:
        projections = umap_obj.fit(representations_).transform(representations)
    
    return {
        'umap': umap_obj,
        'projections': projections
    }


def compute_corrected_orientation_regressor_projections(
    train_representations,
    train_gt_orientations,
    train_sprite_labels,
    test_representations,
    test_gt_orientations,
    test_sprite_labels,
    learning_rate=0.01,
    n_epochs=5000,
    device='cpu',
    num_sprite_classes=8,
    orientation_discr_levels=36,
):
    """
    Compute projections for sin(theta + delta_c) and cos(theta + delta_c)
    where delta_c are class-specific trainable phase shifts.
    
    Args:
        train_representations: Input features for training (n_samples, n_features)
        train_gt_orientations: Ground truth orientations in radians for training (n_samples, 1)
        train_sprite_labels: Discrete class labels for training (n_samples, 1)
        test_representations: Input features for testing (n_samples, n_features)
        test_gt_orientations: Ground truth orientations in radians for testing (n_samples, 1)
        test_sprite_labels: Discrete class labels for testing (n_samples, 1)
        learning_rate: Learning rate for optimization
        n_epochs: Number of training epochs
        device: 'cpu' or 'cuda' for torch computations
        num_sprite_classes: Number of sprite classes (default: 8)
        orientation_discr_levels: Number of discrete levels for orientation (default: 36)
        
    Returns:
        Tuple containing predictions and metrics:
        (predictions_sin, predictions_cos, predictions_discr, 
         targets_sin, targets_cos, targets_discr, 
         r2_sin, r2_cos, acc_discr, phase_shifts)
    """

    final_weights, final_intercepts, final_clf_weights, final_clf_intercepts, phase_shifts_params = train_corrected_orientation_regressor(
        train_representations,
        train_gt_orientations,
        train_sprite_labels,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        device=device,
        num_sprite_classes=num_sprite_classes,
        orientation_discr_levels=orientation_discr_levels
    )

    # Get final parameters
    final_phase_shifts = phase_shifts_params

    final_projections, final_projections_discr, final_target_sin, final_target_cos, shifted_orientations_discr = get_corrected_orientation_projections_test(
        test_representations,
        test_gt_orientations,
        test_sprite_labels,
        final_weights,
        final_intercepts,
        final_clf_weights,
        final_clf_intercepts,
        final_phase_shifts,
        orientation_discr_levels=orientation_discr_levels
    )
    
    # Calculate metrics
    if final_target_sin.size > 0 and final_projections.shape[0] > 0:
        mse_sin = np.mean((final_projections[:, 0] - final_target_sin.flatten())**2)
        mse_cos = np.mean((final_projections[:, 1] - final_target_cos.flatten())**2)
        r_squared_sin = r2_score(final_target_sin.flatten(), final_projections[:, 0])
        r_squared_cos = r2_score(final_target_cos.flatten(), final_projections[:, 1])
        clf_acc = np.mean(final_projections_discr == shifted_orientations_discr)
    else:
        r_squared_sin, r_squared_cos = np.nan, np.nan

    readout_dict = {
        'phase_shifts': final_phase_shifts.cpu().numpy(),
        'sin': {
            'weights': final_weights[:, 0].cpu().numpy(),
            'intercept': final_intercepts[0].cpu().numpy(),
        },
        'cos': {
            'weights': final_weights[:, 1].cpu().numpy(),
            'intercept': final_intercepts[1].cpu().numpy(),
        },
        'discr': {
            'weights': final_clf_weights.cpu().numpy(),
            'intercept': final_clf_intercepts.cpu().numpy()
        }
    }
    
    return (
        final_projections[:, 0],   # predictions_sin
        final_projections[:, 1],   # predictions_cos
        final_projections_discr,   # predictions_discr
        final_target_sin,          # targets_sin
        final_target_cos,          # targets_cos
        shifted_orientations_discr,# targets_discr
        r_squared_sin,             # r2_sin
        r_squared_cos,             # r2_cos
        clf_acc,                   # acc_discr
        final_phase_shifts,         # phase_shifts
        readout_dict               # readout_dict
    )


def train_corrected_orientation_regressor(train_representations,
                                          train_gt_orientations,
                                          train_sprite_labels,
                                          learning_rate=0.01,
                                          n_epochs=5000,
                                          device='cpu',
                                          num_sprite_classes=8,
                                          orientation_discr_levels=36):
    """Train regressors for phase-shift corrected orientation."""
    n_samples, n_features = train_representations.shape

    # Map sprite labels directly
    mapped_sprite_idx = train_sprite_labels.reshape(-1).long()

    # Handle edge cases
    if n_samples == 0:
        empty_result = np.array([])
        return empty_result, empty_result, None, empty_result, empty_result, None, np.nan, np.nan, 0.0, empty_result
    
    # Prepare tensors for training
    representations_tensor = train_representations.clone().detach()
    gt_orientations_tensor = train_gt_orientations.clone().detach().squeeze()
    mapped_sprite_idx_tensor = mapped_sprite_idx.to(device)
    
    # Initialize trainable parameters
    weights = torch.nn.Parameter(torch.randn(n_features, 2, device=device, dtype=torch.float32) * 0.01)
    intercepts = torch.nn.Parameter(torch.zeros(2, device=device, dtype=torch.float32))
    clf_weights = torch.nn.Parameter(torch.randn(n_features, orientation_discr_levels, device=device, dtype=torch.float32) * 0.01)
    clf_intercepts = torch.nn.Parameter(torch.zeros(orientation_discr_levels, device=device, dtype=torch.float32))
    phase_shifts_params = torch.nn.Parameter(torch.randn(num_sprite_classes, device=device, dtype=torch.float32) * 0.01)
    
    # Set up optimizer and loss function
    optimizer = optim.AdamW([weights, intercepts, clf_weights, clf_intercepts, phase_shifts_params], lr=learning_rate, weight_decay=1e-3)
    mse_loss_fn = nn.MSELoss()
    clf_loss_fn = nn.CrossEntropyLoss()
    
    # Training loop with progress bar
    epoch_pbar = tqdm(range(n_epochs), desc="Corrected Orientation Training")
    for epoch in epoch_pbar:
        optimizer.zero_grad()
        
        # Apply phase shifts based on class labels
        selected_phase_shifts = phase_shifts_params[mapped_sprite_idx_tensor]
        shifted_orientations = gt_orientations_tensor + selected_phase_shifts

        shifted_orientations_discr = shifted_orientations / (2 * np.pi) * orientation_discr_levels
        shifted_orientations_discr = shifted_orientations_discr.round().long() % orientation_discr_levels
        shifted_orientations_discr = shifted_orientations_discr.view(-1).to(device)
        
        # Create targets for sin and cos predictions
        target_sin = torch.sin(shifted_orientations).unsqueeze(1)
        target_cos = torch.cos(shifted_orientations).unsqueeze(1)
        targets = torch.cat((target_sin, target_cos), dim=1)
        
        # Forward pass
        predictions = representations_tensor @ weights + intercepts
        predictions_discr = representations_tensor @ clf_weights + clf_intercepts
        predictions_discr = predictions_discr.view(-1, orientation_discr_levels)
        
        # Compute loss and backpropagate
        loss = mse_loss_fn(predictions, targets)
        clf_loss = clf_loss_fn(predictions_discr, shifted_orientations_discr)
        total_loss = loss + clf_loss
        total_loss.backward()
        optimizer.step()

        # Update progress bar with current losses
        epoch_pbar.set_postfix(total_loss=f"{total_loss.item():.6f}", mse_loss=f"{loss.item():.6f}", clf_loss=f"{clf_loss.item():.6f}")
    return (
        weights.detach(),
        intercepts.detach(),
        clf_weights.detach(),
        clf_intercepts.detach(),
        phase_shifts_params.detach()
    )


def get_corrected_orientation_projections_test(test_representations,
                                                test_gt_orientations,
                                                test_sprite_labels,
                                                weights,
                                                intercepts,
                                                clf_weights,
                                                clf_intercepts,
                                                phase_shifts_params,
                                                orientation_discr_levels=36):
    """Get phase-corrected orientation projections for test representations."""
    # Process test data
    test_representations_tensor = test_representations.clone().detach()
    test_gt_orientations_tensor = test_gt_orientations.clone().detach().reshape(-1)
    test_sprite_labels_tensor = test_sprite_labels.clone().detach().reshape(-1).long()

    # Generate test predictions
    final_projections_tensor = test_representations_tensor @ weights + intercepts
    final_projections = final_projections_tensor.cpu().numpy()
    final_projections_discr = test_representations_tensor @ clf_weights + clf_intercepts
    final_projections_discr = final_projections_discr.view(-1, orientation_discr_levels).cpu().numpy()
    final_projections_discr = final_projections_discr.argmax(axis=1)
    
    # Apply phase shifts to test data
    final_selected_phase_shifts = phase_shifts_params[test_sprite_labels_tensor]
    shifted_test_orientations = test_gt_orientations_tensor + final_selected_phase_shifts
    final_target_sin = torch.sin(shifted_test_orientations).unsqueeze(1).cpu().numpy()
    final_target_cos = torch.cos(shifted_test_orientations).unsqueeze(1).cpu().numpy()
    shifted_orientations_discr = shifted_test_orientations / (2 * np.pi) * orientation_discr_levels
    shifted_orientations_discr = shifted_orientations_discr.round().long() % orientation_discr_levels
    shifted_orientations_discr = shifted_orientations_discr.view(-1).cpu().numpy()

    return (
        final_projections,
        final_projections_discr,
        final_target_sin,
        final_target_cos,
        shifted_orientations_discr
    )


def prepare_taskwise_dicts(num_classes):
    """
    Create dictionaries for each task type with empty lists as values.
    
    Args:
        num_classes: List of dictionaries containing task names as keys
        
    Returns:
        List of dictionaries for each task type
    """
    out_dicts = [{} for _ in range(len(num_classes))]
    
    # Populate dictionaries for each task type
    for i, class_dict in enumerate(num_classes):
        if class_dict is not None:
            for task_name in class_dict.keys():
                out_dicts[i][task_name] = []
                
    return out_dicts


def extract_data(model, readouts, device, readout_input_mean_train, readout_input_std_train, 
                 data_loader, num_classes, seq_len, repr_dim, train_set=False,
                 max_samples=None):
    """
    Extract data from model and compute projections using readouts if required.
    
    Args:
        model: The model to extract representations from
        readouts: Readout modules to apply to representations
        device: Device to use for computation
        readout_input_mean_train: Mean for normalizing representations
        readout_input_std_train: Std for normalizing representations
        data_loader: DataLoader containing the data
        num_classes: Dictionary of classes for each task
        seq_len: Sequence length
        repr_dim: Representation dimension
        train_set: Whether this is a training set (don't compute projections) or testing set
        max_samples: Maximum number of samples to process (for pixel readout, needed to avoid memory issues)
    
    Returns:
        Tuple of (representations, gt_orientations, sprite_labels, latent_labels, projections, performance)
    """
    batch_size = data_loader.batch_size
    num_samples = len(data_loader.dataset) if max_samples is None else min(max_samples, len(data_loader.dataset))
    
    # Initialize tensors to hold data
    representations = torch.zeros(num_samples, seq_len, repr_dim).to(device)
    gt_orientations_raw = torch.zeros(num_samples, seq_len, 1).to(device)
    sprite_labels_raw = torch.zeros(num_samples, 1, dtype=torch.long).to(device)
    
    # Initialize dictionaries for results
    latent_labels = prepare_taskwise_dicts(num_classes)
    projections = prepare_taskwise_dicts(num_classes)
    performance = prepare_taskwise_dicts(num_classes)
    
    current_sample_idx = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(data_loader):
            data = data.to(device)
            repr_ = model(data)[1]
            
            # Calculate indices for storing batch data
            start_idx = i * batch_size
            end_idx_batch = min(batch_size, num_samples - start_idx)
            end_idx = start_idx + end_idx_batch

            # Check for maximum samples
            if max_samples is not None and start_idx >= max_samples:
                break

            # Ensure we don't go out of bounds
            if end_idx > representations.shape[0]:
                end_idx_batch = representations.shape[0] - start_idx
                end_idx = representations.shape[0]
            
            # Store representations and labels
            representations[start_idx:end_idx] = repr_[:end_idx_batch]
            sprite_labels_raw[start_idx:end_idx] = labels[0][:end_idx_batch, 0].to(device).unsqueeze(1)
            gt_orientations_raw[start_idx:end_idx] = labels[4][:end_idx_batch].to(device) * 2 * np.pi
            
            current_sample_idx = end_idx
            
            # Compute projections for test set
            if not train_set:
                
                if readout_input_mean_train is not None and readout_input_std_train is not None:
                    # Normalize representations
                    repr = (repr_ - readout_input_mean_train) / (readout_input_std_train + 1e-8)
                else:
                    repr = repr_

                # Move labels to device and unpack them
                labels = tuple([l.to(device) for l in labels])
                seq_labels = labels[0]                                # Shape: [batch_size, 2]
                dense_labels = labels[1].view(-1, labels[1].shape[-1])  # Shape: [batch_size, 4]
                cts_labels = labels[2]                                # Shape: [batch_size, 2]
                cts_dense_labels = labels[3].view(-1, labels[3].shape[-1])  # Shape: [batch_size, 8]
                
                # Track indexes for each type of readout
                readout_idxes = [0, 0, 0, 0]
                
                # Apply each readout
                for readout in readouts:
                    projection = readout(repr)
                    task_name = readout.task_name
                    
                    # Store results in appropriate dictionary based on task type
                    if task_name in latent_labels[0]:
                        latent_labels[0][task_name].append(seq_labels[:, readout_idxes[0]])
                        readout_idxes[0] += 1
                        projections[0][task_name].append(projection)
                    elif task_name in latent_labels[1]:
                        latent_labels[1][task_name].append(dense_labels[:, readout_idxes[1]])
                        readout_idxes[1] += 1
                        projections[1][task_name].append(projection)
                    elif task_name in latent_labels[2]:
                        latent_labels[2][task_name].append(cts_labels[:, readout_idxes[2]])
                        readout_idxes[2] += 1
                        projections[2][task_name].append(projection)
                    elif task_name in latent_labels[3]:
                        latent_labels[3][task_name].append(cts_dense_labels[:, readout_idxes[3]])
                        readout_idxes[3] += 1
                        projections[3][task_name].append(projection)
                    else:
                        raise ValueError(f"Unknown task name: {task_name}")
    
    # Process results for test set
    if not train_set:
        for i in range(len(latent_labels)):
            for task_name in latent_labels[i]:
                # Concatenate data from all batches
                latent_labels[i][task_name] = torch.cat(latent_labels[i][task_name], dim=0).cpu().numpy()
                projections[i][task_name] = torch.cat(projections[i][task_name], dim=0).cpu().numpy()
                
                # Compute appropriate performance metrics
                if i in [0, 1]:  # Classification tasks
                    performance[i][task_name] = (latent_labels[i][task_name] == 
                                                projections[i][task_name].argmax(axis=1)).mean()
                elif i in [2, 3]:  # Regression tasks
                    performance[i][task_name] = r2_score(latent_labels[i][task_name], 
                                                        projections[i][task_name])
    
    # Trim unused parts of pre-allocated tensors
    if current_sample_idx < representations.shape[0]:
        representations = representations[:current_sample_idx]
        gt_orientations_raw = gt_orientations_raw[:current_sample_idx]
        sprite_labels_raw = sprite_labels_raw[:current_sample_idx]
    
    # Reshape for output
    representations = representations.reshape(-1, repr_dim)
    gt_orientations_raw = gt_orientations_raw.reshape(-1, 1)
    sprite_labels_raw = sprite_labels_raw.repeat(1, seq_len).reshape(-1, 1)
    
    return representations, gt_orientations_raw, sprite_labels_raw, latent_labels, projections, performance


def compute_readout_projections(model, 
                                readouts, 
                                device,
                                readout_input_mean_train, 
                                readout_input_std_train, 
                                data_loader, 
                                num_classes, 
                                train_loader,
                                seq_len=32,
                                repr_dim=512,
                                orientation_lr=0.01,
                                orientation_n_epochs=2000,
                                class_idxes=[0, 7],
                                max_samples=None):
    """
    Compute readout projections for a model on given data.
    
    Args:
        model: The model to use for feature extraction
        readouts: Readout modules to apply
        device: Device to use for computation
        readout_input_mean_train: Mean for normalizing representations
        readout_input_std_train: Std for normalizing representations
        data_loader: Test data loader
        num_classes: Dictionary of classes for each task
        train_loader: Training data loader
        seq_len: Sequence length
        repr_dim: Representation dimension
        orientation_lr: Learning rate for orientation correction
        orientation_n_epochs: Number of epochs for orientation correction
        class_idxes: List containing the two class indices for binary classification used in 3d projection plots
        max_samples: Maximum number of samples to process (for pixel readout, needed to avoid memory issues)
        
    Returns:
        Tuple of (projections, latent_labels, performance, phase_shifts)
    """
    model.eval()
    readouts.eval()
    
    # Extract test data
    test_representations, test_gt_orientations_raw, test_sprite_labels_raw, latent_labels, projections, performance = extract_data(
        model, 
        readouts, 
        device, 
        readout_input_mean_train, 
        readout_input_std_train, 
        data_loader, 
        num_classes, 
        seq_len, 
        repr_dim,
        train_set=False
    )
    
    # Extract training data
    train_representations, train_gt_orientations_raw, train_sprite_labels_raw, _, _, _ = extract_data(
        model, 
        readouts, 
        device, 
        readout_input_mean_train, 
        readout_input_std_train, 
        train_loader, 
        num_classes, 
        seq_len, 
        repr_dim,
        train_set=True,
        max_samples=max_samples
    )
    
    # Prepare dictionaries for corrected orientation results
    latent_labels.append({})
    projections.append({})
    performance.append({})
    
    if readout_input_mean_train is None or readout_input_std_train is None:
        norm_train_representations = train_representations
        norm_test_representations = test_representations
    else:
        norm_train_representations = (train_representations - readout_input_mean_train) / (readout_input_std_train + 1e-8)
        norm_test_representations = (test_representations - readout_input_mean_train) / (readout_input_std_train + 1e-8)
    norm_train_representations = norm_train_representations.reshape(-1, repr_dim)
    norm_test_representations = norm_test_representations.reshape(-1, repr_dim)
    
    # Compute corrected orientation projections
    predictions_sin, predictions_cos, predictions_discr, targets_sin, targets_cos, targets_discr, r2_sin, r2_cos, acc_discr, phase_shifts, corrected_readout = compute_corrected_orientation_regressor_projections(
        norm_train_representations, 
        train_gt_orientations_raw, 
        train_sprite_labels_raw,
        norm_test_representations, 
        test_gt_orientations_raw, 
        test_sprite_labels_raw,
        learning_rate=orientation_lr,
        n_epochs=orientation_n_epochs,
        device=device
    )

    # Store results
    projections[4]['sin'] = predictions_sin
    projections[4]['cos'] = predictions_cos
    projections[4]['theta_discr'] = predictions_discr
    latent_labels[4]['sin'] = targets_sin
    latent_labels[4]['cos'] = targets_cos
    latent_labels[4]['theta_discr'] = targets_discr
    performance[4]['sin'] = r2_sin
    performance[4]['cos'] = r2_cos
    performance[4]['theta_discr'] = acc_discr

    # Prepare dictionaries for binary classifier results
    latent_labels.append({})
    projections.append({})
    performance.append({})

    # Compute binary classifier for the last task
    predictions_bin, targets_bin, acc_bin, bin_readout = _train_binary_classifier(
        norm_train_representations,
        train_sprite_labels_raw,
        norm_test_representations,
        test_sprite_labels_raw,
        class_idxes=class_idxes,
        learning_rate=orientation_lr,
        n_epochs=orientation_n_epochs//5,
        device=device
    )
    
    projections[5]['bin'] = predictions_bin
    latent_labels[5]['bin'] = targets_bin
    performance[5]['bin'] = acc_bin
    
    return projections, latent_labels, performance, phase_shifts, corrected_readout


def _train_binary_classifier(
    train_representations,
    train_sprite_labels,
    test_representations,
    test_sprite_labels,
    class_idxes=[0, 1],
    learning_rate=0.01,
    n_epochs=1000,
    device='cpu',
):
    """
    Core binary classifier training function (refactored from original).
    
    Args:
        train_representations: Input features for training (n_samples, n_features)
        train_sprite_labels: Discrete class labels for training (n_samples, 1)
        train_raw_labels: Raw labels for training data (n_samples, n_dims)
        test_representations: Input features for testing (n_samples, n_features)
        test_sprite_labels: Discrete class labels for testing (n_samples, 1)
        test_raw_labels: Raw labels for testing data (n_samples, n_dims)
        class_idxes: List containing the two class indices to classify [class_a, class_b]
        learning_rate: Learning rate for optimization
        n_epochs: Number of training epochs
        device: 'cpu' or 'cuda' for torch computations
        
    Returns:
        Dict containing all results and metrics
    """
    n_samples, n_features = train_representations.shape

    # change all other sprite labels to -1
    train_binary_labels_tensor = torch.where(train_sprite_labels.view(-1) == class_idxes[0], 0,
                                             torch.where(train_sprite_labels.view(-1) == class_idxes[1], 1, -1)).clone().detach().view(-1, 1).float()
    test_binary_labels_tensor = torch.where(test_sprite_labels.view(-1) == class_idxes[0], 0,
                                            torch.where(test_sprite_labels.view(-1) == class_idxes[1], 1, -1)).clone().detach().view(-1, 1).float()
    
    # Prepare tensors for training
    train_representations_tensor = train_representations.clone().detach()

    # exclude -1 labels
    train_representations_tensor = train_representations_tensor[train_binary_labels_tensor.view(-1) != -1]
    train_binary_labels_tensor = train_binary_labels_tensor[train_binary_labels_tensor.view(-1) != -1]
    train_binary_labels_tensor = train_binary_labels_tensor.view(-1, 1).float()
    
    # Initialize trainable parameters
    readout = nn.Linear(n_features, 1, bias=True).to(device)
    
    # Set up optimizer and loss function
    optimizer = optim.AdamW(readout.parameters(), lr=learning_rate, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Training loop with progress bar
    epoch_pbar = tqdm(range(n_epochs), desc=f"Binary Classifier Training (Classes {class_idxes[0]} vs {class_idxes[1]})")
    for epoch in epoch_pbar:
        optimizer.zero_grad()
        
        # Forward pass
        logits = readout(train_representations_tensor)
        
        # Compute loss and backpropagate
        loss = loss_fn(logits, train_binary_labels_tensor)
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (preds == train_binary_labels_tensor).float().mean()
            epoch_pbar.set_postfix(loss=f"{loss.item():.6f}", accuracy=f"{accuracy.item():.4f}")
    
    # Compute final projections for test set
    with torch.no_grad():
        test_logits = readout(test_representations)
        test_predictions = (torch.sigmoid(test_logits) > 0.5).float()
        
        # exclude -1 labels
        test_predictions_ = test_predictions[test_binary_labels_tensor.view(-1) != -1].view(-1, 1)
        test_binary_labels_tensor_ = test_binary_labels_tensor[test_binary_labels_tensor.view(-1) != -1].view(-1, 1)
        test_accuracy = (test_predictions_ == test_binary_labels_tensor_).float().mean()
    
    return (
        test_logits.cpu().numpy(),
        test_binary_labels_tensor.cpu().numpy(),
        test_accuracy.item(),
        readout
    )
