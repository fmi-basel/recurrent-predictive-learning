from copy import deepcopy
import torch
import torch.nn as nn
from repl.networks.backbones import CPCAudioEncoder, Conv2dEncoder, CustomResNetEncoder, MLPEncoder
from repl.networks.networks import (
    EncoderNetwork, 
    IdentityIntegrator,
    LinearPredictor, 
    MLPPredictor, 
    ReLULinearPredictor, 
    RecurrentIntegrator, 
    FlattenConv
)
from repl.networks.repl import HierarchicalRePLModel, RePLModel
from repl.networks._valid_names_lists import ENCODER_NAMES, INTEGRATOR_NAMES, PREDICTOR_NAMES


def additional_data_process(dataset, flatten_enc_output=False):
    """Pre- and post-processing functions for datasets.
    
    Args:
        dataset (str): Dataset name.
        flatten_enc_output (bool): Whether to flatten encoder output.

    Returns:
        tuple: Pre- and post-processing functions.
    """

    preprocess = None  # no preprocessing for the time being

    if dataset in ["animals", "mnist", "oddballs", "mouse", "tlt12"]:
        postprocess = FlattenConv() if flatten_enc_output else None
    else:
        postprocess = None

    return preprocess, postprocess


def setup_encoder(
        kind, 
        input_size, 
        n_in_channels,
        latent_size, 
        use_bn=False, 
        enc_n_layers=1,
        enc_kernel_size=None, 
        enc_stride=None, 
        enc_padding=None, 
        enc_pool_size=None,
        resnet_n_channels=None,
        resnet_n_blocks=None,
        resnet_pool=None,
        resnet_stem_kernel_size=3,
        return_full_features=False,
        is_frozen=False):
    f"""Setup encoder network.

    Args:
        kind (str): Encoder type. One of {ENCODER_NAMES}.
        input_size (int): Input size.
        latent_size (int): Latent size.
        enc_norm_layer (torch.nn.Module): Normalization layer.
        enc_n_layers (int): Number of layers.
        enc_kernel_size (int): Kernel size.
        enc_stride (int): Stride.
        enc_padding (int): Padding.
        enc_pool_size (int): Pool size.
        return_full_features (bool): Whether to return full features (used for video datasets to output B,L,C,H,W instead of B,L,C).
        is_frozen (bool): Whether to freeze the encoder.

    Returns:
        torch.nn.Module: Encoder network.
    """

    if kind == "mlp":
        mlp_network = MLPEncoder(input_dim=input_size, output_dim=latent_size, use_batch_norm=use_bn, n_layers=enc_n_layers)
        encoder = EncoderNetwork(mlp_network, encoded_dim=latent_size)
    elif kind == "cpc":
        encoder = CPCAudioEncoder(hidden_size=latent_size, use_batch_norm=use_bn)
    elif kind == "conv2d":
        encoder = Conv2dEncoder(n_in_channels=n_in_channels,
                                channel_dim=latent_size,
                                use_batch_norm=use_bn,
                                n_layers=enc_n_layers,
                                kernel_size=enc_kernel_size,
                                stride=enc_stride,
                                padding=enc_padding,
                                max_pool_size=enc_pool_size,
                                return_full_feature_map=return_full_features)
    elif kind == "resnet":
        encoder = CustomResNetEncoder(
            n_in_channels=n_in_channels,
            n_channels=resnet_n_channels if resnet_n_channels is not None else [64, 128, 256, 512],
            n_blocks=resnet_n_blocks if resnet_n_blocks is not None else [2, 2, 2, 2],
            pool=resnet_pool if resnet_pool is not None else [False, True, True, True],
            n_layers=enc_n_layers,
            kernel_size=resnet_stem_kernel_size,
            use_batch_norm=use_bn,
            return_full_feature_map=return_full_features,
        )
    else:
        raise NotImplementedError(f"Encoder {kind} not implemented")
    
    if is_frozen:
        for param in encoder.parameters():
            param.requires_grad = False
    return encoder


def setup_integrator(kind, input_size, latent_size, ctx_n_layers=1, is_frozen=False):
    f"""Setup integrator network.

    Args:
        kind (str): Integrator type. One of {INTEGRATOR_NAMES}.
        input_size (int): Input size.
        latent_size (int): Latent size.
        ctx_n_layers (int): Number of layers for context integrator network.
        is_frozen (bool): Whether to freeze the integrator.

    Returns:
        torch.nn.Module: Integrator network.
    """

    if kind in ["rnn", "lstm", "gru"]:
        integrator = RecurrentIntegrator(layer_mode=kind.upper(), input_size=input_size, hidden_size=latent_size, 
                                         num_layers=ctx_n_layers, reverse=False, keep_hidden=False)
    elif kind == "identity":
        integrator = IdentityIntegrator(input_size)
        latent_size = input_size
    else:
        raise NotImplementedError(f"Context integrator {kind} not implemented")
    
    if is_frozen:
        for param in integrator.parameters():
            param.requires_grad = False
    return integrator


def setup_predictor(kind, ctx_size, enc_size, n_layers, hidden_size, pred_steps, dense_prediction, prediction_target, pred_target_dim_override=0, is_frozen=False):
    f"""Setup predictor network.

    Args:
        kind (str): Predictor type. One of {PREDICTOR_NAMES}.
        ctx_size (int): Context integrator size.
        enc_size (int): Encoder size.
        n_layers (int): Number of layers for MLP predictor.
        hidden_size (int): Hidden size for MLP predictor.
        pred_steps (int): Prediction steps.
        dense_prediction (bool): Whether to predict dense output.
        prediction_target (str): Prediction target network. One of ["enc", "ctx"].
        pred_target_dim_override (int): Override dimension for prediction target.
        is_frozen (bool): Whether to freeze the predictor.

    Returns:
        torch.nn.Module: Predictor network.
    """

    output_dim = get_predictor_output_dim(ctx_size, enc_size, pred_steps, dense_prediction, prediction_target, pred_target_dim_override=pred_target_dim_override)
    if kind == "linear":
        predictor = LinearPredictor(ctx_dim=ctx_size, output_dim=output_dim)
    elif kind == "relu-linear":
        predictor = ReLULinearPredictor(ctx_dim=ctx_size, output_dim=output_dim)
    elif kind == "mlp":
        predictor = MLPPredictor(ctx_dim=ctx_size, output_dim=output_dim, n_hidden_layers=n_layers, hidden_size=hidden_size)
    elif kind is None:
        predictor = nn.Identity()
    else:
        raise NotImplementedError(f"Predictor {kind} not implemented")
    
    if is_frozen:
        for param in predictor.parameters():
            param.requires_grad = False
    return predictor


def get_predictor_output_dim(ctx_size, enc_size, pred_steps, dense_prediction, prediction_target, pred_target_dim_override=0, single_readout=False):
    """Get predictor output dimension.

    Args:
        ctx_size (int): Integrator size.
        enc_size (int): Encoder size.
        pred_steps (int): Prediction steps.
        dense_prediction (bool): Whether to predict dense output.
        prediction_target (str): Prediction target network. One of ["enc", "ctx"].
        pred_target_dim_override (int): Override dimension for prediction target.
        single_readout (bool): Whether to multiply output dimension by prediction steps.
        
    Returns:
        int: Output dimension.
    """

    if prediction_target == "pred":
        if pred_target_dim_override > 0:
            return pred_target_dim_override
        else:
            output_dim = enc_size
    elif prediction_target == "enc":
        output_dim = enc_size
    elif prediction_target == "ctx":
        output_dim = ctx_size
    else:
        raise ValueError(f"Invalid prediction target: {prediction_target}")
    output_dim *= pred_steps if dense_prediction and not single_readout else 1
    return output_dim

def prepare_model(
        encoder_kind,
        integrator_kind,
        predictor_kind,
        input_size, 
        enc_dim,
        ctx_dim,
        pred_n_hidden_layers,
        pred_hidden_dim,
        pred_steps,
        ctx_n_layers=1,
        dense_prediction=False,
        prediction_target="enc", 
        pred_target_dim_override=0,
        use_bn_enc=False,
        enc_n_layers=1,
        enc_kernel_size=None,
        enc_stride=None,
        enc_padding=None,
        enc_pool_size=None,
        resnet_n_channels=None,
        resnet_n_blocks=None,
        resnet_pool=None,
        resnet_stem_kernel_size=3,
        preprocess=None,
        postprocess=None,
        state_dict=None,
        return_full_features=False,
        flatten_enc_output=False,
        n_in_channels=None,
        n_areas=None,
        frozen_areas=None):
    f"""Prepare RePL model.
    
    Args:
        encoder_kind (str): Encoder type. One of {ENCODER_NAMES}.
        integrator_kind (str): Context integrator type. One of {INTEGRATOR_NAMES}.
        predictor_kind (str): Predictor type. One of {PREDICTOR_NAMES}.
        input_size (int): Input size.
        enc_dim (int): Encoder size.
        ctx_dim (int): Context integrator size.
        pred_n_hidden_layers (int): Number of layers for MLP predictor.
        pred_hidden_dim (int): Hidden size for MLP predictor.
        pred_steps (int): Prediction steps.
        ctx_n_layers (int): Number of layers for context integrator network.
        dense_prediction (bool): Whether to predict dense output.
        prediction_target (str): Prediction target network. One of ["enc", "ctx"].
        pred_target_dim_override (int): Override dimension for prediction target (only for inv loss).
        use_bn_enc (bool): Whether to use batch normalization for the encoder network.
        enc_n_layers (int): Number of layers for Conv2dEncoder.
        enc_kernel_size (int): Kernel size for Conv2dEncoder.
        enc_stride (int): Stride for Conv2dEncoder.
        enc_padding (int): Padding for Conv2dEncoder.
        enc_pool_size (int): Pool size for Conv2dEncoder.
        preprocess (torch.nn.Module): Pre-processing layer.
        postprocess (torch.nn.Module): Post-processing layer.
        state_dict (dict): Model state dictionary.
        return_full_features (bool): Whether to return full features (used for video datasets to output B,L,C,H,W instead of B,L,C).
        flatten_enc_output (bool): Whether to flatten encoder output.
        n_in_channels (int): Number of input channels.
        n_areas (int): Number of areas in the hierarchical model.
        frozen_areas (list): List of boolean values indicating whether the areas are frozen.

    Returns:
        RePLModel: RePL model.
    """
    if n_areas is None:  # normal RePL model with deep encoder
        encoder = setup_encoder(encoder_kind, input_size, n_in_channels, enc_dim, use_bn_enc, enc_n_layers,
                                enc_kernel_size, enc_stride, enc_padding, enc_pool_size,
                                resnet_n_channels=resnet_n_channels, resnet_n_blocks=resnet_n_blocks,
                                resnet_pool=resnet_pool, resnet_stem_kernel_size=resnet_stem_kernel_size,
                                return_full_features=return_full_features)
        if flatten_enc_output:
            enc_dim *= encoder.get_output_spatial_shape(input_size)**2

        integrator = setup_integrator(integrator_kind, enc_dim, ctx_dim, ctx_n_layers)
        if integrator_kind == "identity":
            ctx_dim = enc_dim
            print("Warning: Identity integrator is used. Context dim is overwritten to encoder dim - {}".format(ctx_dim))

        predictor = setup_predictor(predictor_kind, ctx_dim, enc_dim, pred_n_hidden_layers, pred_hidden_dim, 
                                    pred_steps, dense_prediction, prediction_target, pred_target_dim_override=pred_target_dim_override)
        
        model = RePLModel(encoder=encoder, integrator=integrator, predictor=predictor, preprocess=preprocess, postprocess=postprocess)
        
    else:  # hierarchical RePL model
        assert len(frozen_areas) == n_areas, "A boolean list of frozen areas with length equal to the number of areas in the hierarchy should be provided."
        if len(enc_dim) > 1:
            assert len(enc_dim) == n_areas, "Number of encoder dimensions should match the number of areas."
        else:
            enc_dim = enc_dim * n_areas
        if len(ctx_dim) > 1:
            assert len(ctx_dim) == n_areas, "Number of context dimensions should match the number of areas."
        else:
            ctx_dim = ctx_dim * n_areas
        if pred_hidden_dim is not None:
            if len(pred_hidden_dim) > 1:
                assert len(pred_hidden_dim) == n_areas, "Number of hidden dimensions should match the number of areas."
            else:
                pred_hidden_dim = pred_hidden_dim * n_areas

        encoder_list = nn.ModuleList()
        integrator_list = nn.ModuleList()
        predictor_list = nn.ModuleList()

        enc_spatial_shape = input_size

        for i in range(n_areas):
            input_size = enc_spatial_shape
            if i > 0:
                n_in_channels = enc_dim[i-1]
            kernel_size = [enc_kernel_size[i]] if enc_kernel_size is not None else None
            stride = [enc_stride[i]] if enc_stride is not None else None
            padding = [enc_padding[i]] if enc_padding is not None else None
            pool_size = [enc_pool_size[i]] if enc_pool_size is not None else None
            encoder_list.append(deepcopy(setup_encoder(encoder_kind, input_size, n_in_channels, enc_dim[i], use_bn_enc, enc_n_layers,
                                              kernel_size, stride, padding, pool_size, return_full_features=return_full_features, 
                                              is_frozen=frozen_areas[i])))
            if flatten_enc_output:
                enc_spatial_shape = encoder_list[i].get_output_spatial_shape(enc_spatial_shape)
                integrator_in_dim = enc_dim[i] * enc_spatial_shape **2
            else:
                integrator_in_dim = enc_dim[i]

            integrator_list.append(deepcopy(setup_integrator(integrator_kind, integrator_in_dim, ctx_dim[i], ctx_n_layers, is_frozen=frozen_areas[i])))
            predictor_list.append(deepcopy(setup_predictor(predictor_kind, ctx_dim[i], integrator_in_dim, pred_n_hidden_layers, pred_hidden_dim[i], 
                                                  pred_steps, dense_prediction, prediction_target, is_frozen=frozen_areas[i])))
        model = HierarchicalRePLModel(encoder_list=encoder_list, integrator_list=integrator_list, predictor_list=predictor_list,
                                      preprocess=preprocess, postprocess=postprocess)

    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model