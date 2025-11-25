import torch
import torch.nn as nn


class RePLModel(nn.Module):
    """Temporal joint-embedding model based on CPC (https://arxiv.org/pdf/1807.03748.pdf)

    Args:
        encoder (nn.Module): Encoder network
        integrator (nn.Module): Autoregressive network
        preprocess (nn.Module): Optional preprocessing network. Default: None
        postprocess (nn.Module): Optional postprocessing network. Default: None
        predictor (nn.Module): Optional predictor network. Default: nn.Identity()
        linear_readout (nn.Module): Optional linear readout network. Default: None

    Attributes:
        encoder (nn.Module): Encoder network
        integrator (nn.Module): Autoregressive network
        preprocess (nn.Module): Preprocessing network
        postprocess (nn.Module): Postprocessing network
        predictor (nn.Module): Predictor network
        linear_readout (nn.Module): Linear readout network

    Notes:
        The `forward` method gets a data tensor as input which can be in the form of 
        - (B, C, L) for 1D signal like raw audio
        - (B, L, C, H, W) for video.
        The differences arise because convolutional layers expect the channel dimension to be the second dimension.
        Whereas RNNs expect the channel dimension to be the last dimension.
        
        The `forward` method outputs the following:
        - context_tensor: Encoded representation from the integrator network
        - encoded_data: Encoded representation from the encoder network
        - pred: Predictions from the predictor network
        
        All outputs are in the form of (B, L, C) or (B, L, C, H, W), where:
        - B: Batch size
        - L: length of the sequence
        - C: Number of channels or features
        - H: Height of the image
        - W: Width of the image
        Also note that (B, L, C, H, W) is only valid for visual data.
    """

    def __init__(
        self,
        encoder,
        integrator,
        preprocess=None,
        postprocess=None,
        predictor=nn.Identity(),
    ):
        super(RePLModel, self).__init__()
        self.preprocess = preprocess
        self.encoder = encoder
        self.integrator = integrator
        self.predictor = predictor
        self.postprocess = postprocess

    def forward(self, data):
        preprocess_args = {}
        if self.preprocess is not None:
            data, preprocess_args = self.preprocess(data)

        z = self.encoder(data)  # (B, L, C) or (B, L, C, H, W)

        if self.postprocess is not None:
            z = self.postprocess(z, *preprocess_args)

        context_tensor = self.integrator(z)  # (B, L, C) or (B, L, C, H, W)

        pred = self.predictor(context_tensor)  # (B, L, C*num_pred_steps) or (B, L, C*num_pred_steps, H, W)
        
        return z, context_tensor, pred
    

class HierarchicalRePLModel(nn.Module):
    """Hierarchical RePL model for multi-area processing.
    Args:
        encoder_list (list): List of encoder networks for each area
        integrator_list (list): List of integrator networks for each area
        predictor_list (list): List of predictor networks for each area
        preprocess (nn.Module): Optional preprocessing network. Default: None
        postprocess (nn.Module): Optional postprocessing network. Default: None
        integrator_FB_path (bool): Flag to use feedback path in integrator. Default: False
    Attributes:
        areas (nn.ModuleList): List of areas, each containing encoder, integrator, and predictor networks
        n_areas (int): Number of areas in the model
    Notes:
        The `forward` method processes the input data through each area sequentially.
        Each area consists of an encoder, an integrator, and a predictor.
        The output includes the encoded representations, context tensors, and predictions from each area.

        The `forward` method gets a data tensor as input which can be in the form of
        - (B, C, L) for 1D signal like raw audio
        - (B, L, C, H, W) for video.
        The differences arise because convolutional layers expect the channel dimension to be the second dimension.
        Whereas RNNs expect the channel dimension to be the last dimension.
        This dimensionality standard is enforced for each area in the model.

        The `forward` method outputs the following:
        - z_list: List of encoded representations from each area
        - ctx_list: List of context tensors from each area
        - pred_list: List of predictions from each area
        All outputs are in the form of (B, L, C) or (B, L, C, H, W), where:
        - B: Batch size
        - L: length of the sequence
        - C: Number of channels or features
        - H: Height of the image
        - W: Width of the image
        Also note that (B, L, C, H, W) is only valid for visual data.

        The feedback path is not used in the current implementation but will be added in the future.
    """
    def __init__(
        self,
        encoder_list,
        integrator_list,
        predictor_list,
        preprocess=None,
        postprocess=None,
        integrator_FB_path=False,
    ):
        super(HierarchicalRePLModel, self).__init__()
        self.preprocess = preprocess
        self.postprocess = postprocess

        self.integrator_FB_path = integrator_FB_path  # Not used in the current implementation

        self.areas = nn.ModuleList()
        for encoder, integrator, predictor in zip(encoder_list, integrator_list, predictor_list):
            self.areas.append(
                nn.ModuleDict(
                    {
                        "encoder": encoder,
                        "integrator": integrator,
                        "predictor": predictor,
                    }
                )
            )
        self.n_areas = len(self.areas)

    def forward(self, data):
        # preprocess data if available
        preprocess_args = {}
        if self.preprocess is not None:
            data, preprocess_args = self.preprocess(data)

        z_list = []  # list of z's from each block for later loss computation
        ctx_list = []  # list of ctx's from each block for later loss computation
        pred_list = []  # list of predictions from each block for later loss computation

        inp = data  # input to the first block

        for i, area in enumerate(self.areas):
            # Block i's encoder gets input from detached block i-1's z
            z = area["encoder"](inp)  # (B, L, C) or (B, L, C, H, W)

            # apply postprocessing if available (only for inter-block path (and hence the block loss) and not hierarchical FF path)
            if self.postprocess is not None:
                z_processed = self.postprocess(z, *preprocess_args)
            else:
                z_processed = z

            # Block i's integrator gets input from block i's encoder
            ctx = area["integrator"](z_processed)  # (B, L, C) or (B, L, C, H, W)

            # Block i's predictor gets input from block i's ctx
            pred = area["predictor"](ctx)  # (B, L, C*num_pred_steps) or (B, L, C*num_pred_steps, H, W)

            z_list.append(z_processed)
            ctx_list.append(ctx)
            pred_list.append(pred)

            # Gradient doesn't flow through the instantaneous FF path
            if z.ndim == 5:
                inp = z.detach()
            else:
                inp = z.detach().transpose(-2, -1)  # (B, L, C) -> (B, C, L)

        return z_list, ctx_list, pred_list
