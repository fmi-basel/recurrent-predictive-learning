import torch.nn as nn
import torch


class EncoderNetwork(nn.Module):
    """Base class for encoder networks

    Args:
        backbone (nn.Sequential): Backbone network
        encoded_dim (int): Dimension of the encoded representation. Default: 512

    Attributes:
        backbone (nn.Sequential): Backbone network
        encoded_dim (int): Dimension of the encoded representation

    Notes:
        The `forward` method expects the following inputs:
        - x: Input tensor of shape (B, ..., L)

        The `forward` method outputs the following:
        - z: Encoded representation of shape (B, L, C)
    """

    def __init__(self, backbone=nn.Sequential(), encoded_dim=512):
        super(EncoderNetwork, self).__init__()
        self.backbone = backbone
        self.encoded_dim = encoded_dim

    def get_output_dim(self):
        """Get the output dimension of the encoder

        Returns:
            int: Output dimension of the encoder
        """
        return self.encoded_dim
    
    def get_output_spatial_shape(self, input_shape):
        """Get the output spatial shape of the encoder

        Args:
            input_shape (tuple): Input shape

        Returns:
            int: Output spatial shape of the encoder
        """
        raise NotImplementedError("get_output_spatial_shape method is not implemented for class {}".format(self.__class__.__name__))

    def forward(self, x):
        z = self.backbone(x)  # (B, L, C)
        return z


class IntegratorNetwork(nn.Module):
    """Base class for integrator networks.
    Adapted from https://github.com/facebookresearch/CPC_audio/tree/main

    Args:
        layer_mode (str or nn.Module): Layer mode to use. It can be either one of "RNN", "LSMT", "GRU", "Identity", or a custom layer from backbones.py
        reverse (bool): Whether to reverse the input. Default: False
        keep_hidden (bool): Whether to keep the hidden state. Default: False
        **layer_kwargs: Keyword arguments for the layer

    Attributes:
        backbone (nn.Module): Backbone network
        layer_mode (str or nn.Module): Layer mode to use
        keep_hidden (bool): Whether to keep the hidden state
        hidden (torch.Tensor or tuple): Hidden state
    """

    valid_layer_modes = ["RNN", "LSTM", "GRU", "Identity"]  # valid conventional torch.nn layers

    def __init__(self, layer_mode, reverse=False, keep_hidden=False, **layer_kwargs):
        super(IntegratorNetwork, self).__init__()

        if layer_mode in IntegratorNetwork.valid_layer_modes:
            self.backbone = getattr(nn, layer_mode)(**layer_kwargs)
        elif issubclass(layer_mode, nn.Module):  # custom layer from backbones.py
            self.backbone = layer_mode(**layer_kwargs)
        else:
            raise NotImplementedError(f"Invalid layer mode: {layer_mode}")

        self.layer_mode = layer_mode
        self.reverse = reverse
        self.keep_hidden = keep_hidden
        self.hidden = None

    def get_output_dim(self):
        """Get the output dimension of the integrator."""
        # Used for custom layers to enforce that get_output_dim is implemented
        # and to get the output dimension for Conv layers
        if self.layer_mode not in self.valid_layer_modes:
            return self.backbone.get_output_dim()
        raise NotImplementedError("get_output_dim method is not implemented for class {}".format(self.__class__.__name__))
    
    def get_output_spatial_shape(self, input_shape):
        """Get the output spatial shape of the integrator."""
        # Used for custom layers to enforce that get_output_spatial_shape is implemented
        # and to get the output spatial shape for Conv layers
        if self.layer_mode not in self.valid_layer_modes:
            return self.backbone.get_output_spatial_shape(input_shape)
        raise NotImplementedError("get_output_spatial_shape method is not implemented for class {}".format(self.__class__.__name__))

    def save_hidden(self, hidden):
        """Save the hidden state

        Args:
            hidden (torch.Tensor or tuple): Hidden state
        """
        if self.keep_hidden:
            if isinstance(hidden, tuple):
                self.hidden = tuple(x.detach() for x in hidden)
            else:
                self.hidden = hidden.detach()

    def forward(self, x):
        if self.reverse:
            x = torch.flip(x, dims=[1])
        x, h = self.backbone(x, self.hidden)
        self.save_hidden(h)
        if self.reverse:
            x = torch.flip(x, dims=[1])
        return x
    

class IdentityIntegrator(IntegratorNetwork):
    """Identity integrator network
    
    Args:
        input_size (int): Input size

    Attributes:
        input_size (int): Input size

    Notes:
        The `forward` method expects the following inputs:
        - x: Input tensor of shape (B, L, C)

        The `forward` method outputs the following:
        - x: Output tensor of shape (B, L, C)
    """

    def __init__(self, input_size):
        super(IdentityIntegrator, self).__init__(layer_mode="Identity")
        self.input_size = input_size

    def get_output_dim(self):
        return self.input_size
    
    def forward(self, x):
        return x


class RecurrentIntegrator(IntegratorNetwork):
    """RNN-based autoregressive integrator network

    Args:
        layer_mode (str or nn.Module): Layer mode to use
        reverse (bool): Whether to reverse the input
        keep_hidden (bool): Whether to keep the hidden state
        **layer_kwargs: Keyword arguments for the layer

    Attributes:
        backbone (nn.Module): Backbone network
        layer_mode (str or nn.Module): Layer mode to use
        keep_hidden (bool): Whether to keep the hidden state
        hidden (torch.Tensor or tuple): Hidden state

    Notes:
        The `forward` method expects the following inputs:
        - x: Input tensor of shape (B, L, C)

        The `forward` method outputs the following:
        - x: Output tensor of shape (B, L, C)
    """

    def __init__(
        self,
        layer_mode,
        reverse=False,
        keep_hidden=False,
        input_size=None,
        hidden_size=None,
        num_layers=1,
    ):
        assert layer_mode in ["RNN", "LSTM", "GRU"] or issubclass(
            layer_mode, nn.RNNBase
        )

        super(RecurrentIntegrator, self).__init__(
            layer_mode,
            reverse,
            keep_hidden,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def get_output_dim(self):
        return self.backbone.hidden_size
    
    def get_output_spatial_shape(self, input_shape):
        return 1


class LinearPredictor(nn.Module):
    """Linear predictor network

    Args:
        ctx_dim (int): Output size of the context integrator network
        output_dim (int): Output size of the predictor network

    Attributes:
        layers (nn.ModuleDict): Linear layer

    Notes:
        The `forward` method expects the following inputs:
        - x: Input tensor of shape (B, L, C)

        The `forward` method outputs the following:
        - pred: Output tensor of shape (B, L, pred_dim)
    """

    def __init__(self, ctx_dim, output_dim):
        super(LinearPredictor, self).__init__()
        self.layers = nn.ModuleDict(
            {"linear": nn.Linear(ctx_dim, output_dim)}
        )

    def forward(self, x):
        # (B, L, C) -> (B, L, C*pred_steps)
        return self.layers["linear"](x)


class ReLULinearPredictor(nn.Module):
    """Linear predictor network

    Args:
        ctx_dim (int): Output size of the context integrator network
        output_dim (int): Output size of the predictor network

    Attributes:
        layers (nn.ModuleDict): Linear layer

    Notes:
        The `forward` method expects the following inputs:
        - x: Input tensor of shape (B, L, C)

        The `forward` method outputs the following:
        - pred: Output tensor of shape (B, L, pred_dim)
    """

    def __init__(self, ctx_dim, output_dim):
        super(ReLULinearPredictor, self).__init__()
        self.layers = nn.ModuleDict(
            {
                "relu": nn.ReLU(),
                "linear": nn.Linear(ctx_dim, output_dim)
            }
        )

    def forward(self, x):
        # (B, L, C) -> (B, L, C*pred_steps)
        x = self.layers["relu"](x)
        return self.layers["linear"](x)


class MLPPredictor(nn.Module):
    """MLP predictor network

    Args:
        ctx_dim (int): Output size of the context integrator network
        output_dim (int): Output size of the predictor network
        n_hidden_layers (int): Number of hidden layers in the MLP
        hidden_size (int): Hidden size of the MLP

    Attributes:
        layers (nn.ModuleDict): Linear layer

    Notes:
        The `forward` method expects the following inputs:
        - x: Input tensor of shape (B, L, C)

        The `forward` method outputs the following:
        - pred: Output tensor of shape (B, L, pred_dim)
    """

    def __init__(self, ctx_dim, output_dim, n_hidden_layers, hidden_size):
        super(MLPPredictor, self).__init__()

        self.n_hidden_layers = n_hidden_layers

        self.layers = nn.ModuleDict()
        for i in range(n_hidden_layers):
            if i == 0:
                self.layers[f"linear{i}"] = nn.Linear(ctx_dim, hidden_size)
            else:
                self.layers[f"linear{i}"] = nn.Linear(hidden_size, hidden_size)
            self.layers[f"relu{i}"] = nn.ReLU()
        self.layers[f"linear{n_hidden_layers}"] = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # (B, L, C) -> (B, L, C*pred_steps)
        for i in range(self.n_hidden_layers):
            x = self.layers[f"linear{i}"](x)
            x = self.layers[f"relu{i}"](x)
        out = self.layers[f"linear{self.n_hidden_layers}"](x)
        return out


class LinearReadout(nn.Module):
    """Linear readout network
    
    Args:
        num_classes (int): Number of classes
        input_dim (int): Size of the classifier input
        seq_len (int): Number of prediction steps
        concat_over_time (bool): Whether to concatenate the features over time. Default: False
        task_name (str): Name of the task. Default: None

    Attributes:
        layers (nn.ModuleDict): Linear layer

    Notes:
        The `forward` method expects the following inputs:
        - x: Input tensor of shape (B, L, C)

        The `forward` method outputs the following:
        - pred: Output tensor of shape (B, num_classes)
    """
    
    def __init__(self, num_classes, input_dim, seq_len, concat_over_time=False, concat_over_batch=False, task_name=None):
        super(LinearReadout, self).__init__()
        self.concat_over_time = concat_over_time
        self.concat_over_batch = concat_over_batch
        self.task_name = task_name
        if self.concat_over_time:
            self.layers = nn.ModuleDict(
                {"readout": nn.Linear(input_dim * seq_len, num_classes)}
            )
        else:
            self.layers = nn.ModuleDict(
                {"readout": nn.Linear(input_dim, num_classes)}
            )

    def forward(self, x):
        if self.concat_over_time:
            # (B, L, C) -> (B, L*C)
            x = x.view(x.shape[0], -1)
        elif self.concat_over_batch:
            # (B, L, C) -> (B*L, C)
            x = x.reshape(-1, x.shape[2])
        else:
            # (B, L, C) -> (B, C)
            x = torch.mean(x, dim=1)
        # (B, C) -> (B, num_classes)
        return self.layers["readout"](x)


class FlattenConv(nn.Module):
    """Flatten convolutional features

    Args:
        input_shape (tuple): Shape of the input tensor

    Notes:
        The `forward` method expects the following inputs:
        - x: Input tensor of shape (B, L, C, H, W)

        The `forward` method outputs the following:
        - x: Output tensor of shape (B, C*H*W)
    """

    def __init__(self, input_shape=None):
        super(FlattenConv, self).__init__()
        self.input_shape = input_shape

    def forward(self, x):
        return x.view(x.size(0), x.size(1), -1)
