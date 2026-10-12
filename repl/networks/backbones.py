from repl.networks.networks import EncoderNetwork
from torchvision.models.resnet import BasicBlock
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Credit: https://github.com/facebookresearch/CPC_audio/tree/main
class CPCAudioEncoder(EncoderNetwork):
    """Encoder network used in CPC for speech dataset with sampling rate of 16kHz (https://arxiv.org/pdf/1807.03748.pdf)
    
    Args:
        hidden_size (int): Hidden size of the encoder. Default: 512
        use_batch_norm (bool): Whether to use batch normalization. Default: False

    Attributes:
        backbone (nn.Sequential): Backbone network
        encoded_dim (int): Dimension of the encoded representation

    Note:
        The encoder network is used as the backbone network for the CPC model.
        The network consists of 5 convolutional layers with batch normalization and ReLU activation.
        The overall architecture is:

        Conv1d(1, 512, kernel_size=10, stride=5, padding=2) -> BatchNorm1d(512) -> ReLU() ->
        Conv1d(512, 512, kernel_size=8, stride=4, padding=2) -> BatchNorm1d(512) -> ReLU() ->
        Conv1d(512, 512, kernel_size=4, stride=2, padding=2) -> BatchNorm1d(512) -> ReLU() ->
        Conv1d(512, 512, kernel_size=4, stride=2, padding=2) -> BatchNorm1d(512) -> ReLU() ->
        Conv1d(512, 512, kernel_size=4, stride=2, padding=1) -> BatchNorm1d(512) -> ReLU()

        The overall downsampling factor is 160.
    """
    def __init__(self, hidden_size=512, use_batch_norm=False):
        super(CPCAudioEncoder, self).__init__(encoded_dim=hidden_size)

        conv_fn = getattr(nn, f"Conv1d")

        filter_sizes = [10, 8, 4, 4, 4]
        strides = [5, 4, 2, 2, 2]
        paddings = [2, 2, 2, 2, 1]

        for i in range(5):
            in_channels = 1 if i == 0 else hidden_size
            self.backbone.add_module(
                f"conv{i}",
                conv_fn(
                    in_channels=in_channels,
                    out_channels=hidden_size,
                    kernel_size=filter_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                ),
            )
            if use_batch_norm:
                self.backbone.add_module(f"batchNorm{i}", nn.BatchNorm1d(hidden_size))
            self.backbone.add_module(f"relu{i}", nn.ReLU())

    def forward(self, x):
        return super().forward(x).transpose(1, 2).contiguous()


class MLPEncoder(nn.Module):
    """MLP Encoder network (implemented as a convolutional network for 1D data)
    
    Args:
        input_dim (int): Dimension of the input. Default: 14
        output_dim (int): Dimension of the encoded representation. Default: 256
        use_batch_norm (bool): Whether to use batch normalization. Default: False
        n_layers (int): Number of hidden layers. Default: 3

    Attributes:
        layers (nn.Sequential): MLP network
    
    Note:
        This is a simple MLP encoder network with three hidden layers.
        We implement the MLP as a convolutional network with stride 1 over the sequence length for efficient computation
        on small sequence lengths. This requires the inputs to have channels as the second dimension, hence why forward 
        expects an input tensor of shape (B, C, L) instead of (B, L, C) as for other datasets.
        The output of the network is still of shape (B, L, output_dim) as usual.
    """

    def __init__(self, input_dim=14, output_dim=256, use_batch_norm=False, n_layers=3):
        super().__init__()

        norm_layer = nn.BatchNorm1d
        self.layers = nn.Sequential()
        for i in range(n_layers):
            in_channels = input_dim if i == 0 else output_dim
            self.layers.add_module(
                f"dense{i}",
                nn.Conv1d(in_channels, output_dim, kernel_size=1, stride=1, padding=0),
            )
            if use_batch_norm:
                self.layers.add_module(f"batchNorm{i}", norm_layer(output_dim))
            self.layers.add_module(f"relu{i}", nn.ReLU())
    
    def forward(self, x):
        # return contiguous tensor
        if len(x.shape) == 3:
            # (B, Cin, L) -> (B, L, Cout)
            x = self.layers(x).transpose(1, 2).contiguous()
        elif len(x.shape) == 5:
            # (B, L, C, H, W) -> (B, L, C*H*W) -> (B, C*H*W, L) -> (B, L, Cout)
            x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)
            x = self.layers(x).transpose(1, 2).contiguous()
        return x.contiguous()


class Conv2dEncoder(EncoderNetwork):
    """Conv2d Encoder network (implemented as a convolutional network for 2D data, where width is the sequence length)
    
    Args:
        n_in_channels (int): Number of input channels. Default: 1
        channel_dim (int): Dimension of the encoded representation. Default: 512
        use_batch_norm (bool): Whether to use batch normalization. Default: False
        n_layers (int): Number of convolutional layers. Default: 1
        kernel_size (list): List of kernel sizes. Default: [(5,5)]
        stride (list): List of strides. Default: [(2,2)]
        padding (list): List of paddings. Default: [(2,2)]
        max_pool_size (list): List of max pool sizes. Default: [(2,2)]
        return_full_feature_map (bool): Whether to return the full feature maps of the last layer (only used for vision). Default: False

    Attributes:
        backbone (nn.Sequential): Convolutional network
        encoded_dim (int): Dimension of the encoded representation

    Note:
        This is a tweakable Conv2d encoder network with one or more hidden layers.
        The network consists of convolutional layers with optional batch normalization and ReLU activation.
        The input is expected to be of shape (B, H, L) for audio where B is the batch size, H is the height, and L is the sequence length,
        and (B, L, C, H, W) for video where C, H and W are number of the channels, the height and width of each frame respectively.
        If `return_full_features` is set to True, the output of the network will be the full feature maps of the last layer, i.e. the output 
        of the network is of shape (B, L, C, H, W). Otherwise, the output is obtained by averaging over height for audio and over width and 
        height for video, resulting in a tensor of shape (B, L, C).
    """
    def __init__(self, n_in_channels=1, channel_dim=512, use_batch_norm=False, n_layers=1, kernel_size=[(5,5)], stride=[(2,2)], padding=[(2,2)], max_pool_size=[(2,2)], return_full_feature_map=False):
        super(Conv2dEncoder, self).__init__(encoded_dim=channel_dim)

        # if kernel_size, stride, padding, and max_pool_size are only given for one layer, repeat them for n_layers
        if len(kernel_size) == 1 and n_layers > 1:
            kernel_size = kernel_size * n_layers
            stride = stride * n_layers
            if padding is not None:
                padding = padding * n_layers
            if max_pool_size is not None:
                max_pool_size = max_pool_size * n_layers

        for i in range(n_layers):
            in_channels = n_in_channels if i == 0 else channel_dim
            self.backbone.add_module(
                f"conv{i}",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=channel_dim,
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                    padding=padding[i] if padding is not None else 0
                )
            )
            if use_batch_norm:
                self.backbone.add_module(f"batchNorm{i}", nn.BatchNorm2d(channel_dim))
            self.backbone.add_module(f"relu{i}", nn.ReLU())
            if max_pool_size is not None:
                self.backbone.add_module(f"pool{i}", nn.MaxPool2d(max_pool_size[i]))

        self.return_full_feature_map = return_full_feature_map

        self.kernel_size = np.array(kernel_size)[:, 0]
        self.stride = np.array(stride)[:, 0]
        self.padding = np.array(padding)[:, 0] if padding is not None else np.zeros(n_layers)
        self.max_pool_size = np.array(max_pool_size)[:, 0] if max_pool_size is not None else np.ones(n_layers)
        self.channel_dim = channel_dim

    def forward(self, x):
        # audio data is (B, H, L)
        if len(x.shape) == 3:
            # (B, H, L) -> (B, 1, H, L) -> (B, C, H, L) -> (B, C, L) 
            z = super().forward(x.unsqueeze(1)).mean(-2)
            # (B, C, L) -> (B, L, C)
            return z.transpose(1, 2).contiguous()
        # visual stimulus is (B, L, C, H, W)
        elif len(x.shape) > 3:
            is_video = len(x.shape) == 5
            if is_video:
                L = x.size(1)
                # (B, L, C, H, W) -> (B*L, C, H, W)
                x = x.view(-1, *x.shape[2:])
            z = super().forward(x)
            if is_video:
                # (B*L, C, H, W) -> (B, L, C, H, W)
                z = z.view(-1, L, z.size(1), z.size(2), z.size(3))

            if self.return_full_feature_map:
                return z
            # (B, L, C, H, W) -> (B, L, C) for video and (B, Nx*Ny, C) for image
            return z.mean(-1).mean(-1)

    def get_output_spatial_shape(self, input_shape):
        shape = input_shape if isinstance(input_shape, int) else input_shape[0]
        for i in range(len(self.kernel_size)):
            shape = (shape - self.kernel_size[i] + 2 * self.padding[i]) // self.stride[i] + 1
            if self.max_pool_size is not None:
                shape = (shape - self.max_pool_size[i]) // self.max_pool_size[i] + 1
        return int(shape)


class CustomResNetEncoder(EncoderNetwork):
    def __init__(self,encoded_dim=256, n_layers=4, n_blocks=[2, 2, 2, 2],
                 n_channels=[32, 64, 128, 256], pool=[False, True, True, True],
                 n_in_channels=3, kernel_size=3, use_batch_norm=True, 
                 return_full_feature_map=False):
        super(CustomResNetEncoder, self).__init__(encoded_dim=encoded_dim)

        if len(n_blocks) == 1 and n_layers > 1:
            n_blocks = n_blocks * n_layers
        if len(n_channels) == 1 and n_layers > 1:
            n_channels = n_channels * n_layers
        if len(pool) == 1 and n_layers > 1:
            pool = pool * n_layers

        assert len(n_blocks) == n_layers, "Length of n_blocks must be equal to n_layers"
        assert len(n_channels) == n_layers, "Length of n_channels must be equal to n_layers"
        assert len(pool) == n_layers, "Length of pool must be equal to n_layers"

        # convert elements in pool to boolean
        pool = [bool(p) for p in pool]

        self.n_layers = n_layers
        self.n_blocks = n_blocks
        self.n_channels = n_channels
        self.pool = pool
        self.n_in_channels = n_in_channels
        self.kernel_size = kernel_size
        self.use_batch_norm = use_batch_norm
        self.full_feature_map = return_full_feature_map
        self.downsample_factor = 2 ** np.sum(pool[:n_layers])

        conv1 = nn.Conv2d(n_in_channels, n_channels[0], kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        bn1 = nn.BatchNorm2d(n_channels[0]) if use_batch_norm else nn.Identity()
        relu = nn.ReLU(inplace=True)
        self.backbone.add_module("conv1", conv1)
        self.backbone.add_module("bn1", bn1)
        self.backbone.add_module("relu", relu)

        n_in_channels = n_channels[0]
        for i in range(n_layers):
            layer = self._make_layer(n_in_channels, n_channels[i], n_blocks[i], use_batch_norm, pool[i])
            self.backbone.add_module(f"layer{i+1}", layer)
            n_in_channels = n_channels[i]

        if not return_full_feature_map:
            self.backbone.add_module("avgpool", nn.AdaptiveAvgPool2d((1, 1)))

    def _make_layer(self, n_in_channels, out_channels, blocks, use_batch_norm, pool):
        in_channels = n_in_channels
        layers = []
        norm_layer = nn.BatchNorm2d if use_batch_norm else None
        for i in range(blocks):
            if i == 0 and pool:
                downsample = nn.Sequential(
                    nn.Conv2d(n_in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                    norm_layer(out_channels),
                )
                stride = 2
            else:
                downsample = nn.Identity()
                stride = 1
            layers.append(
                BasicBlock(in_channels, out_channels, stride=stride,
                           downsample=downsample, norm_layer=norm_layer)
            )
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x is (B, L, C, H, W)
        B, L, C, H, W = x.shape
        # (B*L, C, H, W)
        x = x.view(-1, C, H, W)
        z = super().forward(x)
        return z.view(B, L, *z.shape[1:])
    
    def get_output_spatial_shape(self, input_shape):
        if self.full_feature_map:
            return input_shape // self.downsample_factor
        else:
            return 1
