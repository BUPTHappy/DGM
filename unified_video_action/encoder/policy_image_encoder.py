import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderOutput:
    def __init__(self, features):
        self.features = features
    
    def sample(self):
        return self.features

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
              nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
              nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class PolicyImageEncoder(nn.Module):
    """
    Architecture:
    - input: (B*T,3,256,256)
    - output: (B*T,embed_dim, 16,16)
    - downsampling 
    """
    def __init__(
        self,
        embed_dim = 16,
        image_size = 256,
        stride = 16,
        base_channels = 64,
        num_res_blocks = 2,
        use_batch_norm = True,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.stride = stride
        self.use_batch_norm = use_batch_norm
        
        num_stages = int(torch.log2(torch.tensor(stride, dtype=torch.float32)).item())
        assert 2 ** num_stages == stride, "stride must be a power of 2"

        # Initial convolution
        self.conv_in = nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        if use_batch_norm:
            self.bn_in = nn.BatchNorm2d(base_channels)
        self.relu_in = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Downsampling stages
        # now: 256->64 (stride=4)
        current_channels = base_channels
        self.stages = nn.ModuleList()

        #Stage1: 64->32
        if num_stages >= 3:
            stage1_channel = base_channels * 2
            stage1 = self._make_stage(current_channels, stage1_channel, num_res_blocks, stride=2)
            self.stages.append(stage1)
            current_channels = stage1_channel

        #Stage2: 32->16
        if num_stages >= 4:
            stage2_channel = base_channels * 4
            stage2 = self._make_stage(current_channels, stage2_channel, num_res_blocks, stride=2)
            self.stages.append(stage2)
            current_channels = stage2_channel

        self.conv_out = nn.Conv2d(current_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=True)
        
        self._initialize_weights()
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride=2):
        layers = []
        #First block with downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        #Remaining blocks without downsampling
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        #initial convolution and pooling
        x = self.conv_in(x)
        if self.use_batch_norm:
            x = self.bn_in(x)
        x = self.relu_in(x)
        x = self.maxpool(x)

        #Downsampling stages
        for stage in self.stages:
            x = stage(x)

        x = self.conv_out(x)
        return x
    
    def encode(self, x):
        features = self.forward(x)
        return EncoderOutput(features)