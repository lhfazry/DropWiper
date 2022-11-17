from turtle import forward
from torch import nn

class ConvtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvtBlock, self).__init__()

        self.convt = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, 
                stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.convt(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                stride=stride, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, 64, 1, padding='same', bias=False),

            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding='same', bias=False),

            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 256, 1, padding='same', bias=False),
        )

    def forward(self, x):
        return self.residual(x) + x

class ARDCNN(nn.Module):
    def __init__(self, in_channels):
        super(ARDCNN, self).__init__()

        self.conv_block1 = ConvBlock(in_channels, 32, 3) 
        self.conv_block2 = ConvBlock(32, 64, 3, 1)

        self.conv1x = nn.Conv2d(64, 256, 1, bias=False)
        self.residuals = nn.ModuleList([ResidualBlock(256) for _ in range(6)])

        self.conv_block3 = ConvBlock(256, 64, 1)
        self.conv_block4 = ConvtBlock(64, 32, 3, 1)
        
        self.last_conv = nn.Conv2d(32, 1, 3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv1x(x)

        for module in self.residuals:
            x = module(x)

        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.last_conv(x)

        return self.sigmoid(x)
