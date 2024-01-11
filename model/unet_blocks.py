import torch

class ConvBlock(torch.nn.Module):
    def __init__(self, color_channels, channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(color_channels, channels, kernel_size, 1, 1, bias = False),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(channels, channels, kernel_size, 1, 1, bias = False),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class DownBlock(torch.nn.Module):
    def __init__(self, color_channels, channels, kernel_size, pool_size):
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(color_channels, channels, kernel_size)
        self.pool = torch.nn.MaxPool2d(pool_size)
        self.skip = None

    def forward(self, x):
        conv_output = self.conv(x)
        self.skip = conv_output
        return self.pool(conv_output)

class ResidualBlock(torch.nn.Module):
    def __init__(self, color_channels, channels, kernel_size):
        super(ResidualBlock, self).__init__()
        self.residual = torch.nn.Sequential(
            torch.nn.Conv2d(color_channels, channels, kernel_size, 1, 1),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, kernel_size, 1, 1),
            torch.nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        skip = x
        output = self.residual(x)
        output = torch.add(skip, output)
        output = torch.nn.functional.relu(output)
        return output

class UpBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size, conv_kernel):
        super(UpBlock, self).__init__()
        self.up = torch.nn.ConvTranspose2d(channels, channels // 2, kernel_size, 2)
        self.conv = ConvBlock(channels, channels // 2, conv_kernel)

    def forward(self, x, skip):
        x = self.up(x)
        
        skip = self.crop_wrt_center(x, skip)
        full_input = torch.cat([x, skip], dim = 1)
        return self.conv(full_input)

    # in case of size inbalance through u-net
    def crop_wrt_center(self, x, crop_x):
        #BCHW
        height_diff = crop_x.shape[2] - x.shape[2]
        width_diff = crop_x.shape[3] - x.shape[3]

        # crop and concat
        height_start, height_end = height_diff // 2, crop_x.shape[2] - (height_diff - (height_diff // 2))
        width_start, width_end = width_diff // 2, crop_x.shape[3] - (width_diff - (width_diff // 2))
        return crop_x[:, :, height_start:height_end, width_start:width_end]

class ConvBottom(torch.nn.Module):
    def __init__(self, color_channels, channels, kernel_size):
        super(ConvBottom, self).__init__()
        self.conv = ConvBlock(color_channels, channels, kernel_size) # conv block and start upsampling
        self.skip = None

    def forward(self, x):
        conv_output = self.conv(x)
        self.skip = conv_output
        return conv_output
    