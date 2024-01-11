from .unet_blocks import *

class UNet(torch.nn.Module):
    def __init__(self, channel, img_size, embedding_fn, embedding_dim, num_blocks, num_residue, color_channel = 3, kernel_size = 3, pool_size = 2, upsample_kernel = 2):
        super(UNet, self).__init__()
        assert num_blocks > 0
        self.image_conv = torch.nn.Conv2d(color_channel, channel // 2, 1)
        self.noise_upsampling = torch.nn.Upsample(img_size)
        self.downblocks = torch.nn.ModuleList([DownBlock(channel * 2 ** i, channel * 2 ** (i+1), kernel_size, pool_size) for i in range(num_blocks-1)])
        self.resblocks = torch.nn.ModuleList([ResidualBlock(channel * 2 ** (num_blocks), channel * 2 ** (num_blocks), kernel_size) for i in range(num_residue)])
        self.convbottom = ConvBottom(channel * 2 ** (num_blocks-1), channel * 2 ** (num_blocks), kernel_size)
        self.upblocks = torch.nn.ModuleList([UpBlock(channel * 2 ** (num_blocks - i), upsample_kernel, kernel_size) for i in range(num_blocks-1)])
        self.output = torch.nn.Conv2d(channel * 2, color_channel, kernel_size = 1) # compress to color
        self.embedding_fn = embedding_fn(embedding_dim)

    def forward(self, image, x):
      x = self.embedding_fn(x)
      x = self.noise_upsampling(x)
      image = self.image_conv(image)
      x = torch.cat([x, image], dim = 1)
      
      for block in self.downblocks:
        x = block(x)
      x = self.convbottom(x)
      for block in self.resblocks:
        x = block(x)
      for i, block in enumerate(self.upblocks):
        x = block(x, self.downblocks[-1 - i].skip)
      x = self.output(x)
      
      return x