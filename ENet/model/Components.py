import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s, p):
        """
        This is a sequential which includes conv,batchnorm,prelu
        :param in_c: in_channels
        :param out_c: out_channels
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.batchnorm(self.conv(x)))


class InitialBlock(nn.Module):
    def __init__(self, in_c=3, out_c=16):
        """
        Initial Block will contain 2 elements:
        1. A conv2d with kernel_size=3 ,stride=2
        2. An extension branch which only has a maxpooling2d
        Because the maxpooling layer follows after the input layer , the out_channels of this layer will be 3
        To make sure that after concatenating the number of channel is out_c ,the output channel number of conv2d will be out_c - in_c
        :param in_c: in_channels
        :param out_c: out_channels
        """
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c - in_c, kernel_size=3, stride=2, padding=1,
                              bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.prelu = nn.PReLU(out_c - in_c)
        self.batchnorm = nn.BatchNorm2d(out_c - in_c)

    def forward(self, x):
        conv_out = self.prelu(self.batchnorm(self.conv(x)))
        pool_out = self.pool(x)
        out = torch.cat([conv_out, pool_out], 1)
        return out


class BottleNeck(nn.Module):
    def __init__(self, in_c, out_c, asymmetric=False, dilated=False, dil_rate=1, p=0.1):
        """
        asymmetric : True -> bottlenect with asymmetric type
        dilated : True -> bottlenect with dilation option will be used
        """
        super(BottleNeck, self).__init__()
        mid_c = in_c // 3  # reduced rate for 1x1 projection : 3
        # first 1x1 block on extra branch
        self.first_1x1 = ConvBlock(in_c, mid_c, 1, 1, 0)

        # last 1x1 block on extra branch
        self.last_1x1 = ConvBlock(mid_c, out_c, 1, 1, 0)

        # conv block on extra branch
        if asymmetric:
            conv = nn.Sequential(
                nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(5, 1), stride=1, padding=(2, 0),
                          bias=False),
                nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(1, 5), stride=1, padding=(0, 2),
                          bias=False)
            )
        elif dilated:
            conv = nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=3, stride=1, padding=dil_rate,
                             dilation=dil_rate, bias=False)
        else:
            conv = nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=3, stride=1, padding=1, bias=False)

        self.mid = nn.Sequential(
            conv,
            nn.BatchNorm2d(mid_c),
            nn.PReLU(mid_c)
        )

        # Dropout
        self.dropout = nn.Dropout2d(p)
        self.out_relu = nn.PReLU(out_c)

    def forward(self, x):
        # extra branch
        ext = self.first_1x1(x)
        ext = self.mid(ext)
        ext = self.last_1x1(ext)
        ext = self.dropout(ext)
        return self.out_relu(ext + x)


class Downsampling(nn.Module):
    def __init__(self, in_c, out_c, p=0.1):
        super(Downsampling, self).__init__()
        mid_c = in_c // 3

        # Main branch contains maxpool + zero_padding
        self.main_branch = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Extension branch : tthe first 1x1 projection is replaced with a 2 x 2 convolution with stride 2 in both dims
        self.first_2x2 = ConvBlock(in_c, mid_c, k=2, s=2, p=0)
        self.conv = ConvBlock(mid_c, mid_c, k=3, s=1, p=1)
        self.last_2x2 = ConvBlock(mid_c, out_c, k=1, s=1, p=0)

        # Dropout
        self.dropout = nn.Dropout2d(p)

        # prelu
        self.out_relu = nn.PReLU(out_c)

    def forward(self, x):
        # extension branch
        ext_out = self.dropout(self.last_2x2(self.conv(self.first_2x2(x))))

        # main branch
        main_out, max_indices = self.main_branch(x)

        # zero padding
        padding = torch.zeros(main_out.shape[0], ext_out.shape[1] - main_out.shape[1], ext_out.shape[2],
                              ext_out.shape[3])
        if main_out.is_cuda:
            padding = padding.cuda()

        # concat
        main_cat = torch.cat((main_out, padding), 1)
        out = main_cat + ext_out
        return self.out_relu(out), max_indices


class Upsampling(nn.Module):
    def __init__(self, in_c, out_c, p=0.1):
        super(Upsampling, self).__init__()
        mid_c = in_c // 3
        self.conv_main = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.unpool = nn.MaxUnpool2d(kernel_size=2)

        # first 1x1
        self.first_1x1 = ConvBlock(in_c, mid_c, k=1, s=1, p=0)

        # conv transpose for upsample
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(mid_c, mid_c, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.PReLU()
        )

        # last 1x1
        self.last_1x1 = ConvBlock(mid_c, out_c, k=1, s=1, p=0)

        # dropout
        self.dropout = nn.Dropout2d(p)

        # prelu
        self.out_relu = nn.PReLU()

    def forward(self, x, max_indices,output_size):
        main_out = self.unpool(self.conv_main(x), indices=max_indices, output_size=output_size)
        ext_out = self.dropout(self.last_1x1(self.upconv(self.first_1x1(x))))
        return self.out_relu(main_out + ext_out)
