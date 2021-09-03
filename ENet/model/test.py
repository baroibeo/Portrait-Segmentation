import torch
from Components import ConvBlock,InitialBlock,BottleNeck,Downsampling,Upsampling
from ENet import ENet

def test():
    x = torch.randn(1,3,512,512)
    y = torch.randn(1,128,64,64)
    z = torch.randn(1,64,128,128)
    #test output shape of ConvBlock
    conv_block = ConvBlock(3,64,3,1,1)
    print(conv_block(x).shape)

    #test output shape of Initial block
    initial_block = InitialBlock(3,16)
    print(initial_block(x).shape)

    #test bottlenect  : asymmetric
    assymmetric_block = BottleNeck(128,128,asymmetric=True)
    print(assymmetric_block(y).shape)

    #test bottlenect : diltaed
    dilated_block_x2 = BottleNeck(128,128,dilated=True,dil_rate=2)
    dilated_block_x4 = BottleNeck(128,128,dilated=True,dil_rate=4)
    dilated_block_x8 = BottleNeck(128,128,dilated=True,dil_rate=8)
    dilated_block_x16 = BottleNeck(128,128,dilated=True,dil_rate=16)
    print(dilated_block_x2(y).shape)
    print(dilated_block_x4(y).shape)
    print(dilated_block_x8(y).shape)
    print(dilated_block_x16(y).shape)

    #tet bottlenect : regular
    regular_block = BottleNeck(128,128)
    print(regular_block(y).shape)

    #test downsampling
    downsampling_block = Downsampling(64,128,0.1)
    print(downsampling_block(z)[0].shape)
    down_out = downsampling_block(z)[0]
    max_indices = downsampling_block(z)[1]

    #test upsampling
    upsampling_block = Upsampling(128,64,0.1)
    print(upsampling_block(down_out,max_indices,z.shape[2:]).shape)

    #test enet
    enet = ENet(num_classes=2)
    print(enet(x).shape)


if __name__ == '__main__':
    test()