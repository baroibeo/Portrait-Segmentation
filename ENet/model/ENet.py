import torch
import torch.nn as nn
from Components import*

class ENet(nn.Module):
    def __init__(self,num_classes):
        super(ENet,self).__init__()
        #Stage 0 : Initial Block
        self.initial_block = InitialBlock(3,16)

        #Stage 1 : downsampling + 4x bottleneck
        self.downsampling_1_0 = Downsampling(16,64,0.01)
        self.bottlenect_1_1 = BottleNeck(64,64,0.01)
        self.bottlenect_1_2 = BottleNeck(64,64,0.01)
        self.bottlenect_1_3 = BottleNeck(64,64,0.01)
        self.bottlenect_1_4 = BottleNeck(64,64,0.01)

        #Stage 2 : downsampling + bottleneck + dilated_2 + asymmetric 5 + dilated_4 + bottleneck + dilated_8 + asymmetric 5  + dilated_16
        self.downsampling_2_0 = Downsampling(64,128,p=0.1)
        self.bottlenect_2_1 = BottleNeck(128,128,p=0.1)
        self.bottlenect_2_2 = BottleNeck(128,128,dilated=True,dil_rate=2,p=0.1)
        self.bottlenect_2_3 = BottleNeck(128,128,asymmetric=True,p=0.1)
        self.bottlenect_2_4 = BottleNeck(128,128,dilated=True,dil_rate=4,p=0.1)
        self.bottlenect_2_5 = BottleNeck(128,128,p=0.1)
        self.bottlenect_2_6 = BottleNeck(128,128,dilated=True,dil_rate=8,p=0.1)
        self.bottlenect_2_7 = BottleNeck(128,128,asymmetric=True,p=0.1)
        self.bottlenect_2_8 = BottleNeck(128,128,dilated=True,dil_rate=16,p=0.1)

        #Stage 3 :Repeat section 2 , without bottleneck 2.0
        self.bottlenect_3_1 = BottleNeck(128, 128,p=0.1)
        self.bottlenect_3_2 = BottleNeck(128, 128, dilated=True, dil_rate=2,p=0.1)
        self.bottlenect_3_3 = BottleNeck(128, 128, asymmetric=True,p=0.1)
        self.bottlenect_3_4 = BottleNeck(128, 128, dilated=True, dil_rate=4,p=0.1)
        self.bottlenect_3_5 = BottleNeck(128, 128,p=0.1)
        self.bottlenect_3_6 = BottleNeck(128, 128, dilated=True, dil_rate=8,p=0.1)
        self.bottlenect_3_7 = BottleNeck(128, 128, asymmetric=True,p=0.1)
        self.bottlenect_3_8 = BottleNeck(128, 128, dilated=True, dil_rate=16,p=0.1)

        #Stage 4 : 1 upsampling + 2 bottlnect
        self.upsample_4_0 = Upsampling(128,64,p=0.1)
        self.bottlenect_4_1 = BottleNeck(64,64,p=0.1)
        self.bottlenect_4_2 = BottleNeck(64,64,p=0.1)

        #Stage 5 : 1 upsampling + 1 bottlenect
        self.upsample_5_0 = Upsampling(64,16,p=0.1)
        self.bottlenect_5_1 = BottleNeck(16,16,p=0.1)

        #Stage 6 : fullconv
        self.fullconv = nn.ConvTranspose2d(16,num_classes,kernel_size=4,stride=2,padding=1,bias=False)

    def forward(self,x):
        #Initial Block
        x = self.initial_block(x)

        #Stage 1 : downsampling + 4x bottleneck
        stage_1_input_size = x.size()
        x,max_indices_1_0 = self.downsampling_1_0(x)
        x = self.bottlenect_1_1(x)
        x = self.bottlenect_1_2(x)
        x = self.bottlenect_1_3(x)
        x = self.bottlenect_1_4(x)

        #Stage 2 : downsampling + bottleneck + dilated_2 + asymmetric 5 + dilated_4 + bottleneck + dilated_8 + asymmetric 5  + dilated_16
        stage_2_input_size = x.size()
        x,max_indices_2_0 = self.downsampling_2_0(x)
        x = self.bottlenect_2_1(x)
        x = self.bottlenect_2_2(x)
        x = self.bottlenect_2_3(x)
        x = self.bottlenect_2_4(x)
        x = self.bottlenect_2_5(x)
        x = self.bottlenect_2_6(x)
        x = self.bottlenect_2_7(x)
        x = self.bottlenect_2_8(x)

        #Stage 3 : Repeat section 2 , without bottleneck 2.0
        x = self.bottlenect_3_1(x)
        x = self.bottlenect_3_2(x)
        x = self.bottlenect_3_3(x)
        x = self.bottlenect_3_4(x)
        x = self.bottlenect_3_5(x)
        x = self.bottlenect_3_6(x)
        x = self.bottlenect_3_7(x)
        x = self.bottlenect_3_8(x)

        #Stage 4 : 1 upsampling + 2 bottleneck
        x = self.upsample_4_0(x,max_indices_2_0,stage_2_input_size)
        x = self.bottlenect_4_1(x)
        x = self.bottlenect_4_2(x)

        #Stage 5 : 1 upsampling + 1 bottleneck
        x = self.upsample_5_0(x,max_indices_1_0,stage_1_input_size)
        x = self.bottlenect_5_1(x)

        #fullconv
        x = self.fullconv(x)
        return x
