import cv2
import numpy as np
from dataset.augmentation import resize,normalize
from model.ENet import ENet
import dataset.config as cf
from copy import deepcopy
import torch
from utils import load_checkpoint
import matplotlib.pyplot as plt

def resize(img):
    img = cv2.resize(img, (cf.IMG_NEW_WIDTH, cf.IMG_NEW_HEIGHT), cv2.INTER_LINEAR)
    return img

def normalize(img):
    img[:,:,0] = (img[:,:,0]/255.0 - cf.MEAN[0])/cf.STD[0]
    img[:,:,1] = (img[:,:,1]/255.0 - cf.MEAN[1])/cf.STD[1]
    img[:,:,2] = (img[:,:,2]/255.0 - cf.MEAN[2])/cf.STD[2]
    return img

def combine(img,mask):
    """
    img:PIL image RGB
    mask : PIL image gray
    combine mask into img
    """
    height,width=(mask).shape
    mask_rgb=np.zeros((height,width,3),dtype=np.uint8)
    rows,cols=np.where(mask==255.0)
    mask_rgb[rows,cols]=(255,0,150)
    mask_rgb[mask==1.0]=(0,255,0)
    return cv2.addWeighted(img,0.8,mask_rgb,0.2,0).astype(np.uint8)

if __name__ == '__main__':
    # test_x = np.load("../Datasets/dataset_1/img_uint8.npy")[600]
    # mask_x = np.load("../Datasets/dataset_1/msk_uint8.npy")[600]
    # plt.imshow(mask_x)
    # plt.show()
    # test_x = test_x.astype(np.float64)
    # print(test_x.shape)
    net = ENet(num_classes=2)
    model_folder = "./model_folder/Training_100epochs_subdataset"
    model_name = "best.pth.tar"
    print(model_name.split('.'))
    optimizer = torch.optim.Adam(net.parameters())
    net,_, _, _, _, _ = load_checkpoint(net, optimizer,model_folder,model_name=str("best.pth.tar"),device="cpu")
    # input = resize(test_x)
    # input = np.transpose(normalize(input), (2, 0, 1))
    # input = torch.from_numpy(input).view(1, input.shape[0], input.shape[1], input.shape[2])
    # pred = net(input.float()).max(1)[1] * 255
    # pred = pred.view(pred.shape[1], pred.shape[2]).detach().numpy().astype(np.uint8)
    # print(pred.shape)
    # plt.imshow(pred)
    # plt.show()

    vid = cv2.VideoCapture(0)
    while(True):
        _,frame = vid.read()
        frame = frame.astype(np.float64)
        frame = resize(frame)
        input = np.transpose(normalize(deepcopy(frame)),(2,0,1))
        input = torch.from_numpy(input).view(1,input.shape[0],input.shape[1],input.shape[2])
        pred = net(input.float()).max(1)[1]*255
        pred = pred.view(pred.shape[1],pred.shape[2]).detach().numpy().astype(np.uint8)
        comb = combine(frame.astype(np.uint8),pred)
        cv2.imshow("mask",comb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

