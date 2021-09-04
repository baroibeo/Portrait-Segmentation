import config as cf
from ds1 import dataset_1

def test():
    ds_1_train = dataset_1(img_npy_path=cf.dataset_1_xtrain,mask_npy_path=cf.dataset_1_ytrain,isTrain=True)
    ds_1_train[1]

    ds_1_val = dataset_1(img_npy_path=cf.dataset_1_xtrain,mask_npy_path=cf.dataset_1_ytrain,isTrain=False)
    ds_1_val[300]
if __name__ == '__main__':
    test()