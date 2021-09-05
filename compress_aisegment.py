"""
compress ai segment dataset to npy files
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    img_folder = "./Datasets/matting_human_half/clip_img/"
    mask_folder = "./Datasets/matting_human_half/matting/"
    output_npy_folder = "./Datasets/AISegment"
    num_img = 0
    imgs = []
    masks = []
    with open("aisegment_image_names.txt",'w') as f:
        loader = tqdm(os.listdir(img_folder))
        for i,sub_1 in enumerate(loader):
            if i == 0 :
                sub_1_path = os.path.join(img_folder,sub_1)
                mask_sub_1_path = os.path.join(mask_folder,sub_1)
                for sub_2 in os.listdir(sub_1_path):
                    sub_2_path = os.path.join(sub_1_path,sub_2)
                    sub_2_mask = "matting"+"_"+sub_2.split('_')[1]
                    mask_sub_2_path = os.path.join(mask_sub_1_path,sub_2_mask)
                    for img_name in os.listdir(sub_2_path):
                        img = cv2.imread(os.path.join(sub_2_path,img_name),cv2.IMREAD_UNCHANGED)
                        if type(img) is np.ndarray:
                            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img,(224,224))
                            mask_name = img_name.split('.')[0]+".png"
                            mask_path = os.path.join(mask_sub_2_path,mask_name)
                            mask = cv2.imread(mask_path,cv2.IMREAD_UNCHANGED)
                            mask = mask[:,:,3]
                            mask = cv2.resize(mask,(224,224))
                            f.writelines(img_name+"\n")
                            imgs.append(img)
                            num_img+=1
                            masks.append(mask)
                            loader.set_postfix(num_img = num_img)
                        # break
                    # break


        f.close()
        print("Number of sample : ",num_img)

    np.save(os.path.join(output_npy_folder,"imgs/imgs_0.npy"),np.array(imgs,dtype=np.uint8))
    np.save(os.path.join(output_npy_folder,"masks/masks_0.npy"),np.array(masks,dtype=np.uint8))

