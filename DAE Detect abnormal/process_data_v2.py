import cv2
import numpy as np
import pandas as pd
import os
import imgaug.augmenters as iaa
from tqdm.auto import tqdm

class ProcessData():
    def __init__(self,seed=42):
        self.seed = seed

    def get_img_names(self,path):
        return pd.read_csv(path)['image_id'].unique()

    def get_imgs(self,img_names,load_dir,tail='.png',size=None,gray=True):
        imgs =[]
        for name in tqdm(img_names,desc='load_img'):
            path = os.path.join(load_dir,name+tail)
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE) if gray else cv2.imread(path)
            if size is not None:
                img = cv2.resize(img,size).reshape((*size,1)) if gray else cv2.resize(img,size)
            imgs.append(img)
        return np.array(imgs)

    def get_masks(self,img_names,load_dir,size=None):
        masks =[]
        for name in tqdm(img_names,desc='load_mask'):
            path = os.path.join(load_dir,name+'.npy')
            mask = np.load(path)
            if size is not None:
                mask = np.expand_dims(cv2.resize(mask,size),axis=-1)
            masks.append(mask)
        masks = np.array(masks)
        return masks.shape[1:3], masks

    def save_masks(self,img_names,masks,save_dir):
        os.makedirs(save_dir,exist_ok=True)
        for i in tqdm(range(len(img_names)),desc='save_mask'):
            path = os.path.join(save_dir,img_names[i])
            np.save(path,masks[i])

    def get_data(self,imgs,masks):
        return imgs*masks

    def get_data_noise(self,data,masks=None,ntype=3,my_seed=None):
        seed = self.seed
        if my_seed is not None:
            seed=my_seed
        #Tạo noise dạng bông tuyết
        snowflake_big = iaa.weather.Snowflakes(density=0.07,density_uniformity=0.5,flake_size=0.98,flake_size_uniformity=0.7,speed=[0.0,0.05],angle=[80,115],random_state=seed)
        snowflake_line = iaa.weather.Snowflakes(density=[0.035,0.04],density_uniformity=0.5,flake_size=0.7,flake_size_uniformity=0.7,speed=0.1,angle=[80,115],random_state=seed)
        #P1: chọn aplly 1 hoặc 2 cách trên
        someof = iaa.SomeOf((1,2),[snowflake_big,snowflake_line],random_state=seed)
        
        #P2: Tạo noise dropout hình
        codrop = iaa.CoarseDropout(p=(0.3,0.4),size_percent=0.009,random_state=seed)

        aug_list = []
        if ntype == 3:
            aug_list.append(codrop)
            aug_list.append(someof)
        elif ntype == 2:
            aug_list.append(someof)
        elif ntype == 1:
            aug_list.append(codrop)
        else:
            assert False, "ntype is one of: 1 - drop_out, 2 - snow, 3 - all"
        #chọn 1 trong 2 P1 hoặc P2 
        aug = iaa.OneOf(aug_list,random_state=seed)
        img_aug = []
        for d in tqdm(data,desc='noise_type_{:02d}'.format(ntype)):
            img_aug.append(aug(image=d.astype(np.uint8)))
        img_aug = np.array(img_aug)
        if masks is not None:
          img_aug = img_aug*masks
        return img_aug