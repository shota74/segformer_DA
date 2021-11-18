import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import imageio
from . import imutils
import torchvision
import glob

def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    return img_name_list


class cityscapesDataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        name_list_dir=None,
        split='train',
        stage='train',
    ):
        super().__init__()

        self.root_dir = '/workspace/dataset/synscapes/'
        self.stage = stage
        #self.split
        self.img_dir = os.path.join(root_dir, self.stage, 'rgb')
        self.label_dir = os.path.join(root_dir, self.stage, 'class19')
        self.name_list_dir = os.path.join(self.img_dir, '*.png')
        #self.name_list_dir = self.recursive_glob(rootdir=self.img_dir, suffix='.png')
        #print(self.name_list_dir)
        print(self.name_list_dir)
        self.name_list_dir = glob.glob(self.name_list_dir)
        self.name_list = load_img_name_list(self.name_list_dir)


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]
        #print(_img_name.split(os.sep)[-2])
        #_img_name = os.path.basename(_img_name)
        #print(_img_name.split(os.sep)[-2])
        
        img_name = os.path.join(self.img_dir, os.path.basename(_img_name))
        image = np.asarray(imageio.imread(img_name))

        if self.stage == "train":

            label_dir = os.path.join(self.label_dir, os.path.basename(_img_name))
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "val":

            label_dir = os.path.join(self.label_dir, os.path.basename(_img_name))
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "test":
            label = image[:,:,0]

        #return _img_name, image, label

        sample = {'image': image, 'label':label, 'name':_img_name}

        return sample
    
    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, os.path.splitext(filename)[0])
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]




class cityscapesSegDataset(cityscapesDataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = imutils.PhotoMetricDistortion()

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label):
        if self.aug:
            '''
            if self.resize_range: 
                image, label = imutils.random_resize(
                    image, label, size_range=self.resize_range)
            '''
            if self.rescale_range:
                image, label = imutils.random_scaling(
                    image,
                    label,
                    scale_range=self.rescale_range,
                    size_range=self.resize_range)
            if self.img_fliplr:
                image, label = imutils.random_fliplr(image, label)
            image = self.color_jittor(image)
            if self.crop_size:
                image, label = imutils.random_crop(
                    image,
                    label,
                    crop_size=self.crop_size,
                    mean_rgb=[123.675, 116.28, 103.53], 
                    ignore_index=self.ignore_index)
        
        if self.stage != "train":
            image = imutils.img_resize_short(image, min_size=min(self.resize_range))

        image = imutils.normalize_img(image)
        ## to chw
        image = np.transpose(image, (2, 0, 1))

        return image, label

    def __getitem__(self, idx):
        #_img_name, image, label = super().__getitem__(idx)
        sample = super().__getitem__(idx)

        # image, label = self.__transforms(image=image, label=label)

        # return _img_name, image, label
        sample["image"], sample['label'] = self.__transforms(image=sample["image"], label=sample['label'])

        return sample
