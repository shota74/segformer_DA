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
        synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
        synthia_set_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]

        self.root_dir = '/workspace/dataset/SYNTHIA/RAND_CITYSCAPES/'
        self.stage = stage
        #self.split
        self.img_dir = os.path.join(root_dir, 'RGB')
        self.label_dir = os.path.join(root_dir, 'GT', 'label_22')
        self.name_list_dir = os.path.join(self.img_dir, '*.png')
        #self.name_list_dir = self.recursive_glob(rootdir=self.img_dir, suffix='.png')
        #print(self.name_list_dir)
        #print(self.name_list_dir)
        self.name_list_dir = glob.glob(self.name_list_dir)
        self.name_list = load_img_name_list(self.name_list_dir)
        self.id_to_trainid = {
            1: 10, 2: 2, 3: 0, 4: 1, 5: 4, 6: 8, 7: 5, 8: 13, 9: 7, 10: 11, 11: 18,
            12: 17, 15: 6, 16: 9, 17: 12, 18: 14, 19: 15, 20: 16, 21: 3
        }

        # Only consider 16 shared classes
        self.class_16 = class_16
        self.trainid_to_16id = {id: i for i, id in enumerate(synthia_set_16)}
        self.class_13 = False
        


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

            label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
            if self.class_16:
                label_copy_16 = ignore_label * \
                    np.ones(label.shape, dtype=np.float32)
                for k, v in self.trainid_to_16id.items():
                    label_copy_16[label_copy == k] = v
                label_copy = label_copy_16
            if self.class_13:
                label_copy_13 = ignore_label * \
                    np.ones(label.shape, dtype=np.float32)
                for k, v in self.trainid_to_13id.items():
                    label_copy_13[label_copy == k] = v
                label_copy = label_copy_13

            label = label_copy


        elif self.stage == "val":

            label_dir = os.path.join(self.label_dir, os.path.basename(_img_name))
            label = np.asarray(imageio.imread(label_dir))

            label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
            if self.class_16:
                label_copy_16 = ignore_label * \
                    np.ones(label.shape, dtype=np.float32)
                for k, v in self.trainid_to_16id.items():
                    label_copy_16[label_copy == k] = v
                label_copy = label_copy_16
            if self.class_13:
                label_copy_13 = ignore_label * \
                    np.ones(label.shape, dtype=np.float32)
                for k, v in self.trainid_to_13id.items():
                    label_copy_13[label_copy == k] = v
                label_copy = label_copy_13

            label = label_copy

        elif self.stage == "test":
            label = image[:,:,0]

        return _img_name, image, label
    
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
        _img_name, image, label = super().__getitem__(idx)

        image, label = self.__transforms(image=image, label=label)

        return _img_name, image, label
