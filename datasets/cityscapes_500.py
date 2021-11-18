import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import imageio
from . import imutils
import torchvision

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

        self.root_dir = root_dir
        self.stage = stage
        #self.split
        self.img_dir = os.path.join(root_dir, 'leftImg8bit', self.stage)
        self.label_dir = os.path.join(root_dir, 'gtFine', self.stage)

        if self.stage == 'train':
            with open(os.path.join(os.path.join(root_dir, self.stage + '_500.txt')), "r") as f:
                self.name_list = f.read().splitlines()
                
        else:
            with open(os.path.join(os.path.join(root_dir, self.stage + '.txt')), "r") as f:
                self.name_list = f.read().splitlines()

        # self.name_list_dir = self.recursive_glob(rootdir=self.img_dir, suffix='.png')
        # self.name_list = load_img_name_list(self.name_list_dir)
        #print(self.name_list)
        #self.files[split].sort()
        # self.name_list_dir = os.path.join(name_list_dir, self.split + '.txt')
        # self.name_list = load_img_name_list(self.name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]
        #print(_img_name.split(os.sep)[-2])
        name = os.path.basename(_img_name)
        #print(_img_name.split(os.sep)[-2])
        
        img_name = os.path.join(self.img_dir, _img_name + '_leftImg8bit.png')
        image = np.asarray(imageio.imread(img_name))

        if self.stage == "train":

            label_dir = os.path.join(self.label_dir, _img_name +'_gtFine_labelTrainIds.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "val":

            label_dir = os.path.join(self.label_dir, _img_name +'_gtFine_labelTrainIds.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "test":
            label = image[:,:,0]

        return name, image, label
    
    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, os.path.splitext(filename)[0])
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]







# class VOC12ClsDataset(VOC12Dataset):
#     def __init__(self,
#                  root_dir=None,
#                  name_list_dir=None,
#                  split='train',
#                  stage='train',
#                  resize_range=[512, 640],
#                  rescale_range=[0.5, 2.0],
#                  crop_size=512,
#                  img_fliplr=True,
#                  aug=False,
#                  num_classes=21,
#                  ignore_index=255,
#                  **kwargs):

#         super().__init__(root_dir, name_list_dir, split, stage)

#         self.aug = aug
#         self.ignore_index = ignore_index
#         self.resize_range = resize_range
#         self.rescale_range = rescale_range
#         self.crop_size = crop_size
#         self.img_fliplr = img_fliplr
#         self.num_classes = num_classes

#     def __len__(self):
#         return len(self.name_list)

#     def __transforms(self, image, label):
#         if self.aug:
#             '''
#             if self.resize_range:
#                 image, label = imutils.random_resize(
#                     image, label, size_range=self.resize_range)
#             '''
#             if self.rescale_range:
#                 image, label = imutils.random_scaling(
#                     image,
#                     label,
#                     scale_range=self.rescale_range,
#                     size_range=self.resize_range)
#             if self.img_fliplr:
#                 image, label = imutils.random_fliplr(image, label)
#             if self.crop_size:
#                 image, label = imutils.random_crop(
#                     image,
#                     label,
#                     crop_size=self.crop_size,
#                     mean_rgb=[123.675, 116.28, 103.53])

#         image = imutils.normalize_img(image)
#         ## to chw
#         image = np.transpose(image, (2, 0, 1))

#         return image, label

#     @staticmethod
#     def __to_onehot(label, num_classes):
#         #label_onehot = F.one_hot(label, num_classes)
#         label_onehot = np.zeros(shape=(num_classes), dtype=np.uint8)
#         label_onehot[label] = 1
#         return label_onehot

#     def __getitem__(self, idx):
#         _img_name, image, label = super().__getitem__(idx)

#         image, label = self.__transforms(image=image, label=label)

#         _label = np.unique(label).astype(np.int16)
#         _label = _label[_label != self.ignore_index]
#         #_label = _label[_label != 0]
#         _label = self.__to_onehot(_label, self.num_classes)

#         return _img_name, image, _label


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
