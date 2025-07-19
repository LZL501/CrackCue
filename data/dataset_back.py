import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import random


def readIndex(index_path, shuffle=False):
    img_list = []
    with open(index_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            img_list.append(item)
    file_to_read.close()
    if shuffle is True:
        random.shuffle(img_list)
    return img_list

class dataReadPip(object):

    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, item):

        img = cv2.imread(item[0])
        lab = cv2.imread(item[1])
        lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)

        if self.transforms is not None:

            img, lab = self.transforms(img, lab)
        kernel = np.ones((4, 4), np.uint8)
        lab_dila = cv2.dilate(lab, kernel, iterations = 1)    
        img = _preprocess_img(img)
        lab = _preprocess_lab(lab)
        lab_dila = _preprocess_lab(lab_dila)
        # lab = (lab > 0).float()
        lab_dila = (lab_dila > 0).float()

        # print(lab.shape)
        return img, lab, lab_dila


def _preprocess_img(cvImage):
    '''
    :param cvImage: numpy HWC BGR 0~255
    :return: tensor img CHW BGR  float32 cpu 0~1
    '''

    cvImage = cvImage.transpose(2, 0, 1).astype(np.float32) / 255


    return torch.from_numpy(cvImage)

def _preprocess_lab(cvImage):
    '''
    :param cvImage: numpy 0(background) or 255(crack pixel)
    :return: tensor 0 or 1 float32
    '''
    cvImage = cvImage.astype(np.float32) / 255

    return torch.from_numpy(cvImage)


class loadedDataset(Dataset):
    """
    Create a torch Dataset from data
    """

    def __init__(self, dataset, preprocess=None):
        super(loadedDataset, self).__init__()
        self.dataset = dataset
        if preprocess is None:
            preprocess = lambda x: x
        self.preprocess = preprocess

    def __getitem__(self, index):
        return self.preprocess(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


