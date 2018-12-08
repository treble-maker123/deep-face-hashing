import os
import torch
import numpy as np
from torch.utils.data import Dataset
from utils import DATA_DIR, get_data_path, mkdir, lsdir
from PIL import Image
from pdb import set_trace

class FaceScrubDataset(Dataset):
    '''
    The dataset has a total of 63903 images of 530 faces. The most images a person has is 191, and the least images a person has is 39.

    The combination of datapoints is (63903 choose 2), which is
        (63903 * 63902) / 2 = 2,041,764,753

    The permutation of datapoints is
        63903 ** 2 = 4,083,593,409

    Five images per person set aside for validation,
        7,022,500

    Five images per person set aside for test,
        7,022,500
    '''
    def __init__(self, hash_dim=48,
                 type="comparison", mode="train", transform=None):
        if mode not in ["train", "val", "test"]:
            raise Exception("Invalid dataset mode")
        if type not in ["label", "comparison"]:
            raise Exception("Invalid dataset type")

        self.mode = mode
        self.type = type
        self.names = lsdir(DATA_DIR)
        self.img_paths = self._get_all_img_paths()
        self.transform = transform
        self.hash_dim = hash_dim

    def __len__(self):
        if self.type == "comparison":
            return len(self.img_paths) ** 2
        elif self.type == "label":
            # return len(self.img_paths)

    def __getitem__(self, index):
        if self.type == "comparison":
            return self._get_data_comparison(index)
        elif self.type == "label":
            return self._get_data_label(index)
        else:
            raise Exception("Invalid dataset type")

    def _get_data_comparison(self, index):
        '''
        For __getitem__() method. Return data at index in the format,
            (baseline_image, comparison_image, label)

        Label is an integer specifying whether the baseline and comparison are of the same person. 1 is True and 0 is False.
        '''
        baseline, compare = self._get_pair_from_index(index)
        label = baseline.split("/")[2] == compare.split("/")[2]
        bimg = self._get_img_from_path(baseline)
        cimg = self._get_img_from_path(compare)
        return (bimg, cimg, int(label))

    def _get_data_label(self, index):
        '''
        For __getitem__() method. Return data at index in the format,
            (image, hash_code)

        hash_code is a numpy array of integers of 0s and 1s, mapping the name
        of the peson into Hamming space.
        '''
        img_path = self.img_paths[index]
        name = img_path.split("/")[2]
        return (self._get_img_from_path(img_path), self.names.index(name))

    def _get_pair_from_index(self, index):
        '''
        Return the paths to a pair of images based on the index.
        '''
        num_imgs = len(self.img_paths)
        x, y = index % num_imgs, index // num_imgs
        return self.img_paths[x], self.img_paths[y]

    def _get_folder_paths(self):
        '''
        Return a list of folder paths for all of the people.
        '''
        return list(map(lambda name: DATA_DIR + "/" + name, self.names))

    def _get_all_img_paths(self):
        '''
        Return a list of all image paths.
        '''
        paths = list(map(self._get_img_paths, self.names))
        return sum(paths, [])

    def _get_img_paths(self, name):
        '''
        Returns a list of image paths for the given person.
        '''
        folder = DATA_DIR + "/" + name

        if self.mode == "train":
            pass
        elif self.mode == "val":
            folder += "/val"
        elif self.mode == "test":
            folder += "/test"
        else:
            raise Exception("Invalid dataset mode")

        files = list(filter(lambda f: f not in ["val", "test"], lsdir(folder)))
        return list(map(lambda fp: folder + "/" + fp, files))

    def _get_img_from_path(self, path):
        '''
        Returns an image and applies the transformations defined in self.transform.
        '''
        pil_img = Image.open(path)
        if self.transform is not None:
            img = self.transform(pil_img)
        return img

def create_set(mode, num_imgs=5):
    '''
    This method randomly picks num_imgs images from the DATA_DIR folder and places them in a folder.
    '''
    options = ["val", "test"]
    if mode not in options: return
    # path of all of the people names, "./name"
    name_paths = list(map(lambda name: DATA_DIR + "/" + name, lsdir(DATA_DIR)))
    for path in name_paths:
        # "./name/val"
        test_path = path + "/" + mode
        mkdir(test_path)
        file_names = list(filter(lambda i: i not in options, lsdir(path)))
        num_names = len(file_names)
        idx = list(set(np.random.randint(0, num_names, num_names)))[:num_imgs]
        for i in idx:
            os.rename(path+"/"+file_names[i], test_path+"/"+file_names[i])

def undo_create_set(mode):
    '''
    This method will undo create_test_set().
    '''
    options = ["val", "test"]
    if mode not in options: return
    # path of all of the people names
    name_paths = list(map(lambda name: DATA_DIR + "/" + name,lsdir(DATA_DIR)))
    for path in name_paths:
        test_path = path + "/" + mode
        if not os.path.exists(test_path):
            continue
        test_imgs = lsdir(test_path)
        for i in range(len(test_imgs)):
            os.rename(test_path+"/"+test_imgs[i], path+"/"+test_imgs[i])

def assert_data_split_correct():
    undo_create_set("val")
    undo_create_set("test")
    total_num = len(FaceScrubDataset("train"))
    num_people = len(FaceScrubDataset("train").names)
    assert total_num == 4083593409, "INCORRECT NUMBER OF IMAGES"
    create_set("val")
    create_set("test")
    train = len(FaceScrubDataset("train"))
    val = len(FaceScrubDataset("val"))
    test = len(FaceScrubDataset("test"))
    assert val == (num_people * 5) ** 2
    assert test == (num_people * 5) ** 2


if __name__ == "__main__":
    # dataset = FaceScrubDataset()
    # dataset = FaceScrubDataset(type="label")
    # assert_data_split_correct()
    pass
