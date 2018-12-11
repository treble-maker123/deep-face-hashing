import pickle
import numpy as np
import matplotlib.pyplot as plt
from hamming_dist import *
from facenet import *

CODES_PATH = "./codes"
CODES_FILE = "/12-08_16-14-22_BA977C.codes"
IMAGE_PATHS = "./image_paths.pickle"

# how many test subjects to pick up and examine
NUM_TEST_TO_SHOW = 5

if __name__ == "__main__":
    with open(CODES_PATH + CODES_FILE, "rb") as file:
        codes = pickle.load(file)

    gallery_codes, gallery_label, test_codes, test_label = codes
    gallery_codes = np.array(gallery_codes)
    gallery_label = np.array(gallery_label)[:, 0]
    test_codes = np.array(test_codes)
    test_label = np.array(test_label)[:, 0]

    with open(IMAGE_PATHS, "rb") as file:
        gallery_paths, test_paths = pickle.load(file)

    num_gallery, num_test = len(gallery_codes), len(test_codes)

    # reload the dataset without transformation
    gallery = FaceScrubDataset(type="label",
                               mode="train",
                               normalize=False,
                               hash_dim=HASH_DIM)
    test = FaceScrubDataset(type="label",
                            mode="test",
                            normalize=False,
                            hash_dim=HASH_DIM)

    # only looking at a subset of test subjects
    test_idx = np.random.randint(0, num_test, num_test)[:NUM_TEST_TO_SHOW]
    test_subset = test_codes[test_idx, :]
    # calculate the hamming dists
    hamming_dist = hamming_dist(gallery_codes, test_subset)
    # get the sorted idx
    sorted_idx = hamming_dist.argsort(axis=0)

    # get the images of the selected test subjects
    test_imgs = []
