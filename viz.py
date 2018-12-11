import pickle
import numpy as np
import matplotlib.pyplot as plt
from hamming_dist import *
from facenet import *

CODES_PATH = "./codes"
CODES_FILE = "/12-08_16-14-22_BA977C.codes"
DATASET_PATHS = "./dataset.pickle"

# how many test subjects to pick up and examine
NUM_TEST_TO_SHOW = 6
TOP_N_RESULTS = 10

if __name__ == "__main__":
    with open(CODES_PATH + CODES_FILE, "rb") as file:
        codes = pickle.load(file)

    gallery_codes, gallery_labels, test_codes, test_labels = codes
    gallery_codes = np.array(gallery_codes)
    test_codes = np.array(test_codes)

    gallery_labels = gallery_labels.numpy()
    test_labels = test_labels.numpy()
    truth_table = gallery_labels == test_labels.T

    with open(DATASET_PATHS, "rb") as file:
        gallery, test = pickle.load(file)

    num_gallery, num_test = len(gallery_codes), len(test_codes)

    # only looking at a subset of test subjects
    test_idx = np.random.randint(0, num_test, num_test)[:NUM_TEST_TO_SHOW]
    test_subset = test_codes[test_idx, :]
    # calculate the hamming dists
    hamming_dist = hamming_dist(gallery_codes, test_subset)
    # get the sorted idx
    sorted_idx = hamming_dist.argsort(axis=0)

    fig, ax_arr = plt.subplots(NUM_TEST_TO_SHOW, TOP_N_RESULTS+1,
                               figsize=(25,25))

    for i, tidx in enumerate(test_idx):
        assert test_labels[tidx, 0] == test[tidx][1], "Mismatched test labels!"

        # display the image
        test_img = test[tidx][0]
        ax_arr[i, 0].imshow(np.asarray(test_img))
        ax_arr[i, 0].axis("off")
        ax_arr[i, 0].set_title("Query")

        # display the top N images
        gallery_idx = sorted_idx[:TOP_N_RESULTS, i]

        for j, gidx in enumerate(gallery_idx):
            gallery_img = gallery[gidx][0]
            ax_arr[i, j+1].imshow(np.asarray(gallery_img))
            ax_arr[i, j+1].axis("off")

            if truth_table[gidx, tidx]:
                ax_arr[i, j+1].set_title("MATCH", color="g")
            else:
                ax_arr[i, j+1].set_title("MISMATCH", color="r")

    plt.show()
