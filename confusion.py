import pickle
from hamming_dist import *
from functools import reduce
from calc_pre_rec import calc_pre_rec

CODES_PATH = "./codes"
CODES_FILE = "/12-17_20-25-11_25F03D.codes"

if __name__ == "__main__":
    with open(CODES_PATH + CODES_FILE, "rb") as file:
        codes = pickle.load(file)

    gallery_codes, gallery_labels, test_codes, test_labels = codes
    gallery_codes = np.array(gallery_codes)
    test_codes = np.array(test_codes)

    gallery_labels = gallery_labels.numpy()
    test_labels = test_labels.numpy()
    gt = gallery_labels == test_labels.T

    hamm_dist = hamming_dist(gallery_codes, test_codes)
    total = reduce(lambda x,y: x*y, hamm_dist.shape)
    pt = hamm_dist <= 2
    true_pos = ((pt == gt) * pt).sum()
    true_neg = ((~pt == ~gt) * ~pt).sum()
    false_pos = ((pt == ~gt) * pt).sum()
    false_neg = ((~pt == gt) * ~pt).sum()
    print(true_pos, true_neg, false_pos, false_neg, total)
    print(true_pos + true_neg)
    print(false_pos + false_neg)
    print(true_pos + false_pos)
    print(true_neg + false_neg)

    avg_pre, avg_rec, _, _, _ = calc_pre_rec(hamm_dist, gt, 2)
