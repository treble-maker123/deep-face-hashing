import numpy as np
from hamming_dist import *
from calc_map import *
from calc_pre_rec import *

def eval_perf(gallery_codes, gallery_label, test_codes, test_label, **kwargs):
    top_k = kwargs.get("top_k", 50)
    hamm_radius = kwargs.get("hamm_radius", 2)

    gallery_codes = gallery_codes.cpu().numpy()
    gallery_label = gallery_label.cpu().numpy()
    test_codes = test_codes.cpu().numpy()
    test_label = test_label.cpu().numpy()

     # how many matches between train and test
    label_match = (gallery_label == test_label.T).astype("int8")

    dist = hamming_dist(gallery_codes, test_codes)
    rankings = np.argsort(dist, axis=0)

    # mean average precision
    mean_ap = calc_map(label_match, rankings, top_k=top_k)

    # calculate precision and recall curve
    avg_pre, avg_rec, avg_hmean, pre_curve, rec_curve = \
            calc_pre_rec(dist, label_match, hamm_radius)

    return avg_pre, avg_rec, avg_hmean, pre_curve, rec_curve, mean_ap
