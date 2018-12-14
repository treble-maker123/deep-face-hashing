import os
import cv2
import uuid
import pickle
from functools import reduce
from time import time, strftime
from datetime import datetime

import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt

from pdb import set_trace
from logger import *
from eval_perf import *
from predict import *


if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")

# ignore all of the "invalid value encountered in true_divide" errors
np.seterr(divide='ignore', invalid='ignore')

# ==================================
# PARAMETERS
# ==================================
WRITE_TO_FILE = False

# whether to load from an intermediate point.
LOAD_SAVED = True
# required
SAVED_MODEL_PATH = "./saved_models/12-12_15-26-28_FADE57.pt"
RUN_ID = "FADE57"
# optional
SAVED_STATS_PATH = None

# from ddh import *
# from ddh2 import *
# from ddh3 import *
from ddh4 import *

if LOAD_SAVED and SAVED_MODEL_PATH:
    print("Loading existing model...")
    model.load_state_dict(torch.load(SAVED_MODEL_PATH))

    if SAVED_STATS_PATH:
        print("Loading existing stats...")
        with open(SAVED_STATS_PATH, "rb") as file:
            stats = pickle.load(file)

if not SAVED_STATS_PATH:
    print("Creating new stats...")
    stats = {
        "val_mean_aps": [],
        "val_avg_pre": [],
        "val_avg_rec": [],
        "val_avg_hmean": [],
        "highest_hmean": 0.0,

        "test_avg_pre": 0.0,
        "test_avg_rec": 0.0,
        "test_avg_hmean": 0.0,
        "test_mean_ap": 0.0,
        "test_pre_curve": None,
        "test_rec_curve": None,
    }

if RUN_ID:
    run_id = RUN_ID
else:
    run_id = uuid.uuid4().hex.upper()[0:6]

now = datetime.now().strftime("%m-%d_%H-%M-%S")
file_name = now + "_" + run_id
# model checkpoint
saved_models_path = os.getcwd() + "/saved_models"
mkdir(saved_models_path)
checkpoint_path = saved_models_path + "/{}.pt" \
                    .format(file_name)

# stats collection
stats_path = os.getcwd() + "/stats"
mkdir(stats_path)
stats_file_path = stats_path + "/{}.pickle".format(file_name)

with Logger(write_to_file=WRITE_TO_FILE, file_name=file_name) as logger:
    logger.write(
        "Starting run {} for {} epochs with model {}, and following params"
            .format(run_id, NUM_EPOCHS, type(model).__name__))
    logger.write("hash_dim: " + str(HASH_DIM))
    logger.write(OPTIM_PARAMS)
    logger.write(CUSTOM_PARAMS)
    logger.write(BATCH_SIZE)
    logger.write(LOADER_PARAMS)
    logger.write("====== START ======")
    logger.write("")

    for epoch in range(NUM_EPOCHS):
        # ======================================================================
        # TRAINING
        # ======================================================================
        logger.write("Epoch {}/{}".format(epoch+1, NUM_EPOCHS))
        logger.write("--------------")

        start = time()
        train(model, loader_train, optimizer, logger,
                device=device,
                **CUSTOM_PARAMS)
        logger.write("Training completed in {:.0f} seconds."
                        .format(time() - start))
        logger.write("")

        # ======================================================================
        # validation
        # ======================================================================
        start = time()
        # get all of the codes for gallery and test images
        gallery_codes, gallery_label, test_codes, test_label = \
            predict(model, loader_gallery, loader_val, logger, device=device)
        # evaluate the performance
        avg_pre, avg_rec, avg_hmean, _, _, mean_ap = \
            eval_perf(gallery_codes, gallery_label, test_codes, test_label,
                      top_k=TOP_K, hamm_radius=HAMM_RADIUS)
        stats['val_mean_aps'].append(mean_ap)
        stats['val_avg_pre'].append(avg_pre)
        stats['val_avg_rec'].append(avg_rec)
        stats['val_avg_hmean'].append(avg_hmean)

        if avg_hmean > stats["highest_hmean"]:
            logger.write(
                "Higher harmonic mean {:.8f}/{:.8f}, saving!"
                    .format(stats["highest_hmean"], avg_hmean))
            # saves the state of this model
            torch.save(model.state_dict(), checkpoint_path)
            stats["highest_hmean"] = avg_hmean

        logger.write("Validation completed in {:.0f} seconds."
                        .format(time() - start))

        logger.write("val MAP: {:.8f}, ".format(mean_ap) +
                     "avg precision: {:.6f}, ".format(avg_pre) +
                     "avg recall: {:.6f}, ".format(avg_rec) +
                     "avg harmonic mean: {:0.6f}".format(avg_hmean))
        logger.write("")

    # ==========================================================================
    # test
    # ==========================================================================
    best_model = model_class(hash_dim=HASH_DIM)
    best_model.load_state_dict(torch.load(checkpoint_path))

    start = time()

    # get all of the codes for gallery and test images
    gallery_codes, gallery_label, test_codes, test_label= \
        predict(best_model, loader_gallery, loader_test, logger, device=device)
    # evaluate the performance
    stats['test_avg_pre'], stats['test_avg_rec'], stats['test_avg_hmean'], \
    stats['test_pre_curve'], stats['test_rec_curve'], stats['test_mean_ap'] = \
            eval_perf(gallery_codes, gallery_label, test_codes, test_label,
                      top_k=TOP_K, hamm_radius=HAMM_RADIUS)

    logger.write("Test completed in {:0.0f} seconds"
                    .format(time() - start))
    logger.write("test MAP: {:.8f}, ".format(stats['test_mean_ap']) +
                 "avg precision: {:.6f}, ".format(stats['test_avg_pre']) +
                 "avg recall: {:.6f}, ".format(stats['test_avg_rec']) +
                 "avg harmonic mean: {:0.6f}"
                    .format(stats['test_avg_hmean']))

    logger.write("====== END ======")
    logger.write("Completed run for {}".format(run_id))

with open(stats_file_path, 'wb') as file:
    pickle.dump(stats, file)
