import os
import cv2
import uuid
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


from ddh import *

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")

model = model_class(hash_dim=HASH_DIM)
model = model.to(device=device)

optimizer = optim.Adam(model.parameters(), **OPTIM_PARAMS)

run_id = uuid.uuid4().hex.upper()[0:6]
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = now + "+" + run_id
saved_models_path = os.getcwd() + "/saved_models"
mkdir(saved_models_path)
model_name = type(model).__name__
checkpoint_path = saved_models_path + "/{}_{}.pt" \
                    .format(file_name, model_name)


with Logger(write_to_file=True, file_name=file_name) as logger:
    logger.write(
        "Starting run {} for {} epochs with model {}, and following params"
            .format(run_id, NUM_EPOCHS, type(model_class).__name__))
    logger.write("hash_dim: " + str(HASH_DIM))
    logger.write(OPTIM_PARAMS)
    logger.write(CUSTOM_PARAMS)
    logger.write(BATCH_SIZE)
    logger.write(LOADER_PARAMS)
    logger.write("====== START ======")

    start = time()
    logger.write("Loading data...")
    train_set, train_label = set_to_tensor(loader_vocab)
    val_set, val_label = set_to_tensor(loader_val)
    test_set, test_label = set_to_tensor(loader_test)
    logger.write("Finished loading data in {:.0f} seconds"
                    .format(time() - start))

    for epoch in range(NUM_EPOCHS):
        # ======================================================================
        # TRAINING
        # ======================================================================
        quant_loss, score_loss = train(model, loader_train, optimizer, logger,
                                       device=device)

        mean_ap = predict(model, train_set, train_label,
                          val_set, val_label, logger)

        logger.write("Epoch {} - ".format(epoch+1) +
                     "quant loss: {:.8f}, score_loss: {:.8f}, "
                        .format(quant_loss, score_loss) +
                     "MAP on val: {:.8f}".format(mean_ap.mean()))
