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
quant_path = stats_path + "/{}_quant.pickle".format(file_name)
score_path = stats_path + "/{}_score.pickle".format(file_name)
map_path = stats_path + "/{}_map.pickle".format(file_name)

quant_losses = []
score_losses = []
mean_aps = []
highest_map = 0.0

with Logger(write_to_file=True, file_name=file_name) as logger:
    logger.write(
        "Starting run {} for {} epochs with model {}, and following params"
            .format(run_id, NUM_EPOCHS, type(model).__name__))
    logger.write("hash_dim: " + str(HASH_DIM))
    logger.write(OPTIM_PARAMS)
    logger.write(CUSTOM_PARAMS)
    logger.write(BATCH_SIZE)
    logger.write(LOADER_PARAMS)
    logger.write("====== START ======")

    for epoch in range(NUM_EPOCHS):
        # ======================================================================
        # TRAINING
        # ======================================================================
        logger.write("Starting epoch {}/{}:".format(epoch+1, NUM_EPOCHS))

        start = time()
        quant_loss, score_loss = train(model, loader_train, optimizer, logger,
                                       device=device)
        quant_losses.append(quant_loss)
        score_losses.append(score_loss)
        logger.write("Training completed in {:.0f} seconds."
                        .format(time() - start))

        start = time()
        mean_ap = predict(model, loader_gallery, loader_val, logger,
                          device=device)
        if mean_ap > highest_map:
            logger.write(
                "Higher mean average precision {:.8f}/{:.8f}, saving!"
                    .format(highest_map, mean_ap))
            # saves the state of this model
            torch.save(model.state_dict(), checkpoint_path)
            highest_map = mean_ap
        mean_aps.append(mean_ap)
        logger.write("Validation completed in {:.0f} seconds."
                        .format(time() - start))

        logger.write("Epoch {}/{} - ".format(epoch+1, NUM_EPOCHS) +
                     "quant loss: {:.8f}, score_loss: {:.8f}, "
                        .format(quant_loss, score_loss) +
                     "MAP on val: {:.8f}".format(mean_ap))


    best_model = model_class(hash_dim=HASH_DIM)
    best_model.load_state_dict(torch.load(checkpoint_path))
    start = time()
    mean_ap = predict(best_model, loader_gallery, loader_test,
                      logger, device=device)

    logger.write("Test completed in {:0.0f} seconds."
                    .format(time() - start))
    logger.write("Test mean average precision: {:.8f}".format(mean_ap))

    logger.write("====== END ======")
    logger.write("Completed run for {}".format(run_id))

with open(quant_path, 'wb') as file:
    pickle.dump(quant_losses, file)
with open(score_path, 'wb') as file:
    pickle.dump(score_losses, file)
with open(map_path, 'wb') as file:
    pickle.dump(mean_aps, file)
