import torch
import pickle
from facenet import *
from logger import *
from predict import *
from utils import *

MODELS_PATH = "./saved_models"
MODEL_NAME = "/12-08_16-14-22_BA977C.pt"
CODES_PATH = "./codes"

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")

mkdir(CODES_PATH)

if __name__ == "__main__":
    gallery = loader_gallery
    test = loader_test
    model = model_class(hash_dim=HASH_DIM)
    model.load_state_dict(torch.load(MODELS_PATH + MODEL_NAME))

    with Logger(write_to_file=False) as logger:
        gallery_codes, gallery_label, test_codes, test_label = \
            predict(model, gallery, test, logger, device=device)

        logger.write("Finished generating codes, writing to output...")
        output = (gallery_codes, gallery_label, test_codes, test_label)
        output_fn = MODEL_NAME.split(".")[0] + ".codes"
        with open(CODES_PATH + output_fn, "wb") as file:
            pickle.dump(output, file)
