import pickle
import numpy as np
import matplotlib.pyplot as plt

CODES_PATH = "./codes"
CODES_FILE = "/12-08_16-14-22_BA977C.codes"

if __name__ == "__main__":
    with open(CODES_PATH + CODES_FILE, "rb") as file:
        codes = pickle.load(file)

    # gallery_codes, gallery_label, test_codes, test_label = codes
