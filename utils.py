import os
import cv2
import numpy as np
from time import time
from multiprocessing import Pool, cpu_count
from functools import reduce
from pdb import set_trace as st

# ==============================================================================
# NOTE: Run all of the code from the ./code directory!
# ==============================================================================

DATA_DIR = "./data"
FACESCRUB_DIR = "../facescrub"
RUN_ASSERTS = True
VERBOSE = False

def mkdir(path):
    '''
    Creates the specified directory if it doesn't exist.
    '''
    if not os.path.exists(path):
        os.mkdir(path)

def lsdir(path):
    '''
    Lists all of the files in the specified path, excluding files that start
    with a "."
    '''
    files = os.listdir(path)
    return list(filter(lambda name: name[0] != ".", files))

def preprocess():
    '''
    Preprocess the data in ../facescrub/download and store them in ./data
    folder.
    '''
    start = time()
    mkdir(DATA_DIR)
    names = lsdir(FACESCRUB_DIR + "/download")
    with Pool(max(1, cpu_count())) as pool:
        pool.map(_process_faces, names)
    print("Preprocessed images in {:.2f} seconds.".format(time() - start))

def _count_before_imgs():
    '''
    Count the number of images in the ./facescrub/download/*/face folders.
    '''
    names = lsdir(FACESCRUB_DIR + "/download")
    paths = list(map(lambda name: get_facescrub_path(name), names))
    return sum(list(map(lambda path: len(lsdir(path)), paths)))

def _count_after_imgs():
    '''
    Count the number of images in the ./code/data/* folders
    '''
    names = lsdir(DATA_DIR)
    paths = list(map(lambda name: get_data_path(name), names))
    return sum(list(map(lambda path: len(lsdir(path)), paths)))

def _process_faces(name):
    '''
    Process the person's face images and save them in the ./code/data directory.
    '''
    if VERBOSE:
        print_name = name.replace("_", " ")
        print("Processing images for {}...".format(print_name))
        start = time()

    # using cropped faces
    faces_dir = get_facescrub_path(name)
    # create directory for the person in the ./code/data folder
    output_dir = get_data_path(name)
    mkdir(output_dir)
    # list of names of images
    img_names = lsdir(faces_dir)
    for img_name in img_names:
        output_path = output_dir + "/" + img_name

        if os.path.isfile(output_path):
            if VERBOSE:
                print("File {} already exists, skipping...".format(img_name))
            continue

        img = cv2.imread(faces_dir + "/" + img_name)

        if img is None:
            if VERBOSE:
                print("Invalid image, skipping...")
            continue

        # eliminate empty images (white for some reason), threshold set at 85%
        max_threshold = round(reduce(lambda x,y: x*y, img.shape) * 255 * 0.85)
        if img.sum() > max_threshold:
            if VERBOSE:
                print("Image above pixel value threshold, skipping...")
            continue

        # save it
        cv2.imwrite(output_path, img)

    output_files = lsdir(output_dir)
    if VERBOSE:
        print("Processed images for {} in {:.2f} seconds. {} images before, {} images after".format(print_name, time() - start,
                         len(img_names), len(output_files)))

def get_facescrub_path(name):
    return FACESCRUB_DIR + "/download/{}/face".format(name)

def get_data_path(name):
    return DATA_DIR + "/{}".format(name)

if __name__ == "__main__":
    preprocess()
    print("There are {} images in ./facescrub/download/*/face folder."
            .format(_count_before_imgs()))
    print("There are {} images in ./code/data/*."
            .format(_count_after_imgs()))
