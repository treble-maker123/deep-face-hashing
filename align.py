import os
import cv2
import dlib
import numpy as np
from pdb import set_trace
from matplotlib import pyplot as plt
from utils import lsdir, mkdir

# From
# https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
PREDICTOR_MODEL_PATH = "./saved_models/shape_predictor_68_face_landmarks.dat"
RIGHT_EYE_IDX = np.arange(36, 42)
LEFT_EYE_IDX = np.arange(42, 48)

def align(img_path, **kwargs):
    '''
    https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
    '''
    desired_left_x = kwargs.get("desired_left_eye_x", 0.25)
    img_width = kwargs.get("img_width", 400)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_MODEL_PATH)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_width, img_width))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    # get all of the 68 points
    shape = _shape_to_np(predictor(gray, rects[0]))
    # center of all eye points
    left_eye_center = shape[LEFT_EYE_IDX, :].mean(axis=0).astype('int')
    right_eye_center = shape[RIGHT_EYE_IDX, :].mean(axis=0).astype('int')
    # difference between x and y
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # rotated angle
    angle = np.degrees(np.arctan2(dy, dx)) - 180
    # where the eyes are along x-axis
    desired_right_x = 1.0 - desired_left_x
    # calculate scale to get to desired size
    dist = np.sqrt((dx ** 2) + (dy ** 2))
    desired_dist = desired_right_x - desired_left_x
    desired_dist *= round(img_width * 0.7)
    scale = desired_dist / dist
    # calculate eye center
    eye_center = ((left_eye_center[0] + right_eye_center[0])) // 2, \
                 ((left_eye_center[1] + right_eye_center[1])) // 2
    # rotational matrix
    M = cv2.getRotationMatrix2D(eye_center, angle, scale)
    # update translation component
    tx = img_width * 0.45
    ty = img_width * desired_left_x + 20
    M[0,2] += (tx - eye_center[0])
    M[1,2] += (ty - eye_center[1])
    output = cv2.warpAffine(img_rgb, M, (img_width, img_width))
    return output[:350, :350, :]

def _shape_to_np(shape, dtype="int"):
    '''
    From
    https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    '''
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def _align_imgs(img_files, from_path, to_path):
    counter = 0
    for img_name in img_files:
        img_path = from_path + "/" +  img_name
        aligned_path = to_path + "/" + img_name
        try:
            aligned_img = align(img_path)
            cv2.imwrite(aligned_path,
                        cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR))
        except Exception as error:
            pass
        finally:
            counter += 1
    return counter

if __name__ == "__main__":
    # from dataset import FaceScrubDataset
    # dataset = FaceScrubDataset()
    # img_path = dataset.img_paths[4000]
    # img = cv2.imread(img_path)
    # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # output = align(img_path)
    # plt.imshow(output)
    # plt.show()

    root_path = "./data"
    target_path = "./aligned_data"
    mkdir(target_path)
    names = lsdir(root_path)
    counter = 0
    for name in names:
        from_path = root_path + "/" + name
        to_path = target_path + "/" + name
        mkdir(to_path)
        # training files
        img_names = list(filter(lambda x: ".jpg" in x, lsdir(from_path)))
        counter += _align_imgs(img_names, from_path, to_path)
        # validation files
        val_from_path = from_path + "/val"
        val_to_path = to_path + "/val"
        mkdir(val_to_path)
        img_names = lsdir(val_from_path)
        counter += _align_imgs(img_names, val_from_path, val_to_path)
        # test files
        test_from_path = from_path + "/test"
        test_to_path = to_path + "/test"
        mkdir(test_to_path)
        img_names = lsdir(test_from_path)
        counter += _align_imgs(img_names, test_from_path, test_to_path)
        print("Aligned {} images.".format(counter))
