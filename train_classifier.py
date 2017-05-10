import matplotlib.image as mpimg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split


def main():
    # get image paths
    vehicles = glob.glob('dataset/vehicles/**/*.png', recursive=True)
    nonvehicles = glob.glob('dataset/non-vehicles/**/*.png', recursive=True)
    print('{} images of vehicles'.format(len(vehicles)))
    print('{} images of non-vehicles'.format(len(nonvehicles)))


main()
