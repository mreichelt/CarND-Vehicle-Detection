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


def read_rgb_image(file):
    """Read in RGB image on a (0, 256) color scale"""
    image = mpimg.imread(file)
    if (np.max(image) <= 1):
        return image * 255
    return image


def bin_spatial(img, size=(32, 32)):
    """Compute binned color features"""
    return cv2.resize(img, size).ravel()


def color_hist(img, bins=32, bins_range=(0, 256)):
    """Compute color histogram features"""

    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=bins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=bins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=bins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    histogram_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    # Return the individual histograms, bin_centers and feature vector
    return histogram_features


def convert_color(image, color_space='RGB'):
    """Convert the color space of an image"""
    if color_space == 'RGB':
        return image
    codes = {
        'HSV': cv2.COLOR_RGB2HSV,
        'LUV': cv2.COLOR_RGB2LUV,
        'HLS': cv2.COLOR_RGB2HLS,
        'YUV': cv2.COLOR_RGB2YUV,
        'YCrCb': cv2.COLOR_RGB2YCrCb
    }
    return cv2.cvtColor(image, codes[color_space])


def get_hog_features(image, orientations, pix_per_cell, cell_per_block, visualize=False, feature_vector=True):
    """Get HOG features and visualization"""
    return hog(image,
               orientations=orientations,
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               transform_sqrt=True,
               visualise=visualize,
               feature_vector=feature_vector)


def extract_features(images,
                     color_space='RGB',

                     add_spatial_features=True,
                     spatial_size=(32, 32),

                     add_histogram_features=True,
                     histogram_bins=32,

                     add_hog_features=True,
                     hog_orientations=9,
                     hog_pixels_per_cell=8,
                     hog_cells_per_block=2,
                     hog_channel=0
                     ):
    """Extract features from a list of images"""

    features = []

    for file in images:
        file_features = []
        image = read_rgb_image(file)
        image = convert_color(image, color_space=color_space)

        if add_spatial_features:
            file_features.append(bin_spatial(image, size=spatial_size))

        if add_histogram_features:
            file_features.append(color_hist(image, bins=histogram_bins))

        if add_hog_features:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(image.shape[2]):
                    hog_features.append(get_hog_features(image[:, :, channel],
                                                         hog_orientations,
                                                         hog_pixels_per_cell,
                                                         hog_cells_per_block,
                                                         visualize=False,
                                                         feature_vector=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(image[:, :, hog_channel],
                                                hog_orientations,
                                                hog_pixels_per_cell,
                                                hog_cells_per_block,
                                                visualize=False,
                                                feature_vector=True)
            file_features.append(hog_features)

        features.append(np.concatenate(file_features))

    return np.array(features)


def main(
        sample_size=None,
        color_space='RGB',

        add_spatial_features=True,
        spatial_size=(32, 32),

        add_histogram_features=True,
        histogram_bins=32,

        add_hog_features=True,
        hog_orientations=9,
        hog_pixels_per_cell=8,
        hog_cells_per_block=2,
        hog_channel=0
):
    # get image paths
    vehicles = glob.glob('dataset/vehicles/**/*.png', recursive=True)
    non_vehicles = glob.glob('dataset/non-vehicles/**/*.png', recursive=True)
    print('{} images of vehicles'.format(len(vehicles)))
    print('{} images of non-vehicles'.format(len(non_vehicles)))

    # reduce data set for quick learning
    if sample_size is not None:
        vehicles = vehicles[0:sample_size]
        non_vehicles = non_vehicles[0:sample_size]

    # extract features for vehicles and non vehicles
    print('Extracting features…')
    t = time.clock()
    vehicle_features, non_vehicles_features = [
        extract_features(images,
                         color_space=color_space,

                         add_spatial_features=add_spatial_features,
                         spatial_size=spatial_size,

                         add_histogram_features=add_histogram_features,
                         histogram_bins=histogram_bins,

                         add_hog_features=add_hog_features,
                         hog_orientations=hog_orientations,
                         hog_pixels_per_cell=hog_pixels_per_cell,
                         hog_cells_per_block=hog_cells_per_block,
                         hog_channel=hog_channel
                         ) for images in [vehicles, non_vehicles]  # neat trick to avoid writing method call twice :-)
        ]
    print('took {:.0f} ms'.format((time.clock() - t) * 1000))

    print('vehicle shape:     {}'.format(vehicle_features.shape))
    print('non-vehicle shape: {}'.format(non_vehicles_features.shape))


main(
    sample_size=50
)
