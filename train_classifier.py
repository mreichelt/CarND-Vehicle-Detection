import glob
import time
import pickle
import cv2
import os.path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def read_rgb_image(file):
    """Read RGB image on (0, 256) color scale and uint8 type"""
    image = mpimg.imread(file)
    if np.max(image) <= 1:
        image *= 255
    return np.uint8(image)


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


def pickle_save_big(data, file):
    n_bytes = 2 ** 31
    max_bytes = n_bytes - 1
    bytes_out = pickle.dumps(data)
    with open(file, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def pickle_load_big(file):
    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file)
    with open(file, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)


def save_features(X, y, X_scaled, X_scaler: StandardScaler, file='extracted_features.p'):
    """Save X, y, X_scaled and X_scaler to file"""
    p = {
        'X': X,
        'y': y,
        'X_scaled': X_scaled,
        'X_scaler': X_scaler
    }
    print('saving features to file')
    pickle_save_big(p, file)


def load_features(file='extracted_features.p'):
    """Load extracted features from file, returns [X, y, X_scaled, X_scaler]"""
    print('loading features from file')
    p = pickle_load_big(file)
    return p['X'], p['y'], p['X_scaled'], p['X_scaler']


def save_classifier(clf: LinearSVC, X_scaler: StandardScaler, file='classifier.p'):
    """Save classifier and X_scaler to file"""
    p = {
        'clf': clf,
        'X_scaler': X_scaler
    }
    print('saving features to file')
    pickle.dump(p, open(file, mode='wb'))


def main(
        load_extracted_features=False,  # load features from file (faster development of classifier)
        save_extracted_features=True,  # save extracted features to file
        sample_size=None,  # max number of samples to use
        random_state=1,  # for splitting dataset into train/test sets, set to fixed number for non-random
        test_size=0.2,  # fraction of test set
        color_space='YUV',  # color space to use
        linearsvc_c=0.001,  # C parameter of LinearSVC

        add_spatial_features=True,  # True to include spatial features (resized image)
        spatial_size=(32, 32),

        add_histogram_features=True,  # True to include color histogram
        histogram_bins=32,

        add_hog_features=True,  # True to include HOG (histogram of gradients) features
        hog_orientations=9,
        hog_pixels_per_cell=8,
        hog_cells_per_block=2,
        hog_channel='ALL'
):
    if not load_extracted_features:
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
        t = time.time()
        vehicles_features, non_vehicles_features = [
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
                             ) for images in [vehicles, non_vehicles]
            # neat trick to avoid writing method call twice :-)
            ]
        print('took {:.0f} ms'.format((time.time() - t) * 1000))

        print('vehicle shape:     {}'.format(vehicles_features.shape))
        print('non-vehicle shape: {}'.format(non_vehicles_features.shape))

        # normalizing / scaling features
        X = np.vstack((vehicles_features, non_vehicles_features)).astype(np.float64)
        X_scaler = StandardScaler().fit(X)
        X_scaled = X_scaler.transform(X)

        # define the labels
        y = np.hstack((np.ones(len(vehicles_features)), np.zeros(len(non_vehicles_features))))

        if save_extracted_features:
            save_features(X, y, X_scaled, X_scaler)

    else:
        X, y, X_scaled, X_scaler = load_features()

    # split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    # train a classifier
    print('Training an SVM…')
    t = time.time()
    clf = LinearSVC(C=linearsvc_c)
    clf.fit(X_train, y_train)

    print('\nTraining SVC took {:.2f} seconds'.format(time.time() - t))
    print('Test Accuracy of SVC = {:.4f}'.format(clf.score(X_test, y_test)))

    save_classifier(clf, X_scaler)


main()
