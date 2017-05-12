import time
import pickle
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import train_classifier


def slide_window(img,
                 x_start_stop=[None, None],
                 y_start_stop=[None, None],
                 xy_window=(64, 64),
                 xy_overlap=(0.5, 0.5)
                 ):
    """Get sliding windows in an image"""

    # If x and/or y start/stop positions not defined, set to image size
    x_start_stop = list(x_start_stop)
    y_start_stop = list(y_start_stop)
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)

    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))
    return window_list


def sliding_windows_for_vehicles(img, xy_overlap=(0.5, 0.5), y_min=320):
    # TODO
    window_definitions = [
        # window size, y_start, y_end
        # [200, y_min, None],
        [128, y_min, None],
        # [96, y_min, None],
        # [64, y_min, None]
    ]

    windows = []
    for definition in window_definitions:
        size = definition[0]
        y_start_stop = (definition[1], definition[2])
        windows.extend(slide_window(img, xy_window=(size, size), y_start_stop=y_start_stop, xy_overlap=xy_overlap))

    return windows


def part(image, bbox):
    """Returns a part of an image (defined by a bounding box)"""
    start, end = bbox
    return image[start[1]:end[1], start[0]:end[0]]


def scale_window(window, scale):
    return scale_tuple(window[0], scale), scale_tuple(window[1], scale)


def scale_tuple(point, scale):
    point = scale * point
    return int(point[0]), int(point[1])


def feature_vector(image, window,
                   size=64,
                   conversion=cv2.COLOR_RGB2YUV,
                   hog_orientations=9,
                   hog_pixels_per_cell=8,
                   hog_cells_per_block=2,
                   hog_channel=0
                   ):
    image = cv2.cvtColor(image, conversion)
    height, width = image.shape[:2]
    scale = 64 / 128
    image = cv2.resize(image, (int(scale * width), int(scale * height)))
    window = scale_window(window, scale)
    image = part(image, window)


    # vec, vis = train_classifier.get_hog_features(image[:, :, hog_channel],
    #                                              orientations=hog_orientations,
    #                                              pix_per_cell=hog_pixels_per_cell,
    #                                              cell_per_block=hog_cells_per_block,
    #                                              visualize=True,
    #                                              feature_vector=False)
    # plt.imshow(vis)
    # plt.show()
    # exit()

    return train_classifier.extract_features(
        [image],
        color_space='YUV',
        add_spatial_features=True,
        spatial_size=(32, 32),
        add_histogram_features=True,
        histogram_bins=32,
        add_hog_features=True,
        hog_orientations=9,
        hog_pixels_per_cell=8,
        hog_cells_per_block=2,
        hog_channel='ALL'
    )[0]


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """Draw bounding boxes"""
    result = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(result, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), color, thick)
    return result


def main():
    # load image, get sliding windows
    image = train_classifier.read_rgb_image('bbox-example-image.jpg')
    bboxes = np.array(sliding_windows_for_vehicles(image))

    # get feature vectors for bboxes
    clf, X_scaler = train_classifier.load_classifier()
    feature_vectors = np.array([feature_vector(image, bbox) for bbox in bboxes])
    print(feature_vectors.shape)

    # scale like training data
    scaled_feature_vectors = X_scaler.transform(feature_vectors)

    # now make predictions
    predictions = clf.predict(scaled_feature_vectors)
    bboxes = bboxes[predictions == 1]
    output = draw_boxes(image, bboxes)
    plt.imshow(output)
    plt.show()


main()
