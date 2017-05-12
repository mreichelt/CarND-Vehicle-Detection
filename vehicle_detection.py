from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from scipy.ndimage.measurements import label

import train_classifier
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip


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


def scale_windows(windows, scale):
    return [(scale_tuple(window[0], scale), scale_tuple(window[1], scale)) for window in windows]


def scale_tuple(point, scale):
    return int(point[0] * scale), int(point[1] * scale)


def get_feature_vectors(image,
                        y_min=320,
                        conversion=cv2.COLOR_RGB2YUV,

                        add_spatial_features=True,  # True to include spatial features (resized image)
                        spatial_size=(32, 32),

                        add_histogram_features=True,  # True to include color histogram
                        histogram_bins=32,

                        add_hog_features=True,  # True to include HOG (histogram of gradients) features
                        hog_orientations=9,
                        hog_pixels_per_cell=8,
                        hog_cells_per_block=2,
                        hog_channels=[0, 1, 2],
                        xy_overlap=(0.75, 0.75),  # how much to overlap sliding windows
                        train_size=64  # image size used to create the original feature vectors (64x64)
                        ):
    vectors = []
    all_windows = []

    # color conversion only needed once
    image = cv2.cvtColor(image, conversion)

    window_definitions = [
        # window size, y_start, y_end
        # [240, y_min, None],
        [192, y_min, None],
        [128, y_min, None],
        # [96, y_min, None],
        # [64, y_min, None]
    ]

    # run multiple windows of each definition once - allows us to reuse images for multiple windows
    for window_definition in window_definitions:
        window_size, y_start, y_stop = window_definition
        y_start_stop = (y_start, y_stop)

        # define all windows in one size
        windows = slide_window(image, xy_window=(window_size, window_size), y_start_stop=y_start_stop,
                               xy_overlap=xy_overlap)
        all_windows.extend(windows)

        # rescale the image and windows
        scale = train_size / window_size
        height, width = image.shape[:2]
        scaled_image = cv2.resize(image, (int(scale * width), int(scale * height)))
        scaled_windows = scale_windows(windows, scale)

        # precompute whole HOG once for one scale
        if add_hog_features:
            hog_precomputed = np.array([
                                           train_classifier.get_hog_features(scaled_image[:, :, hog_channel],
                                                                             orientations=hog_orientations,
                                                                             pix_per_cell=hog_pixels_per_cell,
                                                                             cell_per_block=hog_cells_per_block,
                                                                             feature_vector=False
                                                                             ) for hog_channel in hog_channels
                                           ])
            hog_shape = (
                hog_pixels_per_cell - 1, hog_pixels_per_cell - 1,
                hog_cells_per_block, hog_cells_per_block,
                hog_orientations)

        # now we have the same size as in training mode, so let's create the feature vector(s)
        # for each window create a single vector
        for scaled_window in scaled_windows:
            window_features = []

            # get only part of image that interests us
            p = part(scaled_image, scaled_window)

            if add_spatial_features:
                window_features.append(train_classifier.bin_spatial(p, size=spatial_size))

            if add_histogram_features:
                window_features.append(train_classifier.color_hist(p, bins=histogram_bins))

            if add_hog_features:
                hog_features = []
                for hog_one_channel in hog_precomputed:
                    # extract HOG features of the scaled window
                    start = scaled_window[0]
                    hog_start_x = int(start[0] / hog_pixels_per_cell)
                    hog_start_y = int(start[1] / hog_pixels_per_cell)
                    hog_end_x = hog_start_x + hog_shape[1]
                    hog_end_y = hog_start_y + hog_shape[0]
                    window_hog = hog_one_channel[hog_start_y:hog_end_y, hog_start_x:hog_end_x]
                    if window_hog.shape != hog_shape:
                        exit('HOG shapes not equal: {} vs {}'.format(window_hog.shape, hog_shape))
                    hog_features.extend(window_hog.ravel())
                window_features.append(hog_features)

            vectors.append(np.concatenate(window_features))

    return np.array(vectors), np.array(all_windows)


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """Draw bounding boxes"""
    result = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(result, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), color, thick)
    return result


def draw_labeled_bboxes(image, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 6)
    return image


class HeatMapHistory:
    def __init__(self,
                 n=5,  # maximum number of heatmaps to store
                 threshold=10  # threshold of how many heat is needed for a vehicle to be detected (summed up on frames)
                 ):
        self.n = n
        self.threshold = threshold
        self.heatmaps = []

    def append(self, heatmap):
        self.heatmaps.append(heatmap)
        # only store n heatmaps at maximum
        self.heatmaps = self.heatmaps[-self.n:]

    def get_thresholded_heatmap(self):
        heatmap = np.sum(self.heatmaps, axis=0)
        heatmap[heatmap < self.threshold] = 0
        return heatmap


def get_heatmap(image, bboxes):
    heatmap = np.zeros_like(image)
    for box in bboxes:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def pipeline(image, history: HeatMapHistory, clf: LinearSVC, scaler: StandardScaler):
    # get those feature vectors, includes sliding window for performance reasons
    feature_vectors, bboxes = get_feature_vectors(image)

    # scale like training data
    feature_vectors = scaler.transform(feature_vectors)

    # predict!
    predictions = clf.predict(feature_vectors)
    bboxes = bboxes[predictions == 1]

    # generate heatmap from bboxes
    heatmap = get_heatmap(image, bboxes)
    history.append(heatmap)

    # get thresholded heatmap from history (includes heatmaps from last frames)
    thresholded_heatmap = history.get_thresholded_heatmap()
    labels = label(thresholded_heatmap)

    # draw output image
    return draw_labeled_bboxes(image, labels)


def main_test_image():
    # load SVM classifier & scaler from file
    clf, X_scaler = train_classifier.load_classifier()

    # load image & classifier
    image = train_classifier.read_rgb_image('bbox-example-image.jpg')

    # get feature vectors and according bboxes
    print('extracting features')
    t = time.time()
    feature_vectors, bboxes = get_feature_vectors(image)
    print(feature_vectors.shape)
    print('took {:.0f} ms'.format((time.time() - t) * 1000))

    # scale like training data
    scaled_feature_vectors = X_scaler.transform(feature_vectors)

    # now make predictions
    predictions = clf.predict(scaled_feature_vectors)
    bboxes = bboxes[predictions == 1]
    output = draw_boxes(image, bboxes)
    plt.imshow(output)
    plt.show()


def main(video_file, duration=None, end=False):
    """Runs pipeline on a video and writes it to temp folder"""
    print('processing video file {}'.format(video_file))
    clip = VideoFileClip(video_file)

    if duration is not None:
        if end:
            clip = clip.subclip(clip.duration - duration)
        else:
            clip = clip.subclip(0, duration)

    # load SVM classifier & scaler from file
    clf, scaler = train_classifier.load_classifier()
    history = HeatMapHistory()
    processed = clip.fl(lambda gf, t: pipeline(gf(t), history, clf, scaler), [])
    processed.write_videofile('output.mp4', audio=False)


# main_test_image()
main('project_video.mp4')
