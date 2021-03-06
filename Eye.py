import cv2

import numpy as np
from matplotlib import pyplot as plt
from skimage import color


class Eye:
    """This class contains all data about one eye: raw picture, correct result, and mask."""

    __raw = None
    __rawGrey = None
    __manual = None
    __mask = None
    __calculated = None
    x_patch_size = 0
    x = int()
    y = int()

    def __init__(self, raw, manual, mask, x_path_size, y_path_size, resize=None):
        if resize:
            self.__raw = cv2.resize(raw, dsize=(int(raw.shape[1]*resize), int(raw.shape[0]*resize)), interpolation=cv2.INTER_CUBIC)
            self.__manual = cv2.resize(manual, dsize=(int(manual.shape[1]*resize), int(manual.shape[0]*resize)), interpolation=cv2.INTER_CUBIC)
            self.__mask = cv2.resize(mask, dsize=(int(mask.shape[1]*resize), int(mask.shape[0]*resize)), interpolation=cv2.INTER_CUBIC)
        else:
            self.__raw = raw
            self.__manual = manual
            self.__mask = mask

        self.__raw_batches = None
        self.__manual_batches = None
        self.__calculated = np.zeros(self.__manual.shape)
        self.__calculated_batches = None
        self.__rawGrey = color.rgb2gray(self.__raw)
        if x_path_size > self.get_raw().shape[0] or x_path_size > self.get_raw().shape[1]:
            self.x_patch_size = min(self.get_raw().shape[0], self.get_raw().shape[1])
            self.y_patch_size = min(self.get_raw().shape[0], self.get_raw().shape[1])
            print("Warning: Size of patch bigger than image. Current patch size: " + str(self.x_patch_size))
        else:
            self.x_patch_size = x_path_size
            self.y_patch_size = y_path_size
        self.side_shift = (self.x_patch_size - self.y_patch_size) // 2

    def get_raw(self):
        return self.__raw

    def get_manual(self):
        return self.__manual

    def get_mask(self):
        return self.__mask

    def get_calculated(self):
        return self.__calculated

    def _generate_batches(self, picture, patch_size, side_shift=0):
        batches = []
        for y in range(0, picture.shape[0] - self.x_patch_size + 1, self.x_patch_size):
            for x in range(0, picture.shape[1] - self.x_patch_size + 1, self.x_patch_size):
                sub_picture = picture[y + side_shift
                                        : y + side_shift + patch_size,
                                        x + side_shift
                                        : x + side_shift + patch_size]
                if len(sub_picture.shape) == 2:
                    sub_picture = sub_picture.reshape((sub_picture.shape[0], sub_picture.shape[1], 1))
                batches.append(sub_picture)
        return batches

    def get_batches_of_raw(self):
        if not self.__raw_batches:
            self.generate_batches_of_raw()
        return self.__raw_batches

    def get_batches_of_manual(self):
        if not self.__manual_batches:
            self.generate_batches_of_manual()
        return self.__manual_batches

    def get_batches_of_calculated(self):
        if not self.__calculated_batches:
            self.generate_batches_of_calculated()
        return self.__calculated_batches

    def generate_batches_of_raw(self):
        self.__raw_batches = self._generate_batches(self.__raw, self.x_patch_size)

    def generate_batches_of_manual(self):
        self.__manual_batches = self._generate_batches(self.__manual, self.y_patch_size, self.side_shift)

    def generate_batches_of_calculated(self):
        self.__calculated_batches = self._generate_batches(self.__calculated, self.y_patch_size, self.side_shift)

    def build_image_from_batches(self, batches):
        # Todo: Fix according to smaller batch size
        picture = self.get_calculated()
        next_batch = 0
        for y in range(0, picture.shape[0] - self.x_patch_size + 1, self.x_patch_size):
            for x in range(0, picture.shape[1] - self.x_patch_size + 1, self.x_patch_size):
                batch = batches[next_batch]
                picture[y:y+self.x_patch_size, x:x + self.x_patch_size] = batch.reshape(batch.shape[0], batch.shape[0])
                next_batch += 1
        return picture

    @staticmethod
    def compare(correct, predicted):
        correct = np.round(correct/255, decimals=1)
        # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
        TP = np.sum(np.logical_and(predicted == 1, correct == 1))

        # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        TN = np.sum(np.logical_and(predicted == 0, correct == 0))

        # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        FP = np.sum(np.logical_and(predicted == 1, correct == 0))

        # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        FN = np.sum(np.logical_and(predicted == 0, correct == 1))

        print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP, FP, TN, FN))
        return TP, TN, FP, FN

    def convert_to_binary_image(self, image, threshold=None, positive=True):
        if threshold is None:
            mean = (np.max(image) + np.min(image)) / 2.0
        else:
            mean = threshold

        if positive:
            binary_image = image > mean
        else:
            binary_image = image < mean
        return binary_image

    def plot_raw(self, extraStr=''):
        self.plot_image(self.get_raw(), 'Raw ' + str(extraStr))

    def plot_manual(self, extraStr=''):
        self.plot_image(self.get_manual(), "Manual " + str(extraStr))

    def plot_calculated(self, extraStr='', binary=False):
        image = self.get_calculated()
        if binary:
            image = self.convert_to_binary_image(image)
        self.plot_image(image, "Calculated " + str(extraStr))

    @staticmethod
    def plot_image(image, title=''):
        if len(image.shape) > 2 and image.shape[2] == 1:
            image = image.reshape(image.shape[0], image.shape[1])
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()
