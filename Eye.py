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
    patchSize = 0
    x = int()
    y = int()

    def __init__(self, raw, manual, mask, patchSize, resize=None):
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
        if patchSize > self.get_raw().shape[0] or patchSize > self.get_raw().shape[1]:
            self.patchSize = min(self.get_raw().shape[0], self.get_raw().shape[1])
            print("Warning: Size of patch bigger than image. Current patch size: " + str(self.patchSize))
        else:
            self.patchSize = patchSize

    def get_raw(self):
        return self.__raw

    def get_manual(self):
        return self.__manual

    def get_mask(self):
        return self.__mask

    def get_calculated(self):
        return self.__calculated

    def _generate_batches(self, picture):
        batches = []
        for y in range(0, picture.shape[0] - self.patchSize + 1, self.patchSize):
            for x in range(0, picture.shape[1] - self.patchSize + 1, self.patchSize):
                sub_picture = picture[y:y + self.patchSize, x:x + self.patchSize]
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
        self.__raw_batches = self._generate_batches(self.__raw)

    def generate_batches_of_manual(self):
        self.__manual_batches = self._generate_batches(self.__manual)

    def generate_batches_of_calculated(self):
        self.__calculated_batches = self._generate_batches(self.__calculated)

    def build_image_from_batches(self, batches):
        picture = self.get_calculated()
        next_batch = 0
        for y in range(0, picture.shape[0] - self.patchSize + 1, self.patchSize):
            for x in range(0, picture.shape[1] - self.patchSize + 1, self.patchSize):
                batch = batches[next_batch]
                picture[y:y+self.patchSize, x:x+self.patchSize] = batch.reshape(batch.shape[0], batch.shape[0])
                next_batch += 1
        return picture

    def compare(self):
        w = self.get_calculated().shape[0]
        h = self.get_calculated().shape[1]
        total_pixels = 0
        difference = 0.0

        for x in range(w):
            for y in range(h):
                if self.get_mask()[x][y][0] == self.get_mask()[x][y][1] == self.get_mask()[x][y][2] == 255:
                    if self.get_manual()[x][y] != self.get_calculated()[x][y]:
                        total_pixels += 1
                        difference += abs((self.get_manual()[x][y] / 255) - (self.get_calculated()[x][y] / 255))

        if total_pixels > 0:
            difference /= total_pixels
        else:
            difference = 0.0
        return difference

    def plot_raw(self, extraStr=''):
        self.plot_image(self.get_raw(), 'Raw ' + str(extraStr))

    def plot_manual(self, extraStr=''):
        self.plot_image(self.get_manual(), "Manual " + str(extraStr))

    def plot_calculated(self, extraStr=''):
        self.plot_image(self.get_calculated(), "Calculated " + str(extraStr))

    @staticmethod
    def plot_image(image, title=''):
        if len(image.shape) > 2 and image.shape[2] == 1:
            image = image.reshape(image.shape[0], image.shape[1])
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()
