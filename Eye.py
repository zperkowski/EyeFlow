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

    def __init__(self, raw, manual, mask, patchSize):
        self.__raw = raw
        self.__manual = manual
        self.__mask = mask
        self.__calculated = np.zeros(self.__manual.shape)
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

    def _get_batches(self, picture):
        batches = []
        for y in range(0, picture.shape[1] - self.patchSize + 1):
            for x in range(0, picture.shape[0] - self.patchSize + 1):
                sub_picture = picture[y:y + self.patchSize, x:x + self.patchSize]
                batches.append(sub_picture)
        return batches

    def get_batches_of_raw(self):
        return self._get_batches(self.get_raw())

    def get_batches_of_manual(self):
        return self._get_batches(self.get_manual())

    def build_image_from_batches(self, batches):
        picture = self.get_calculated()
        next_batch = 0
        for y in range(0, picture.shape[1] - self.patchSize + 1):
            for x in range(0, picture.shape[0] - self.patchSize + 1):
                picture[y:y + self.patchSize, x:x + self.patchSize] = batches[next_batch]
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

    def build_image(self, classification, threshold=0.5):
        flat_calculated = np.zeros(classification.shape)
        t = np.max(classification) * threshold
        for y in range(int(classification.shape[1])):
            for x in range(int(classification.shape[0])):
                if classification[x, y] > t:
                    flat_calculated[x, y] = 255

        self.__calculated = flat_calculated

    def plot_raw(self, extraStr=''):
        self.plot_image(self.get_raw(), 'Raw ' + str(extraStr))

    def plot_manual(self, extraStr=''):
        self.plot_image(self.get_manual(), "Manual " + str(extraStr))

    def plot_calculated(self, extraStr=''):
        self.plot_image(self.get_calculated(), "Calculated " + str(extraStr))

    @staticmethod
    def plot_image(image, title=''):
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()
