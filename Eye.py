from skimage import color
import numpy as np
from random import randint
from matplotlib import pyplot as plt


class Eye:
    """This class contains all data about one eye: raw picture, correct result, and mask."""

    __raw = None
    __rawGrey = None
    __manual = None
    __mask = None
    __calculated = None
    patchSize = 5
    offset = int()
    x = int()
    y = int()

    def __init__(self, raw, manual, mask, patchSize):
        self.__raw = raw
        self.__manual = manual
        self.__mask = mask
        self.__calculated = np.zeros(self.__manual.shape)
        self.__rawGrey = color.rgb2gray(self.__raw)
        self.patchSize = patchSize
        self.offset = int(patchSize / 2)
        self.x = self.y = 0 + self.offset

    def getRaw(self):
        return self.__rawGrey

    def getManual(self):
        return self.__manual

    def getMask(self):
        return self.__mask

    def getCalculated(self):
        return self.__calculated

    def compare(self):
        w = self.getCalculated().shape[0]
        h = self.getCalculated().shape[1]
        total_pixels = 0
        difference = 0.0

        for x in range(w):
            for y in range(h):
                if (self.getMask()[x][y][0] == self.getMask()[x][y][1] == self.getMask()[x][y][2] == 255):
                    if (self.getManual()[x][y] != self.getCalculated()[x][y]):
                        total_pixels += 1
                        difference += abs((self.getManual()[x][y] / 255) - (self.getCalculated()[x][y] / 255))

        difference /= total_pixels
        return difference

    def numOfSamples(self):
        numOfSamples = (self.getRaw().shape[0] - self.offset * 2) * (self.getRaw().shape[1] - self.offset * 2)
        return numOfSamples

    """Returns a patch and true value of detected vein.
    If detected(white) [0, 1]. If not(black) [1, 0]."""
    def getNextBatch(self, batchSize, random=False):
        batch = (np.empty([batchSize, self.patchSize * self.patchSize]), np.empty([batchSize, 2]))
        for i in range(batchSize):
            if (random):
                self.x = randint(int(self.getCalculated().shape[0] * 0.15),
                                 int(self.getCalculated().shape[0] * 0.75))
                self.y = randint(int(self.getCalculated().shape[1] * 0.15),
                                 int(self.getCalculated().shape[1] * 0.75))

            pathRaw = self.getRaw()[self.x - self.offset: self.x + self.offset + 1, self.y - self.offset: self.y + self.offset + 1]
            if (self.getManual()[self.x][self.y] == 255):
                found = [0, 1]
            else:
                found = [1, 0]
            batch[0][i] = np.asarray(pathRaw).flatten()
            batch[1][i] = found
            self.x += 1
            if (self.getRaw().shape[0] - self.offset - 1 < self.x):
                self.x = 0 + self.offset
                self.y += 1
            if (self.getRaw().shape[1] - self.offset - 1 < self.y):
                self.x = 0 + self.offset
                self.y = 0 + self.offset
        return batch

    def buildImage(self, classification):
        flat_calculated = np.zeros(len(classification))
        for i in range(flat_calculated.shape[0]):
            # Adds only white on the black background
            if (classification[i][1] >= 0.5):
                flat_calculated[i] = 255

        # first lines of pixels are ignored depending on patch size
        offset = self.getCalculated().shape[0] * self.offset
        # Need to extend the list to the proper length - width * height
        # When the batch size is bigger than 1 it can cut the last part of the image
        missing = self.getCalculated().shape[0] * self.getCalculated().shape[1] - flat_calculated.shape[0] - offset
        flat_calculated = np.pad(flat_calculated, (offset, missing), 'constant', constant_values=(0, 0))
        self.__calculated = flat_calculated.reshape(self.getCalculated().shape)

    def plotRaw(self, extraStr=''):
        self.plotImage(self.getRaw(), 'Raw ' + str(extraStr))

    def plotManual(self, extraStr=''):
        self.plotImage(self.getManual(), "Manual " + str(extraStr))

    def plotCalculated(self, extraStr=''):
        self.plotImage(self.getCalculated(), "Calculated " + str(extraStr))

    def plotImage(self, image, title=''):
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()
