from skimage import color
import numpy as np
from random import randint
from matplotlib import pyplot as plt
import cv2


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
        self.__raw = cv2.resize(raw, dsize=(int(raw.shape[1]*0.5), int(raw.shape[0]*0.5)), interpolation=cv2.INTER_CUBIC)
        self.__manual = cv2.resize(manual, dsize=(int(manual.shape[1]*0.5), int(manual.shape[0]*0.5)), interpolation=cv2.INTER_CUBIC)
        self.__mask = cv2.resize(mask, dsize=(int(mask.shape[1]*0.5), int(mask.shape[0]*0.5)), interpolation=cv2.INTER_CUBIC)
        self.__calculated = np.zeros(self.__manual.shape)
        self.__rawGrey = color.rgb2gray(self.__raw)
        self.patchSize = patchSize
        self.offset = int(patchSize / 2)
        self.x = self.y = 0 + self.offset

    def getRaw(self):
        return self.__raw

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

        if (total_pixels > 0):
            difference /= total_pixels
        else:
            difference = 0.0
        return difference

    def buildImage(self, classification, threshold=0.5):
        flat_calculated = np.zeros(classification.shape)
        t = np.max(classification) * threshold
        for y in range(int(classification.shape[1])):
            for x in range(int(classification.shape[0])):
                if (classification[x, y] > t):
                    flat_calculated[x, y] = 255

        self.__calculated = flat_calculated

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
