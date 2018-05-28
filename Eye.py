from skimage import color
import numpy as np
from random import randint

class Eye:
    """This class contains all data about one eye: raw picture, correct result, and mask."""

    __raw = None
    __rawGrey = None
    __manual = None
    __mask = None
    __calculated = None
    patchSize = 5
    offset = int(patchSize / 2)
    x = 0 + offset
    y = 0 + offset

    def __init__(self, raw, manual, mask):
        self.__raw = raw
        self.__manual = manual
        self.__mask = mask
        self.__calculated = np.zeros(self.__manual.shape)
        self.__rawGrey = color.rgb2gray(self.__raw)

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

    def getNextBatch(self, batchSize, random=False):
        batch = (np.empty([batchSize, 25]), np.empty([batchSize, 2]))
        for i in range(batchSize):
            if (random):
                self.x = randint(int(0.0 + self.getCalculated().shape[0] * 0.15),
                                 int(self.getCalculated().shape[0]
                                 - self.getCalculated().shape[0] * 0.15))
                self.y = randint(0, self.getCalculated().shape[1])

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