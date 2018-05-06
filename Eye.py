import matplotlib.image as mp_i
import numpy as np

class Eye:
    """This class contains all data about one eye: raw picture, correct result, and mask."""

    __raw = None
    __manual = None
    __mask = None
    __calculated = None

    def __init__(self, raw, manual, mask):
        self.__raw = raw
        self.__manual = manual
        self.__mask = mask
        self.__calculated = np.zeros(self.__manual.shape)

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
                    total_pixels += 1
                    difference += abs((self.getManual()[x][y] / 255) - (self.getCalculated()[x][y] / 255))

        difference /= total_pixels
        return difference
