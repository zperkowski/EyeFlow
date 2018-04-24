class Eye:
    """This class contains all data about one eye: raw picture, correct result, and mask."""

    __raw = None
    __manual = None
    __mask = None

    def __init__(self, raw, manual, mask):
        self.__raw = raw
        self.__manual = manual
        self.__mask = mask
