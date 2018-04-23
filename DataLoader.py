import os


class DataLoader:
    """Loades images from given directory."""

    __data_path = "./data"
    __raw_path = os.path.join(__data_path, "images")
    __manual_path = os.path.join(__data_path, "manual1")
    __mask_path = os.path.join(__data_path, "mask")
    __healthy = False
    __glaucomatous = False
    __diabetic = False
    __healthy_ends_with = "_h*"
    __glaucomatous_ends_with = "_g*"
    __diabetic_ends_with = "_dr*"

    def __init__(self, healthy=True, glaucomatous=False, diabetic=False):
        self.__healthy = healthy
        self.__glaucomatous = glaucomatous
        self.__diabetic = diabetic

        if (healthy):
            if (self.__isMissingFile(self.__healthy_ends_with)):
                raise FileNotFoundError("Missing files")
        if (glaucomatous):
            if (self.__isMissingFile(self.__glaucomatous_ends_with)):
                raise FileNotFoundError("Missing files")
        if (diabetic):
            if (self.__isMissingFile(self.__diabetic_ends_with)):
                raise FileNotFoundError("Missing files")

    def loadData(self):
        pass
        # TODO: Load data

    def countFiles(self, path, ends_with):
        counter = 0
        for file in os.listdir(path):
            if file.endswith(ends_with):
                counter += 1
        return counter

    def __isMissingFile(self, file_ending):
        images = self.countFiles(self.__raw_path, file_ending)
        manuals = self.countFiles(self.__manual_path, file_ending)
        masks = self.countFiles(self.__mask_path, file_ending)
        if (images == manuals == masks):
            return False
        else:
            return True
