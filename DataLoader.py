import os
import Eye


class DataLoader:
    """Loades images from given directory."""

    _data_path = "./data"
    _raw_path = os.path.join(_data_path, "images")
    _manual_path = os.path.join(_data_path, "manual1")
    _mask_path = os.path.join(_data_path, "mask")
    _load_healthy = False
    _load_glaucomatous = False
    _load_diabetic = False
    _healthy_ends_with = "_h"
    _glaucomatous_ends_with = "_g"
    _diabetic_ends_with = "_dr"

    def __init__(self, healthy=True, glaucomatous=False, diabetic=False):
        self._load_healthy = healthy
        self._load_glaucomatous = glaucomatous
        self._load_diabetic = diabetic

        if (healthy):
            if (self.__isMissingFile(self._healthy_ends_with)):
                raise FileNotFoundError("Missing files")
        if (glaucomatous):
            if (self.__isMissingFile(self._glaucomatous_ends_with)):
                raise FileNotFoundError("Missing files")
        if (diabetic):
            if (self.__isMissingFile(self._diabetic_ends_with)):
                raise FileNotFoundError("Missing files")

    def loadData(self):
        dict_of_eyes = {"h": [],  # healthy
                        "g": [],  # glaucomatous
                        "d": []}  # diabetic
        files_raw = os.listdir(self._raw_path)
        files_manual = os.listdir(self._manual_path)
        files_mask = os.listdir(self._mask_path)
        for i in range(0, len(files_raw)):
            print("Loading: " + files_raw[i] + "\t" + files_manual[i] + "\t" + files_mask[i])
            if (self._load_healthy
                    and files_raw[i].endswith(self._healthy_ends_with, 2, 4)
                    and files_manual[i].endswith(self._healthy_ends_with, 2, 4)
                    and files_mask[i].endswith(self._healthy_ends_with, 2, 4)):
                h_eye = Eye.Eye(files_raw[i], files_manual[i], files_mask[i])
                dict_of_eyes.get("h").append(h_eye)
            if (self._load_glaucomatous
                    and files_raw[i].endswith(self._glaucomatous_ends_with, 2, 4)
                    and files_manual[i].endswith(self._glaucomatous_ends_with, 2, 4)
                    and files_mask[i].endswith(self._glaucomatous_ends_with, 2, 4)):
                g_eye = Eye.Eye(files_raw[i], files_manual[i], files_mask[i])
                dict_of_eyes.get("g").append(g_eye)
            if (self._load_diabetic
                    and files_raw[i].endswith(self._diabetic_ends_with, 2, 5)
                    and files_manual[i].endswith(self._diabetic_ends_with, 2, 5)
                    and files_mask[i].endswith(self._diabetic_ends_with, 2, 5)):
                d_eye = Eye.Eye(files_raw[i], files_manual[i], files_mask[i])
                dict_of_eyes.get("d").append(d_eye)

        print("Loaded " + str(len(dict_of_eyes.get("h"))) + " healthy images")
        print("Loaded " + str(len(dict_of_eyes.get("g"))) + " glaucomatous images")
        print("Loaded " + str(len(dict_of_eyes.get("d"))) + " diabetic images")

        return dict_of_eyes

    def countFiles(self, path, ends_with):
        counter = 0
        for file in os.listdir(path):
            if file.endswith(ends_with):
                counter += 1
        return counter

    def __isMissingFile(self, file_ending):
        images = self.countFiles(self._raw_path, file_ending)
        manuals = self.countFiles(self._manual_path, file_ending)
        masks = self.countFiles(self._mask_path, file_ending)
        if (images == manuals == masks):
            return False
        else:
            return True
