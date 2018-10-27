import os
import Eye
import matplotlib.image as mp_i


class DataLoader:
    """Loades images from given directory."""

    _data_path = os.path.join(".", "data")
    _raw_path = os.path.join(_data_path, "images")
    _manual_path = os.path.join(_data_path, "manual1")
    _mask_path = os.path.join(_data_path, "mask")
    _load_healthy = False
    _load_glaucomatous = False
    _load_diabetic = False
    _healthy_ends_with = "_h"
    _glaucomatous_ends_with = "_g"
    _diabetic_ends_with = "_dr"

    files_raw = []
    files_manual = []
    files_mask = []
    patchSize = int()

    _startLearning = int()
    _endLearning = int()
    _startProcessing = int()
    _endProcessing = int()

    def __init__(self, healthy=True, glaucomatous=False, diabetic=False,
                 start_learning=1, end_learning=1, start_processing=2, end_processing=2,
                 patch_size=15):
        self._load_healthy = healthy
        self._load_glaucomatous = glaucomatous
        self._load_diabetic = diabetic
        self._startLearning = start_learning - 1
        self._endLearning = end_learning - 1
        self._startProcessing = start_processing - 1
        self._endProcessing = end_processing - 1
        self.patchSize = patch_size

    def loadData(self, verbose):
        if self._load_healthy:
            if not self.__validate_number_of_files(self._healthy_ends_with):
                raise FileNotFoundError("Missing files of healthy images")
        if self._load_glaucomatous:
            if not self.__validate_number_of_files(self._glaucomatous_ends_with):
                raise FileNotFoundError("Missing files of glaucomatous images")
        if self._load_diabetic:
            if not self.__validate_number_of_files(self._diabetic_ends_with):
                raise FileNotFoundError("Missing files of diabetic images")

        eyes_to_train = {"h": [],  # healthy
                         "g": [],  # glaucomatous
                         "d": []}  # diabetic
        eyes_to_process = {"h": [],  # healthy
                           "g": [],  # glaucomatous
                           "d": []}  # diabetic
        self.files_raw = os.listdir(self._raw_path)
        self.files_manual = os.listdir(self._manual_path)
        self.files_mask = os.listdir(self._mask_path)

        if self._load_healthy:
            files_raw_healthy = self.filter_files(self.files_raw,
                                                  self._healthy_ends_with,
                                                  self._startLearning,
                                                  self._endLearning)
            eyes_to_train["h"] = self.load_list_of_eyes(files_raw_healthy,
                                                        self._startLearning,
                                                        self._endLearning,
                                                        verbose)
            eyes_to_process["h"] = self.load_list_of_eyes(files_raw_healthy,
                                                          self._startProcessing,
                                                          self._endProcessing,
                                                          verbose)

        if self._load_diabetic:
            files_raw_diabetic = self.filter_files(self.files_raw,
                                                   self._diabetic_ends_with,
                                                   self._startLearning,
                                                   self._endLearning)
            eyes_to_train["d"] = self.load_list_of_eyes(files_raw_diabetic,
                                                        self._startLearning,
                                                        self._endLearning,
                                                        verbose)
            eyes_to_process["d"] = self.load_list_of_eyes(files_raw_diabetic,
                                                          self._startProcessing,
                                                          self._endProcessing,
                                                          verbose)

        if self._load_glaucomatous:
            files_raw_glaucomatous = self.filter_files(self.files_raw,
                                                       self._glaucomatous_ends_with,
                                                       self._startLearning,
                                                       self._endLearning)
            eyes_to_train["g"] = self.load_list_of_eyes(files_raw_glaucomatous,
                                                        self._startLearning,
                                                        self._endLearning,
                                                        verbose)
            eyes_to_process["g"] = self.load_list_of_eyes(files_raw_glaucomatous,
                                                          self._startProcessing,
                                                          self._endProcessing,
                                                          verbose)

        print("Loaded " + str(len(eyes_to_train.get("h"))) + " healthy images to train")
        print("Loaded " + str(len(eyes_to_train.get("g"))) + " glaucomatous images to train")
        print("Loaded " + str(len(eyes_to_train.get("d"))) + " diabetic images to train")

        print("Loaded " + str(len(eyes_to_process.get("h"))) + " healthy images to process")
        print("Loaded " + str(len(eyes_to_process.get("g"))) + " glaucomatous images to process")
        print("Loaded " + str(len(eyes_to_process.get("d"))) + " diabetic images to process")

        return eyes_to_train, eyes_to_process

    @staticmethod
    def filter_files(list_of_files, filename_ends_with, start, end):
        files = []
        numbers = DataLoader.generate_list_of_numbers(start+1, end+1)
        for file in list_of_files:
            if filename_ends_with in file and True in [num in file for num in numbers]:
                files.append(file)
        return files

    def load_list_of_eyes(self, list_of_files, start, end, verbose):
        list_of_eyes = []
        for i in range(start, end):
            path = list_of_files[i]
            if verbose:
                print("Loading: " + path)
            eye = self.loadEye(path, i % 2 != 0)
            if eye is not None:
                list_of_eyes.append(eye)
        return list_of_eyes

    @staticmethod
    def generate_list_of_numbers(start, end):
        numbers = []
        for i in range(start, end + 1):
            s = "%02d" % i
            numbers.append(s)
        return numbers

    @staticmethod
    def count_files(path, path_ends_with):
        counter = 0
        for file in os.listdir(path):
            if file.endswith(path_ends_with):
                counter += 1
        return counter

    def __validate_number_of_files(self, filename_ending):
        images = self.count_files(self._raw_path, filename_ending)
        manuals = self.count_files(self._manual_path, filename_ending)
        masks = self.count_files(self._mask_path, filename_ending)
        if images == manuals == masks:
            return True
        else:
            return False

    def loadEye(self, file, reverse=False):
        # Todo: Refactor .split('.')[0] + ".tif" and "_mask.tif"
        if not reverse:
            img_raw = mp_i.imread(os.path.join(self._raw_path, file))
            img_manual = mp_i.imread(os.path.join(self._manual_path, file.split('.')[0] + ".tif"))
            img_mask = mp_i.imread(os.path.join(self._mask_path, file.split('.')[0] + "_mask.tif"))
        else:
            img_raw = mp_i.imread(os.path.join(self._raw_path, file))[:, ::-1]
            img_manual = mp_i.imread(os.path.join(self._manual_path, file.split('.')[0] + ".tif"))[:, ::-1]
            img_mask = mp_i.imread(os.path.join(self._mask_path, file.split('.')[0] + "_mask.tif"))[:, ::-1]
        eye = Eye.Eye(img_raw, img_manual, img_mask, self.patchSize)
        eye.plot_raw()
        eye.plot_manual()
        eye.plot_image(eye.get_mask())
        return eye
