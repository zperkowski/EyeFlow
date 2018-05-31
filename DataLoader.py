import os
import Eye
import matplotlib.image as mp_i


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

    files_raw = []
    files_manual = []
    files_mask = []
    patchSize = int()

    _startLearning = int()
    _endLearning = int()
    _startProcessing = int()
    _endProcessing = int()

    def __init__(self, healthy=True, glaucomatous=False, diabetic=False,
                 startLearning=1, endLearning=1, startProcessing=2, endProcessing=2,
                 patchSize=15):
        self._load_healthy = healthy
        self._load_glaucomatous = glaucomatous
        self._load_diabetic = diabetic
        self._startLearning = startLearning-1
        self._endLearning = endLearning-1
        self._startProcessing = startProcessing-1
        self._endProcessing = endProcessing-1
        self.patchSize = patchSize

        if (healthy):
            if (self.__isMissingFile(self._healthy_ends_with)):
                raise FileNotFoundError("Missing files")
        if (glaucomatous):
            if (self.__isMissingFile(self._glaucomatous_ends_with)):
                raise FileNotFoundError("Missing files")
        if (diabetic):
            if (self.__isMissingFile(self._diabetic_ends_with)):
                raise FileNotFoundError("Missing files")

    def loadData(self, verbose):
        eyesToTrain = {"h": [],  # healthy
                       "g": [],  # glaucomatous
                       "d": []}  # diabetic
        eyesToProcess = {"h": [],  # healthy
                         "g": [],  # glaucomatous
                         "d": []}  # diabetic
        self.files_raw = os.listdir(self._raw_path)
        self.files_manual = os.listdir(self._manual_path)
        self.files_mask = os.listdir(self._mask_path)
        for i in range(self._startLearning*3, (self._endLearning+1)*3):
            if (verbose):
                print("Loading: " + self.files_raw[i] + "\t" + self.files_manual[i] + "\t" + self.files_mask[i])
            if (self._load_healthy):
                eye = self.loadHealthyEye(i)
                if (eye != None):
                    eyesToTrain.get("h").append(eye)

            if (self._load_glaucomatous):
                eye = self.loadGlaucomatousEye(i)
                if (eye != None):
                    eyesToTrain.get("d").append(eye)

            if (self._load_diabetic):
                eye = self.loadDiabeticEye(i)
                if (eye != None):
                    eyesToTrain.get("g").append(eye)

        for i in range(self._startProcessing*3, (self._endProcessing+1)*3):
            if (verbose):
                print("Loading: " + self.files_raw[i] + "\t" + self.files_manual[i] + "\t" + self.files_mask[i])
            if (self._load_healthy):
                eye = self.loadHealthyEye(i)
                if (eye != None):
                    eyesToProcess.get("h").append(eye)

            if (self._load_glaucomatous):
                eye = self.loadGlaucomatousEye(i)
                if (eye != None):
                    eyesToProcess.get("d").append(eye)

            if (self._load_diabetic):
                eye = self.loadDiabeticEye(i)
                if (eye != None):
                    eyesToProcess.get("g").append(eye)

        print("Loaded " + str(len(eyesToTrain.get("h"))) + " healthy images to train")
        print("Loaded " + str(len(eyesToTrain.get("g"))) + " glaucomatous images to train")
        print("Loaded " + str(len(eyesToTrain.get("d"))) + " diabetic images to train")

        print("Loaded " + str(len(eyesToProcess.get("h"))) + " healthy images to process")
        print("Loaded " + str(len(eyesToProcess.get("g"))) + " glaucomatous images to process")
        print("Loaded " + str(len(eyesToProcess.get("d"))) + " diabetic images to process")

        return eyesToTrain, eyesToProcess

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

    def loadHealthyEye(self, number):
        if (self.files_raw[number].endswith(self._healthy_ends_with, 2, 4)
                and self.files_manual[number].endswith(self._healthy_ends_with, 2, 4)
                and self.files_mask[number].endswith(self._healthy_ends_with, 2, 4)):
            img_raw = mp_i.imread(os.path.join(self._raw_path, self.files_raw[number]))
            img_manual = mp_i.imread(os.path.join(self._manual_path, self.files_manual[number]))
            img_mask = mp_i.imread(os.path.join(self._mask_path, self.files_mask[number]))
            h_eye = Eye.Eye(img_raw, img_manual, img_mask, self.patchSize)
            return h_eye
        else:
            return None

    def loadGlaucomatousEye(self, number):
        if (self.files_raw[number].endswith(self._glaucomatous_ends_with, 2, 4)
                and self.files_manual[number].endswith(self._glaucomatous_ends_with, 2, 4)
                and self.files_mask[number].endswith(self._glaucomatous_ends_with, 2, 4)):
            img_raw = mp_i.imread(os.path.join(self._raw_path, self.files_raw[number]))
            img_manual = mp_i.imread(os.path.join(self._manual_path, self.files_manual[number]))
            img_mask = mp_i.imread(os.path.join(self._mask_path, self.files_mask[number]))
            g_eye = Eye.Eye(img_raw, img_manual, img_mask, self.patchSize)
            return g_eye
        else:
            return None

    def loadDiabeticEye(self, number):
        if (self.files_raw[number].endswith(self._diabetic_ends_with, 2, 5)
                and self.files_manual[number].endswith(self._diabetic_ends_with, 2, 5)
                and self.files_mask[number].endswith(self._diabetic_ends_with, 2, 5)):
            img_raw = mp_i.imread(os.path.join(self._raw_path, self.files_raw[number]))
            img_manual = mp_i.imread(os.path.join(self._manual_path, self.files_manual[number]))
            img_mask = mp_i.imread(os.path.join(self._mask_path, self.files_mask[number]))
            d_eye = Eye.Eye(img_raw, img_manual, img_mask, self.patchSize)
            return d_eye
        else:
            return None
