import os

# Disabling the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflowloader as tfl
import DataLoader, ui


if __name__ == '__main__':
    args = ui.parseArgs()

    dataLoader = DataLoader.DataLoader(
        args.healthy,
        args.glaucomatous,
        args.diabetic,
        args.startLearning,
        args.endLearning+1,
        args.startProcessing,
        args.endProcessing+1,
        x_patch_size=572,
        y_patch_size=388)   # Todo: Add to arg parser
    eyes_to_train, eyes_to_calculate = dataLoader.loadData(verbose=args.verbose)

    tfl.runTensorFlow(
        eyes_to_train,
        eyes_to_calculate,
        batch_size=args.batch,
        learning_rate=args.learningRate,
        training_epochs=args.epochs,
        x_patch_size=572,
        y_patch_size=388,   # Todo: Add to arg parser
        verbose=args.verbose)

# X patch size: 572
# Y patch size: 388
