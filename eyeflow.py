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
        args.endLearning,
        args.startProcessing,
        args.endProcessing,
        args.patchSize)
    eyesToTrain, eyesToCalculate = dataLoader.loadData(verbose=args.verbose)

    tfl.runTensorFlow(
        eyesToTrain,
        eyesToCalculate,
        batch_size=args.batch,
        learning_rate=args.learningRate,
        training_epochs=args.epochs,
        patch_size=args.patchSize,
        verbose=args.verbose)
