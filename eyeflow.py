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
        args.patchSize)
    eyes_to_train, eyes_to_calculate = dataLoader.loadData(verbose=args.verbose)

    tfl.runTensorFlow(
        eyes_to_train,
        eyes_to_calculate,
        batch_size=args.batch,
        learning_rate=args.learningRate,
        training_epochs=args.epochs,
        patch_size=args.patchSize,
        verbose=args.verbose)
