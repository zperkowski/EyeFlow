import os

# Disabling the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflowloader as tfl
import DataLoader, ui


if __name__ == '__main__':
    args = ui.parseArgs()

    dataLoader = DataLoader.DataLoader(args.healthy, args.glaucomatous, args.diabetic)
    eyes = dataLoader.loadData(verbose=args.verbose)

    tfl.runTensorFlow(eyes, verbose=args.verbose)
