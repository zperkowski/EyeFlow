import argparse


def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("-sl", "--startLearning", action='store', default=1,
                        dest='startLearning', help='at which image will start learning', type=int)
    parser.add_argument("-el", "--endLearning", action='store', default=1,
                        dest='endLearning', help='at which image will stop learning', type=int)
    parser.add_argument("-sp", "--startProcessing", action='store', default=2,
                        dest='startProcessing', help='at which image will start processing', type=int)
    parser.add_argument("-ep", "--endProcessing", action='store', default=2,
                        dest='endProcessing', help='at which image will stop processing', type=int)
    parser.add_argument("-b", "--batch", action='store', default=1000,
                        dest='batch', help='size of a batch', type=int)
    parser.add_argument("-e", "--epochs", action='store', default=5,
                        dest='epochs', help='number of epochs', type=int)
    parser.add_argument("-lr", "--learningRate", action='store', default=0.01,
                        dest='learningRate', help='learning rate', type=float)
    parser.add_argument("-p", "--patchSize", action='store', default=15,
                        dest='patchSize', help='size of patch', type=int)
    parser.add_argument("-y", "--healthy", action='store_true', default=True,
                        dest='healthy', help='analysing healthy images')
    parser.add_argument("-g", "--glaucomatous", action='store_true', default=False,
                        dest='glaucomatous', help='analysing glaucomatous images')
    parser.add_argument("-d", "--diabetic", action='store_true', default=False,
                        dest='diabetic', help='analysing diabetic images')
    parser.add_argument("-v", "--verbose", action='store_true', default=False,
                        dest='verbose', help='generates more output and TensorFlow graphs')

    return parser.parse_args()
