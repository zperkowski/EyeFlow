import os

# Disabling the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import DataLoader, ui


def runTensorFlow(verbose):
    a = tf.constant(10, name="a")
    b = tf.constant(90, name="b")
    y = tf.Variable(a + b * 2, name="y")

    model = tf.global_variables_initializer()
    with tf.Session() as session:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/tmp/tensorflowlogs", session.graph)
        session.run(model)
        print(session.run(y))

    if (verbose):
        print("\nTo use TensorBoard run:\n$tensorboard", "--logdir=/tmp/tensorflowlogs\n")


if __name__ == '__main__':
    args = ui.parseArgs()

    dataLoader = DataLoader.DataLoader(args.healthy, args.glaucomatous, args.diabetic)
    eyes = dataLoader.loadData(verbose=args.verbose)

    runTensorFlow(verbose=args.verbose)
