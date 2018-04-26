import tensorflow as tf
import numpy as np
import matplotlib.image as mp_i


def runTensorFlow(eyes, verbose):
    if(verbose):
        printInfoAboutImages(eyes)

    model = tf.global_variables_initializer()
    with tf.Session() as session:
        merged = tf.summary.merge_all()
        if (verbose):
            writer = tf.summary.FileWriter("/tmp/tensorflowlogs", session.graph)
        session.run(model)

    if (verbose):
        print("\nTo use TensorBoard run:\n$tensorboard", "--logdir=/tmp/tensorflowlogs\n")


def printInfoAboutImages(eyes):
    eyes_h = eyes.get("h")
    eyes_g = eyes.get("g")
    eyes_d = eyes.get("d")
    if (len(eyes_h) > 0):
        print("\nShape of healthy images:")
        print("\t\tRaw:\t\t\t\t\tManual:\t\t\t\t\tMask:")
        for i in range(0, len(eyes_h)):
            print(i, end="\t\t")
            print(eyes_h[i].getRaw().shape, end="\t\t\t")
            print(eyes_h[i].getManual().shape, end="\t\t\t")
            print(eyes_h[i].getMask().shape, end="\n")
    if (len(eyes_g) > 0):
        print("\nShape of glaucomatous images:")
        print("\t\tRaw:\t\t\t\t\tManual:\t\t\t\t\tMask:")
        for i in range(0, len(eyes_g)):
            print(i, end="\t\t")
            print(eyes_g[i].getRaw().shape, end="\t\t\t")
            print(eyes_g[i].getManual().shape, end="\t\t\t")
            print(eyes_g[i].getMask().shape, end="\n")
    if (len(eyes_d) > 0):
        print("\nShape of diabetic images:")
        print("\t\tRaw:\t\t\t\t\tManual:\t\t\t\t\tMask:")
        for i in range(0, len(eyes_d)):
            print(i, end="\t\t")
            print(eyes_d[i].getRaw().shape, end="\t\t\t")
            print(eyes_d[i].getManual().shape, end="\t\t\t")
            print(eyes_d[i].getMask().shape, end="\n")
