import tensorflow as tf
import numpy as np
import matplotlib.image as mp_i
import matplotlib.pyplot as plt


def runTensorFlow(eyes, verbose):
    if(verbose):
        printInfoAboutImages(eyes)

    # Choose a picture
    eye = eyes.get('h')[0]

    # Parameters
    learning_rate = 0.01
    training_epochs = 25
    batch_size = 100
    display_step = 1

    # tf Graph Input
    # image of 5x5 pixels = 25
    x = tf.placeholder("float", [None, 25])
    # 2 possibilities to place 2 colors on the center pixel
    y = tf.placeholder("float", [None, 2])

    # Create model

    # Set model weights
    W = tf.Variable(tf.zeros([25, 2]))
    b = tf.Variable(tf.zeros([2]))

    # Construct model
    activation = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

    # Minimize error using cross entropy
    cross_entropy = y * tf.log(activation)
    cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices=1))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Plot settings
    avg_set = []
    epoch_set = []

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        # TODO: Find why it crashes
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(eye.numOfSamples() / batch_size)
            # Loop over all batches
            tmp = 0
            for i in range(total_batch):
                batch_xs, batch_ys = eye.getNextBatch(batch_size)
                tmp += len(batch_xs)
                # Fit training using batch data
                session.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
                # Compute average loss
                avg_cost += session.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            avg_set.append(avg_cost)
            epoch_set.append(epoch + 1)
        print("Training phase finished")

        plt.plot(epoch_set, avg_set, 'o', label='Logistic Regression Training phase')
        plt.ylabel('cost')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

        # Test model
        correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Model accuracy:", accuracy.eval({x: eye.getNextBatch()[0], y: eye.getNextBatch()[1]}))

        if (verbose):
            tf.summary.FileWriter("/tmp/tensorflowlogs", session.graph)
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
