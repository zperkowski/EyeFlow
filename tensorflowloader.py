import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os


def runTensorFlow(eyesToTrain, eyesToCalculate, batch_size, learning_rate, training_epochs, patch_size, verbose):
    if (verbose):
        print("Training data:", end='')
        printInfoAboutImages(eyesToTrain)
        print("\nProcess data:", end='')
        printInfoAboutImages(eyesToCalculate)

    # Parameters (with batch_size, learning_rate and training_epochs)
    display_step = 1
    print("\nMachine learning paramteres")
    print("Learning rate:\t\t" + str(learning_rate))
    print("Traning epochs:\t\t" + str(training_epochs))
    eye = eyesToTrain.get('h')[0]
    # tf Graph Input
    x = tf.placeholder(tf.float32, [1,
                                    eye.getManual().shape[0],
                                    eye.getManual().shape[1], 3])
    # 2 possibilities to place 2 colors on the center pixel
    y = tf.placeholder(tf.float32, [1,
                                    eye.getCalculated().shape[0],
                                    eye.getCalculated().shape[1]])

    scores = tf.layers.conv2d(x, 1, [patch_size,
                                     patch_size], padding='same', activation=None)
    y_pred = tf.sigmoid(scores)

    # Construct model
    loss = (tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.reshape(y, [-1]),
        logits=tf.reshape(scores, [-1])))

    # Minimize error using cross entropy
    optim = tf.train.AdamOptimizer(1e-2)
    step = optim.minimize(loss)

    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    # Training cycle
    # TODO: Process other than only healthy
    for i_eye in range(len(eyesToTrain.get('h'))):
        # Plot settings
        avg_set = []
        epoch_set = []
        print("\nTraining on " + str(i_eye + 1) + " image")
        eye = eyesToTrain.get('h')[i_eye]
        eye.plotRaw(extraStr=str(i_eye + 1) + " traning")
        eye.plotManual(extraStr=str(i_eye + 1) + " traning")
        for epoch in range(training_epochs):
            avg_cost = 0.
            # Loop over all patches
            prevProgress = "0.00%"

            batch_xs = eye.getRaw()
            batch_ys = eye.getCalculated()
            # Fit training using batch data
            session.run(step, feed_dict={x: batch_xs[None, :, :, :], y: batch_ys[None, :, :]})

            # Compute average loss
            # avg_cost += session.run(loss, feed_dict={x: batch_xs[None, :, :, :], y: batch_ys[None, :, :]})
            # Display logs per epoch step
            if epoch % display_step == 0:
                print(
                    "\rEpoch: " + '%04d' % (epoch + 1) + "\t\t100.00%" + "\t\tcost = " + "{:.9f}".format(avg_cost),
                    flush=True)
            avg_set.append(avg_cost)
            epoch_set.append(epoch + 1)
        print("Training phase finished for " + str(i_eye + 1) + " eye")

        # plt.plot(epoch_set, avg_set, 'o', label='Logistic Regression Training phase - ' + str(i_eye + 1) + ' eye')
        # plt.ylabel('cost')
        # plt.xlabel('epoch')
        # plt.legend()
        # plt.show()

        # Test model
        # correct_prediction = tf.equal(tf.argmax(step, 1), tf.argmax(y, 1))
        # Calculate accuracy
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # print("Model accuracy (training) " + str(i_eye + 1) + " eye: \t",
        #       accuracy.eval({x: eye.getNextBatch(accuracy_batch_size, True)[0],
        #                      y: eye.getNextBatch(accuracy_batch_size, True)[1]}))

    for i_eye in range(len(eyesToCalculate.get('h'))):
        eye = eyesToCalculate.get('h')[i_eye]
        eye.plotRaw(extraStr=str(i_eye + 1) + " processing")
        eye.plotManual(extraStr=str(i_eye + 1) + " processing")
        # Test model
        # correct_prediction = tf.equal(tf.argmax(step, 1), tf.argmax(y, 1))
        # # Calculate accuracy
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # # print("\nModel accuracy (processing) " + str(i_eye + 1) + " eye: \t",
        # #       accuracy.eval({x: eye.getNextBatch(accuracy_batch_size, True)[0],
        # #                      y: eye.getNextBatch(accuracy_batch_size, True)[1]}))

        feed_dict = {x: eye.getRaw()[None, :, :, :]}
        # Gets only the middle point
        classification = session.run(y_pred, feed_dict)
        print('\r' + "Calculated " + str(i_eye + 1)
              + " eye:\t\t\t\t\t\t\t100.00%", flush=True)
        print("Building an image based on predictions...")

        plt.imshow(classification[0, :, :, 0], cmap="gray")
        plt.show()

        for threshold in range(5, 10):
            threshold *= 0.1
            eye.buildImage(classification, threshold)
            eye.plotCalculated(extraStr=str(i_eye + 1) + " processing " + str(threshold))
            print("Difference between manual and predicted " + str(threshold) + ":\t\t"
                  + "{:.2f}".format(eye.compare() * 100) + "%")

        if (verbose):
            path = os.path.join(tempfile.gettempdir(), "tensorflowlogs")
            tf.summary.FileWriter(path, session.graph)
            print("\nTo use TensorBoard run:\n$tensorboard", "--logdir=" + path + "\n")


def printInfoAboutImages(eyes):
    eyes_h = eyes.get("h")
    eyes_g = eyes.get("g")
    eyes_d = eyes.get("d")
    if (len(eyes_h) > 0):
        print("\nShape of healthy images:")
        print("\t\tRaw:\t\t\t\t\tManual:\t\t\t\t\tMask:")
        for i in range(0, len(eyes_h)):
            print(str(i + 1), end="\t\t")
            print(eyes_h[i].getRaw().shape, end="\t\t\t")
            print(eyes_h[i].getManual().shape, end="\t\t\t")
            print(eyes_h[i].getMask().shape, end="\n")
    if (len(eyes_g) > 0):
        print("\nShape of glaucomatous images:")
        print("\t\tRaw:\t\t\t\t\tManual:\t\t\t\t\tMask:")
        for i in range(0, len(eyes_g)):
            print(str(i + 1), end="\t\t")
            print(eyes_g[i].getRaw().shape, end="\t\t\t")
            print(eyes_g[i].getManual().shape, end="\t\t\t")
            print(eyes_g[i].getMask().shape, end="\n")
    if (len(eyes_d) > 0):
        print("\nShape of diabetic images:")
        print("\t\tRaw:\t\t\t\t\tManual:\t\t\t\t\tMask:")
        for i in range(0, len(eyes_d)):
            print(str(i + 1), end="\t\t")
            print(eyes_d[i].getRaw().shape, end="\t\t\t")
            print(eyes_d[i].getManual().shape, end="\t\t\t")
            print(eyes_d[i].getMask().shape, end="\n")
