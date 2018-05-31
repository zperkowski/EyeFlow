import tensorflow as tf
import matplotlib.pyplot as plt
import tempfile
import os


def runTensorFlow(eyesToTrain, eyesToCalculate, batch_size, learning_rate, training_epochs, verbose):
    if (verbose):
        print("Training data:", end='')
        printInfoAboutImages(eyesToTrain)
        print("\nProcess data:", end='')
        printInfoAboutImages(eyesToCalculate)

    # Parameters (with batch_size, learning_rate and training_epochs)
    accuracy_batch_size = batch_size * 10
    display_step = 1
    print("\nMachine learning paramteres")
    print("Batch size:\t\t\t" + str(batch_size))
    print("Learning rate:\t\t" + str(learning_rate))
    print("Traning epochs:\t\t" + str(training_epochs))

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

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        # Training cycle
        # TODO: Process other than only healthy
        for i_eye in range(len(eyesToTrain.get('h'))):
            # Plot settings
            avg_set = []
            epoch_set = []
            print("\nTraining on " + str(i_eye + 1) + " image")
            eye = eyesToTrain.get('h')[i_eye]
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(eye.numOfSamples() / batch_size)
                # Loop over all batches
                prevProgress = "0.00%"
                for i in range(total_batch):
                    progress = "{:.2f}".format(i / total_batch * 100) + "%"
                    if (progress != prevProgress):
                        print('\r' + "Epoch:", '%04d' % (epoch + 1) + "\t\t" + progress, end='', flush=True)
                        prevProgress = progress
                    batch_xs, batch_ys = eye.getNextBatch(batch_size)
                    # Fit training using batch data
                    session.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
                    # Compute average loss
                    avg_cost += session.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print(
                        "\rEpoch: " + '%04d' % (epoch + 1) + "\t\t100.00%" + "\t\tcost = " + "{:.9f}".format(avg_cost),
                        flush=True)
                avg_set.append(avg_cost)
                epoch_set.append(epoch + 1)
            print("Training phase finished for " + str(i_eye + 1) + " eye")

            plt.plot(epoch_set, avg_set, 'o', label='Logistic Regression Training phase - ' + str(i_eye + 1) + ' eye')
            plt.ylabel('cost')
            plt.xlabel('epoch')
            plt.legend()
            plt.show()

            # Test model
            correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Model accuracy (training) " + str(i_eye + 1) + " eye: \t",
                  accuracy.eval({x: eye.getNextBatch(accuracy_batch_size, True)[0],
                                 y: eye.getNextBatch(accuracy_batch_size, True)[1]}))

        for i_eye in range(len(eyesToCalculate.get('h'))):
            eye = eyesToCalculate.get('h')[i_eye]
            # Test model
            correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("\nModel accuracy (processing) " + str(i_eye + 1) + " eye: \t",
                  accuracy.eval({x: eye.getNextBatch(accuracy_batch_size, True)[0],
                                 y: eye.getNextBatch(accuracy_batch_size, True)[1]}))

            total_batch = int(eye.numOfSamples() / batch_size)
            classification = []
            prevProgress = "0.00%"
            for i in range(total_batch):
                feed_dict = {x: eye.getNextBatch(batch_size)[0]}
                classification.extend(session.run(activation, feed_dict))
                progress = "{:.2f}".format(i / total_batch * 100) + "%"
                if (progress != prevProgress):
                    print('\r' + "Calculating " + str(i_eye + 1)
                          + " eye:\t\t\t\t\t\t\t\t" + progress, end='', flush=True)
                    prevProgress = progress
            print('\r' + "Calculated " + str(i_eye + 1)
                  + " eye:\t\t\t\t\t\t\t\t100.00%", flush=True)
            print("Building an image based on predictions...")
            eye.buildImage(classification)

            print("Difference between manual and predicted:\t\t"
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
