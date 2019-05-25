import random

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
from sklearn.preprocessing import normalize

global x, y, loss


def setup_model(batches, shape_x, shape_y, patch_size):
    global x, y, loss
    # tf Graph Input
    x = tf.placeholder(tf.float32, (batches, shape_x, shape_y, 3))
    # 2 possibilities to place 2 colors on the center pixel
    y = tf.placeholder(tf.float32, (batches, shape_x, shape_y, 1))

    y_pred = tf.layers.conv2d(x, 1, (1, 1), activation=tf.nn.relu)

    conv3x3_relu = lambda x, num_f: tf.layers.conv2d(x, num_f, (3, 3), activation=tf.nn.relu)
    upconv2x2 = lambda x, num_f: tf.layers.conv2d_transpose(x, num_f, (2, 2), (2, 2))
    # downsampling
    h = x  # 572

    h = conv3x3_relu(h, 64)  # 570
    h = conv3x3_relu(h, 64)  # 568
    out_568 = h
    h = tf.layers.max_pooling2d(h, (2, 2), (2, 2))  # 284
    h = conv3x3_relu(h, 128)  # 282
    h = conv3x3_relu(h, 128)  # 280
    out_280 = h
    h = tf.layers.max_pooling2d(h, (2, 2), (2, 2))  # 140
    h = conv3x3_relu(h, 256)  # 138
    h = conv3x3_relu(h, 256)  # 136
    out_136 = h
    h = tf.layers.max_pooling2d(h, (2, 2), (2, 2))  # 68
    h = conv3x3_relu(h, 512)  # 66
    h = conv3x3_relu(h, 512)  # 64
    out_64 = h
    h = tf.layers.max_pooling2d(h, (2, 2), (2, 2))  # 32
    h = conv3x3_relu(h, 1024)  # 30
    h = conv3x3_relu(h, 1024)  # 28

    # upsampling
    h = upconv2x2(h, 512)  # 56
    h = tf.concat((h, out_64[:, 4:-4, 4:-4, :]), 3)
    h = conv3x3_relu(h, 512)  # 54
    h = conv3x3_relu(h, 512)  # 52

    h = upconv2x2(h, 256)  # 104
    h = tf.concat((h, out_136[:, 16:-16, 16:-16, :]), 3)
    h = conv3x3_relu(h, 256)  # 102
    h = conv3x3_relu(h, 256)  # 100

    h = upconv2x2(h, 128)  # 200
    h = tf.concat((h, out_280[:, 40:-40, 40:-40, :]), 3)
    h = conv3x3_relu(h, 128)  # 198
    h = conv3x3_relu(h, 128)  # 196

    h = upconv2x2(h, 64)  # 392
    h = tf.concat((h, out_568[:, 88:-88, 88:-88, :]), 3)
    h = conv3x3_relu(h, 64)  # 390
    h = conv3x3_relu(h, 64)  # 388

    h = upconv2x2(h, 64)  # 392
    h = tf.layers.conv2d(h, 32, (3, 3), activation=tf.nn.relu)

    # final layer
    y_pred = tf.layers.conv2d(h, 1, (203, 203))

    # Construct model
    loss = (tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y,
        logits=y_pred))

    # Minimize error using cross entropy
    optim = tf.train.AdamOptimizer(1e-2)
    step = optim.minimize(loss)

    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    return step, y_pred, session


def runTensorFlow(eyesToTrain, eyesToCalculate, batch_size, learning_rate, training_epochs, patch_size, verbose):
    if verbose:
        print("Training data:", end='')
        print_info_about_images(eyesToTrain)
        print("\nProcess data:", end='')
        print_info_about_images(eyesToCalculate)
        print("\nMachine learning paramteres")
        print("Learning rate:\t\t" + str(learning_rate))
        print("Traning epochs:\t\t" + str(training_epochs))

    # Todo: Generalize
    batches_xs = eyesToTrain.get('h')[0].get_batches_of_raw()
    shape_x = batches_xs[0].shape[1]
    shape_y = batches_xs[0].shape[0]
    batches_ys = eyesToTrain.get('h')[0].get_batches_of_calculated()
    step, y_pred, session = setup_model(len(batches_xs), shape_y, shape_x, patch_size)

    # Parameters (with batch_size, learning_rate and training_epochs)
    display_step = 1

    # # Training cycle
    # Todo: Process other than only healthy
    for i_eye in range(len(eyesToTrain.get('h'))):
        # Plot settings
        avg_set = []
        epoch_set = []
        print("\nTraining on " + str(i_eye + 1) + " image")
        eye = eyesToTrain.get('h')[i_eye]
        eye.plot_raw(extraStr=str(i_eye + 1) + " traning")
        eye.plot_manual(extraStr=str(i_eye + 1) + " traning")
        avg_cost = 0.
        for epoch in range(training_epochs):
            # Loop over all patches
            prevProgress = "0.00%"
            avg_cost += session.run(loss, feed_dict={x: batches_xs, y: batches_ys})
            batches_xs = eye.get_batches_of_raw()
            batches_xs = normalize(np.array(batches_xs).reshape(1, -1), norm='max').reshape(len(batches_xs), shape_x, shape_y, 3)
            batches_ys = eye.get_batches_of_calculated()
            batches_ys = normalize(np.array(batches_ys).reshape(1, -1), norm='max').reshape(len(batches_ys), shape_x, shape_y, 1)
            # Fit training using batch data
            session.run(step, feed_dict={x: batches_xs, y: batches_ys})
            # Display logs per epoch step
            if epoch % display_step == 0:
                print(
                    "\rEpoch: " + '%04d' % (epoch + 1) + "\t\t100.00%" + "\t\tcost = " + "{:.9f}".format(np.average(avg_cost)),
                    flush=True)
            epoch_set.append(epoch + 1)
            avg_set.append(np.average(avg_cost))
        print("Training phase finished for " + str(i_eye + 1) + " eye")

        plt.plot(epoch_set, avg_set, 'o', label='Logistic Regression Training phase - ' + str(i_eye + 1) + ' eye')
        plt.ylabel('cost')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

    for i_eye in range(len(eyesToCalculate.get('h'))):
        eye = eyesToCalculate.get('h')[i_eye]
        eye.plot_raw(extraStr=str(i_eye + 1) + " processing")
        eye.plot_manual(extraStr=str(i_eye + 1) + " processing")
        eye.plot_calculated(extraStr=str(i_eye + 1) + " processing")

        feed_dict = {x: eye.get_batches_of_raw()}
        # Gets only the middle point
        classification = session.run(y_pred, feed_dict=feed_dict)
        print('\r' + "Calculated " + str(i_eye + 1)
              + " eye:\t\t\t\t\t\t\t100.00%", flush=True)
        print("Building an image based on predictions...")

        for p in range(5):
            i = random.randint(0, len(classification) - 1)
            eye.plot_image(eye.get_batches_of_manual()[i], "Random calculated batch #" + str(p))
            eye.plot_image(classification[i, :, :, 0], "Random predicted batch #" + str(p))

        batches_of_manual = eye.get_batches_of_manual()
        eye.build_image_from_batches(np.array(batches_of_manual))
        eye.plot_calculated(extraStr=str(i_eye + 1) + " manual")
        eye.build_image_from_batches(classification.reshape(classification.shape[:-1]))
        eye.plot_calculated(extraStr=str(i_eye + 1) + " calculated")
        print("Difference between manual and predicted:\t\t"
              + "{:.2f}".format(eye.compare() * 100) + "%")

        if verbose:
            path = os.path.join(tempfile.gettempdir(), "tensorflowlogs")
            # tf.summary.FileWriter(path, session.graph)
            print("\nTo use TensorBoard run:\n$tensorboard", "--logdir=" + path + "\n")


def print_info_about_images(eyes):
    eyes_h = eyes.get("h")
    eyes_g = eyes.get("g")
    eyes_d = eyes.get("d")
    print_info_about_image(eyes_h, "healthy")
    print_info_about_image(eyes_g, "glaucomatous")
    print_info_about_image(eyes_d, "diabetic")


def print_info_about_image(eyes, name):
    if len(eyes) > 0:
        print("\nShape of " + name + " images:")
        print("\t\tRaw:\t\t\t\t\tManual:\t\t\t\t\tMask:")
        for i in range(0, len(eyes)):
            print(str(i + 1), end="\t\t")
            print(eyes[i].get_raw().shape, end="\t\t\t")
            print(eyes[i].get_manual().shape, end="\t\t\t")
            print(eyes[i].get_mask().shape, end="\n")
