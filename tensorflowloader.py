import random

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
from sklearn.preprocessing import normalize

global x, y


def setup_model(batches, x_patch_size, y_patch_size):
    global x, y
    # tf Graph Input
    x = tf.placeholder(tf.float32, (batches, x_patch_size, x_patch_size, 3))
    # 2 possibilities to place 2 colors on the center pixel
    y = tf.placeholder(tf.float32, (batches, y_patch_size, y_patch_size, 1))

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

    # final layer
    y_pred = tf.layers.conv2d(h, 1, (1, 1))

    # Minimize error using cross entropy
    optim = tf.train.AdamOptimizer(learning_rate=0.0001)
    cost = tf.reduce_mean(tf.square(y_pred - y))
    cost = optim.minimize(cost)

    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    return cost, y_pred, session


def runTensorFlow(eyesToTrain, eyesToCalculate, batch_size, learning_rate, training_epochs, x_patch_size, y_patch_size, verbose):
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
    batches_ys = eyesToTrain.get('h')[0].get_batches_of_calculated()
    cost, y_pred, session = setup_model(len(batches_xs), x_patch_size, y_patch_size)
    # Parameters (with batch_size, learning_rate and training_epochs)
    display_step = 1

    # # Training cycle
    # Todo: Process other than only healthy
    avg_set = []
    epoch_set = []
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Plot settings
        for i_eye in range(len(eyesToTrain.get('h'))):
            print("\nTraining on " + str(i_eye + 1) + " image")
            eye = eyesToTrain.get('h')[i_eye]
            eye.plot_raw(extraStr=str(i_eye + 1) + " traning")
            eye.plot_manual(extraStr=str(i_eye + 1) + " traning")
            # Loop over all patches
            cost_value = session.run(cost, feed_dict={x: batches_xs, y: batches_ys})
            avg_cost += 0.
            batches_xs = eye.get_batches_of_raw()
            batches_xs = normalize(np.array(batches_xs).reshape(1, -1), norm='max').reshape(len(batches_xs), x_patch_size, x_patch_size, 3)
            batches_ys = eye.get_batches_of_manual()
            batches_ys = normalize(np.array(batches_ys).reshape(1, -1), norm='max').reshape(len(batches_ys), y_patch_size, y_patch_size, 1)
            # Fit training using batch data
            print("Training phase finished for " + str(i_eye + 1) + " eye")
        # Display logs per epoch step
        if epoch % display_step == 0:
            print(
                "\rEpoch: " + '%04d' % (epoch + 1) + "\t\t100.00%" + "\t\tcost = " + "{:.9f}".format(np.average(avg_cost)),
                flush=True)

        epoch_set.append(epoch + 1)
        avg_set.append(np.average(avg_cost))
    plt.plot(epoch_set, avg_set, 'o', label='Logistic Regression Training phase')
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
        classification = normalize(np.array(classification).reshape(1, -1), norm='max').reshape(len(batches_ys), y_patch_size, y_patch_size, 1)
        print('\r' + "Calculated " + str(i_eye + 1)
              + " eye:\t\t\t\t\t\t\t100.00%", flush=True)
        print("Building an image based on predictions...")

        for p in range(5):
            i = random.randint(0, len(classification) - 1)
            eye.plot_image(eye.get_batches_of_manual()[i], "Random calculated batch #" + str(p))
            eye.plot_image(classification[i, :, :, 0], "Random predicted batch #" + str(p))
            bin_image = eye.convert_to_binary_image(classification[i, :, :])
            text = "Random predicted binary batch #" + str(p) + " mean"
            eye.plot_image(bin_image, text)
            print(text)
            eye.compare(eye.get_batches_of_manual()[i], bin_image)
            for threshold in range(9):
                threshold = threshold / 10.0
                bin_image = eye.convert_to_binary_image(classification[i, :, :], threshold, True)
                text = "Random predicted binary batch #" + str(p) + " threshold " + str(threshold)
                eye.plot_image(bin_image, text)
                print(text)
                eye.compare(eye.get_batches_of_manual()[i], bin_image)

        # batches_of_manual = eye.get_batches_of_manual()
        # eye.build_image_from_batches(np.array(batches_of_manual))
        # eye.plot_calculated(extraStr=str(i_eye + 1) + " manual")
        # eye.build_image_from_batches(classification.reshape(classification.shape[:-1]))
        # eye.plot_calculated(extraStr=str(i_eye + 1) + " calculated")
        # eye.plot_calculated(extraStr=str(i_eye + 1) + " calculated", binary=True)

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
