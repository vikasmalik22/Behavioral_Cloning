import csv
import cv2
import numpy as np
import skimage.transform as sktransform
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Lambda, Convolution2D, Dropout
from keras.layers import Cropping2D
from keras.regularizers import l2
from keras.optimizers import Adam
import pandas as pd
import random

samples = []

def Read_Data():
    for i in range(1):
        with open('data/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for line in reader:
                samples.append(line)


def brightness_adjustment(image, bval=None):
    # convert to HSV so that its easy to adjust brightness
    new_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    if bval:
        random_bright = bval
    else:
        random_bright = .25 + np.random.uniform()

    # Apply the brightness reduction to the V channel
    new_img[:, :, 2] = new_img[:, :, 2] * random_bright

    # convert to RGB again
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)

    return new_img

def flip_image(image, steering_angle):
    image = cv2.flip(image,1)
    steering_angle = -1 * steering_angle
    return image, steering_angle


def shift_image(image, steer_ang, shift_range):
    # Translation
    rows, cols = image.shape[0:2]
    tr_x = shift_range * np.random.uniform() - shift_range / 2
    steer_ang = steer_ang + tr_x / shift_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, M, (cols, rows))

    return image_tr, steer_ang


def random_shadow(img):

    shadow_factor = 0.25
    image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_width = image.shape[1]
    img_height = image.shape[0]
    x = random.randint(0, img_width)
    y = random.randint(0, img_height)

    width = random.randint(int(img_width / 2), img_width)
    if (x + width > img_width):
        x = img_width - x

    height = random.randint(int(img_height / 2), img_height)
    if (y + height > img_height):
        y = img_height - y

    image[y:y + height, x:x + width, 2] = image[y:y + height, x:x + width, 2] * shadow_factor
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

def PreProcess_Data(image, steering_angle):
    # crop image
    image = image[65:(image.shape[0] - 25), 0:image.shape[1]]
    image = brightness_adjustment(image)
    image, steer_angle = shift_image(image, steering_angle, 100)

    if (random.random() <= 0.5):
        image = random_shadow(image)

    if (random.random() <= 0.7):
        image, steer_angle = flip_image(image, steer_angle)

    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    return image, steer_angle

# def process_image(image):
#     '''
#     :param image: Input image
#     :return: output image with randomly adjusted brightness
#     '''
#
#     # convert to HSV so that its easy to adjust brightness
#     new_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#
#     # randomly generate the brightness reduction factor
#     # Add a constant so that it prevents the image from being completely dark
#     random_bright = .25+np.random.uniform()
#
#     # Apply the brightness reduction to the V channel
#     new_img[:,:,2] = new_img[:,:,2]*random_bright
#
#     # convert to RGB again
#     new_img = cv2.cvtColor(new_img,cv2.COLOR_HSV2RGB)
#
#     # random shadow - full height, random left/right side, random darkening
#     h, w = new_img.shape[0:2]
#     mid = np.random.randint(0, w)
#     factor = np.random.uniform(0.6, 0.8)
#     if np.random.rand() > .5:
#         new_img[:, 0:mid, 0] = (new_img[:, 0:mid, 0] * factor).astype(np.int32)
#     else:
#         new_img[:, mid:w, 0] = (new_img[:, mid:w, 0] * factor).astype(np.int32)
#
#     # randomly shift horizon
#     h, w, _ = new_img.shape
#     horizon = 2 * h / 5
#     v_shift = np.random.randint(-h / 8, h / 8)
#     pts1 = np.float32([[0, horizon], [w, horizon], [0, h], [w, h]])
#     pts2 = np.float32([[0, horizon + v_shift], [w, horizon + v_shift], [0, h], [w, h]])
#     M = cv2.getPerspectiveTransform(pts1, pts2)
#     new_img = cv2.warpPerspective(new_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
#     return new_img.astype(np.uint8)

def generator(samples, batch_size=32):
    num_samples = samples.shape[0]
    threshold = 1
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steering_angles = []
            for batch_sample in batch_samples:
                # filename = batch_sample[i].split('\\')[-1]
                # current_path = batch_sample[i].split('\\')[-2] + '/' + filename
                # filename = batch_sample[i].split('/')[-1]
                # current_path = 'data/IMG/' + filename
                # image = cv2.imread(current_path)
                correction = 0.25  # this is a parameter to tune
                steering_center = float(batch_sample[3])

                # idea borrowed from Vivek Yadav: Sample images such that images with lower angles
                # have lower probability of getting represented in the dataset. This alleviates
                # any problems we may ecounter due to model having a bias towards driving straight.

                # keep = 0
                # while keep == 0:
                #     if abs(steering_center) < .1:
                #         val = np.random.uniform()
                #         if val > threshold:
                #             keep = 1
                #     else:
                #         keep = 1
                #
                # if keep:
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                img_center = cv2.imread('data/IMG/' + batch_sample[0].split('/')[-1])
                img_left = cv2.imread('data/IMG/' + batch_sample[1].split('/')[-1])
                img_right = cv2.imread('data/IMG/' + batch_sample[2].split('/')[-1])

                img_center, steering_center = PreProcess_Data(img_center, steering_center)
                img_left, steering_left = PreProcess_Data(img_left, steering_left)
                img_right, steering_right = PreProcess_Data(img_right, steering_right)

                images.append(img_center)
                steering_angles.append(steering_center)

                images.append(img_left)
                steering_angles.append(steering_left)

                images.append(img_right)
                steering_angles.append(steering_right)

            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Step 1
Read_Data()
print(len(samples))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def NVIDIA_Model():
    model = Sequential()
    #1st Layer - Add a flatten layer)
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64,64,3)))
    model.add(Convolution2D(24,5,5,border_mode='valid', activation='elu', subsample=(2,2), W_regularizer=l2(0.0001), init='normal'))
    model.add(Convolution2D(36,5,5,border_mode='valid', activation='elu', subsample=(2,2), W_regularizer=l2(0.0001), init='normal'))
    model.add(Convolution2D(48,5,5,border_mode='valid', activation='elu', subsample=(2,2), W_regularizer=l2(0.0001), init='normal'))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='elu', subsample=(1,1), W_regularizer=l2(0.0001), init='normal'))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='elu', subsample=(1,1), W_regularizer=l2(0.0001), init='normal'))
    model.add(Flatten())
    model.add(Dense(1164, activation='elu', W_regularizer=l2(0.0001), init='normal'))
    model.add(Dense(100, activation='elu', W_regularizer=l2(0.0001), init='normal'))
    model.add(Dense(50, activation='elu', W_regularizer=l2(0.0001), init='normal'))
    model.add(Dense(10, activation='elu', W_regularizer=l2(0.0001), init='normal'))
    model.add(Dense(1, W_regularizer=l2(0.0001), init='normal'))
    return model

nvidia_model = NVIDIA_Model()
nvidia_model.compile(loss='mse', optimizer=Adam(lr=0.0001))
train_samples = np.array(train_samples)
validation_samples = np.array(validation_samples)

print(train_samples.shape)
print(validation_samples.shape)

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

history = nvidia_model.fit_generator(train_generator, samples_per_epoch=3*train_samples.shape[0],
                           validation_data=validation_generator,
                           nb_val_samples=3*validation_samples.shape[0], nb_epoch=30)

nvidia_model.save('model.h5')
print(nvidia_model.summary())
