# Commented out IPython magic to ensure Python compatibility.
from __future__ import print_function

# %tensorflow_version 2.x
import tensorflow as tf
import matplotlib.pyplot as plt
import SimpleITK as sitk
import glob
import os
import numpy as np
import time
import cv2
import pywt
from data_generator import DataGenerator
from sklearn import preprocessing
from random import randint
from random import shuffle
from random import seed
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import Dropout, MaxPool2D, LeakyReLU
from keras.initializers import Constant
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras import metrics


def preprocess_data(src, dest='/home/apostolos/PycharmProjects/TestEnv/data/training/'):
    """Function to perform preprocessing and store the resulting 2D images in a new folder

        Inputs:
                src: Path for the files to be processed
                dest: Destination path for the processed files to be stored

        Output:
                Creation of a folder named training_data  containing numpy array images
    """
    # Get information for every patient
    files = glob.glob(src + '/*/', recursive=True)
    shuffle(files)
    files = files[0:15]

    sample_idx = 0

    # For every patient folder
    for file in files:
        os.chdir(file)
        modalities = glob.glob('*.mha')

        reader = sitk.ImageFileReader()
        reader.SetImageIO('MetaImageIO')

        images3D = {}

        for mod in modalities:
            reader.SetFileName(mod)
            image3D = reader.Execute()
            if 'T1' in mod:
                if 'T1c' in mod:
                    images3D.update({'T1c': image3D})
                else:
                    images3D.update({'T1': image3D})
            elif 'T2' in mod:
                images3D.update({'T2': image3D})
            elif 'Flair' in mod:
                images3D.update({'Flair': image3D})
            elif 'OT' in mod:
                images3D.update({'GT': image3D})

        # For every slice
        for slice_idx in range(50, 130):
            imgs = []
            print(sample_idx)

            # T1 modality information
            t1_mod = images3D['T1'][:, :, slice_idx]
            t1_mod = n4itk_bias_correction(t1_mod)
            np_t1 = sitk.GetArrayFromImage(t1_mod)
            np_t1 = preprocessing.normalize(np_t1)
            imgs.append(np_t1)

            # T1c modality information
            t1c_mod = images3D['T1c'][:, :, slice_idx]
            t1c_mod = n4itk_bias_correction(t1c_mod)
            np_t1c = sitk.GetArrayFromImage(t1c_mod)
            np_t1c = preprocessing.normalize(np_t1c)
            imgs.append(np_t1c)

            # T2 modality information
            t2_mod = images3D['T2'][:, :, slice_idx]
            np_t2 = sitk.GetArrayFromImage(t2_mod)
            np_t2 = preprocessing.normalize(np_t2)
            imgs.append(np_t2)

            # Flair modality information
            flair_mod = images3D['Flair'][:, :, slice_idx]
            np_flair = sitk.GetArrayFromImage(flair_mod)
            np_flair = preprocessing.normalize(np_flair)
            imgs.append(np_flair)

            # Ground truth information
            gt_mod = images3D['GT'][:, :, slice_idx]
            np_gt = sitk.GetArrayFromImage(gt_mod)
            imgs.append(np_gt)

            # Store the result
            name = dest + 'sample_' + str(sample_idx)
            np.save(name, np.array(imgs).astype('float32'))
            sample_idx = sample_idx + 1


def n4itk_bias_correction(img):
    """Function to implement the n4itk bias correction algorithm
            
        Input: 
                img: SimpleITK image to be processed
        
        Output: 
                out_img: Corrected SimpleITK image
    """
    begin = time.time()
    mask_img = sitk.OtsuThreshold(img, 0, 1, 200)
    img = sitk.Cast(img, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    out_img = corrector.Execute(img, mask_img)
    end = time.time()

    print('Elapsed time in sec: ', end - begin)

    return out_img


def resolution_enhancement(input_img, alpha=4):
    """Function to increase the resolution of the input image
    
        Input:
                img: Numpy array image to be processed
                
        Output:
                out_img: Enhanced Numpy array image
    """

    # Get the swt coefficients and the corresponding subband images
    shape = input_img.shape
    print(shape)
    out_img = np.zeros((alpha * shape[0], alpha * shape[1], shape[2]))

    for i in range(shape[2]):
        img = input_img[:, :, i]
        swt_coeffs2 = pywt.swt2(img, 'db9', level=1)
        swt_LL, (swt_LH, swt_HL, swt_HH) = swt_coeffs2[0]

        enh_est_LL = cv2.resize(img, None, fx=alpha, fy=alpha, interpolation=cv2.INTER_CUBIC)
        enh_est_LH = cv2.resize(swt_LH, None, fx=alpha, fy=alpha, interpolation=cv2.INTER_CUBIC)
        enh_est_HL = cv2.resize(swt_HL, None, fx=alpha, fy=alpha, interpolation=cv2.INTER_CUBIC)
        enh_est_HH = cv2.resize(swt_HH, None, fx=alpha, fy=alpha, interpolation=cv2.INTER_CUBIC)

        coeffs = enh_est_LL, (enh_est_LH, enh_est_HL, enh_est_HH)

        out_img[:, :, i] = pywt.iswt2([coeffs], 'db9')

    return out_img


def load_data(data_path):
    """Function to load the data form the path """

    samples = glob.glob(data_path + '*.npy')
    x_train = []
    y_train = []

    iter = 0
    # For every npy file in the path
    for spl in samples:
        print(iter)
        iter = iter + 1
        img = np.load(spl)
        # img = normalize(img)
        (x_data, y_data) = patch_extraction(img, num_patches=12, dim=33)

        for k in range(len(y_data)):
            x_train.append(np.asarray(x_data)[k, :, :])
            y_train.append(np.asarray(y_data)[k])

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    return x_train, y_train


def normalize(img):
    """Function to perform intensity normalization"""

    width = img.shape[0]
    height = img.shape[1]

    out_img = np.zeros(img.shape)

    max_val = np.max(img)
    min_val = np.min(img)

    for x_idx in range(0, width):
        for y_idx in range(0, height):
            out_img[x_idx, y_idx] = 1 / (max_val - min_val) * img[x_idx, y_idx] - 1 * min_val / (
                    max_val - min_val)

    return out_img


def patch_extraction(img, dim=33, num_patches=256):
    """Function to perform random patch extraction from an image"""

    x_data = []
    y_data = []
    # Get the dimensions of the image
    height = img.shape[1]
    width = img.shape[2]

    class_1 = np.argwhere(img[4, :, :] == 1)
    np.random.shuffle(class_1)
    class_2 = np.argwhere(img[4, :, :] == 2)
    np.random.shuffle(class_2)
    class_3 = np.argwhere(img[4, :, :] == 3)
    np.random.shuffle(class_3)
    class_4 = np.argwhere(img[4, :, :] == 4)
    np.random.shuffle(class_4)

    num_can_patches = int(num_patches / 4)

    c_cells = np.argwhere(img[4, :, :] != 0)
    np.random.shuffle(c_cells)

    # Normal cell locations
    n_cells = np.argwhere(img[4, :, :] == 0)
    back = []

    for k in range(n_cells.shape[0]):
        x_idx = n_cells[k, 0]
        y_idx = n_cells[k, 1]
        if img[0, x_idx, y_idx] == 0:
            back.append(k)
    n_cells = np.delete(n_cells, back, axis=0)
    # Shuffle the indices randomly
    np.random.shuffle(n_cells)

    # Normal tissue patches
    for i in range(num_patches):

        x_idx = n_cells[i][0]
        y_idx = n_cells[i][1]

        if (x_idx < dim / 2) or (height - x_idx < dim / 2):
            continue

        if (y_idx < dim / 2) or (width - y_idx < dim / 2):
            continue

        patch = np.copy(img[0:4, x_idx - int((dim - 1) / 2):x_idx + int((dim - 1) / 2 + 1),
                        y_idx - int((dim - 1) / 2):y_idx + int((dim - 1) / 2 + 1)])

        if np.count_nonzero(patch[0, :, :]) < 0.25 * dim ** 2:
            continue

        patch = np.transpose(patch, (1, 2, 0))
        for j in range(4):
            patch[:, :, j] = patch[:, :, j] - np.mean(patch[:, :, j])

        x_data.append(patch)

        label = np.zeros(5)
        label[int(img[4, x_idx, y_idx])] = 1
        y_data.append(label)

    # Cancerous tissue patches
    for i in range(min(num_patches, len(c_cells))):

        x_idx = c_cells[i][0]
        y_idx = c_cells[i][1]

        if (x_idx < dim / 2) or (height - x_idx < dim / 2):
            continue

        if (y_idx < dim / 2) or (width - y_idx < dim / 2):
            continue

        patch = np.copy(img[0:4, x_idx - int((dim - 1) / 2):x_idx + int((dim - 1) / 2 + 1),
                        y_idx - int((dim - 1) / 2):y_idx + int((dim - 1) / 2 + 1)])

        if np.count_nonzero(patch[0, :, :]) < 0.25 * dim ** 2:
            continue

        patch = np.transpose(patch, (1, 2, 0))
        for j in range(4):
            patch[:, :, j] = patch[:, :, j] - np.mean(patch[:, :, j])

        x_data.append(patch)

        label = np.zeros(5)
        label[int(img[4, x_idx, y_idx])] = 1
        y_data.append(label)

    return (x_data, y_data)


def local_net():
    """Function to define the CNN architecture"""

    bias_init = Constant(value=0.1)
    act_func = LeakyReLU(alpha=0.333)
    act_func.__name__ = 'relu'

    # create model
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation=act_func, border_mode='same', input_shape=(33, 33, 4),
                     kernel_initializer='glorot_normal', bias_initializer=bias_init))
    model.add(Conv2D(64, kernel_size=3, activation=act_func, border_mode='same', kernel_initializer='glorot_normal',
                     bias_initializer=bias_init))
    model.add(Conv2D(64, kernel_size=3, activation=act_func, border_mode='same', kernel_initializer='glorot_normal',
                     bias_initializer=bias_init))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(128, kernel_size=3, activation=act_func, border_mode='same', kernel_initializer='glorot_normal',
                     bias_initializer=bias_init))
    model.add(Conv2D(128, kernel_size=3, activation=act_func, border_mode='same', kernel_initializer='glorot_normal',
                     bias_initializer=bias_init))
    model.add(Conv2D(128, kernel_size=3, activation=act_func, border_mode='same', kernel_initializer='glorot_normal',
                     bias_initializer=bias_init))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(units=256, activation=act_func, kernel_initializer='glorot_normal', bias_initializer=bias_init))
    model.add(Dropout(0.1))
    model.add(Dense(units=256, activation=act_func, kernel_initializer='glorot_normal', bias_initializer=bias_init))
    model.add(Dropout(0.1))
    model.add(Dense(units=5, activation='softmax', kernel_initializer='glorot_normal', bias_initializer=bias_init))

    epochs = 25
    learning_rate = 0.003
    decay_rate = learning_rate / epochs
    momentum = 0.9
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

    return model


def neural_net():
    """Function to define the CNN architecture"""

    bias_init = Constant(value=0.1)
    act_func = 'relu'

    # create model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=7, activation=act_func, border_mode='same', input_shape=(53, 53, 4),
                     kernel_initializer='glorot_normal', bias_initializer=bias_init))
    model.add(Conv2D(32, kernel_size=7, activation=act_func, border_mode='same', kernel_initializer='glorot_normal',
                     bias_initializer=bias_init))
    model.add(MaxPool2D(pool_size=(7, 7), strides=(2, 2)))

    model.add(Conv2D(64, kernel_size=7, activation=act_func, border_mode='same', kernel_initializer='glorot_normal',
                     bias_initializer=bias_init))
    model.add(Conv2D(64, kernel_size=7, activation=act_func, border_mode='same', kernel_initializer='glorot_normal',
                     bias_initializer=bias_init))
    model.add(MaxPool2D(pool_size=(7, 7), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(units=256, activation=act_func, kernel_initializer='glorot_normal', bias_initializer=bias_init))
    model.add(Dropout(0.1))
    model.add(Dense(units=256, activation=act_func, kernel_initializer='glorot_normal', bias_initializer=bias_init))
    model.add(Dropout(0.1))
    model.add(Dense(units=5, activation='softmax', kernel_initializer='glorot_normal', bias_initializer=bias_init))

    epochs = 25
    learning_rate = 0.003
    decay_rate = learning_rate / epochs
    momentum = 0.9
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

    return model


def display(img_):
    """Function to display the T2 modality of an MRI image with ground truth cancer location"""

    max_val = np.max(img_[1, :, :])
    min_val = np.min(img_[1, :, :])

    width = img_.shape[1]
    height = img_.shape[2]
    out_img = np.zeros((width, height, 3), dtype=int)

    # Normalize the input image
    for x_idx in range(0, width):
        for y_idx in range(0, height):
            img_[1, x_idx, y_idx] = 255 / (max_val - min_val) * img_[1, x_idx, y_idx] - 255 * min_val / (
                    max_val - min_val)

            if img_[4, x_idx, y_idx] == 0:
                out_img[x_idx, y_idx, 0] = int(img_[1, x_idx, y_idx])
                out_img[x_idx, y_idx, 1] = int(img_[1, x_idx, y_idx])
                out_img[x_idx, y_idx, 2] = int(img_[1, x_idx, y_idx])
            elif img_[4, x_idx, y_idx] == 1:
                out_img[x_idx, y_idx, 0] = 255
                out_img[x_idx, y_idx, 1] = 0
                out_img[x_idx, y_idx, 2] = 0
            elif img_[4, x_idx, y_idx] == 2:
                out_img[x_idx, y_idx, 0] = 0
                out_img[x_idx, y_idx, 1] = 255
                out_img[x_idx, y_idx, 2] = 0
            elif img_[4, x_idx, y_idx] == 3:
                out_img[x_idx, y_idx, 0] = 0
                out_img[x_idx, y_idx, 1] = 0
                out_img[x_idx, y_idx, 2] = 255
            else:
                out_img[x_idx, y_idx, 0] = 255
                out_img[x_idx, y_idx, 1] = 255
                out_img[x_idx, y_idx, 2] = 0

    plt.figure()
    plt.imshow(out_img)
    plt.show()


def predict(model_, img_, dim=33):
    width = img_.shape[1]
    height = img_.shape[2]

    out_img = np.zeros((width, height))
    n_cells = np.argwhere(img_[0, :, :] != 0)

    for idx in n_cells:
        x_idx = idx[0]
        y_idx = idx[1]

        x_new = img_[0:4, x_idx - int((dim - 1) / 2):x_idx + int((dim - 1) / 2 + 1),
                y_idx - int((dim - 1) / 2):y_idx + int((dim - 1) / 2 + 1)]

        for j in range(4):
            x_new[j, :, :] = x_new[j, :, :] - np.mean(x_new[j, :, :])

        x_new = x_new.transpose((1, 2, 0))
        print(np.max(x_new))
        print(np.min(x_new))
        y_new = model_.predict(np.array([x_new, ]))

        pred = np.argmax(y_new)
        print('Predicted value = ', pred)
        print('Ground Truth = ', int(img_[4, x_idx, y_idx]))

        out_img[x_idx, y_idx] = pred

    return out_img


def train_model():
    # Load the data
    (x_train, y_train) = load_data('/home/apostolos/PycharmProjects/TestEnv/data/training/')
    (x_val, y_val) = load_data('/home/apostolos/PycharmProjects/TestEnv/data/testing/')

    # Load the model
    model = local_net()
    print('# Fit the model on training data')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('/home/apostolos/PycharmProjects/TestEnv/best_model.h5', monitor='val_loss', mode='min',
                         verbose=1, save_best_only=True)

    # Train the model
    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=25,
                        # We pass some validation for
                        # monitoring validation loss and metrics
                        # at the end of each epoch
                        validation_data=(x_val, y_val),
                        verbose=1,
                        callbacks=[es, mc])
    print('\nhistory dict:', history.history)
    saved_model = load_model('/home/apostolos/PycharmProjects/TestEnv/best_model.h5')

    _, train_acc = saved_model.evaluate(x_train, y_train, verbose=0)
    _, val_acc = saved_model.evaluate(x_val, y_val, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, val_acc))


if __name__ == '__main__':
    train_model()

    """
    model = load_model('/home/apostolos/PycharmProjects/TestEnv/best_model.h5')
    img = np.load('/home/apostolos/PycharmProjects/TestEnv/data/training/sample_830.npy')
    # display(img)
    img_1 = np.load('/home/apostolos/PycharmProjects/TestEnv/data/training/sample_830.npy')
    out_img = predict(model, img)
    predicted = np.zeros(img.shape)
    predicted[0:4, :, :] = img_1[0:4, :, :]
    predicted[4, :, :] = out_img
    display(predicted)
    """
