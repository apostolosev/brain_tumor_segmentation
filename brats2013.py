from __future__ import print_function

import matplotlib.pyplot as plt
import SimpleITK as sitk
import glob
import os
from sys import getsizeof
import numpy as np
import time
import cv2
import pywt
from random import randint
from random import seed
import tensorflow
import medpy


def preprocess_data(src, dest='/home/apostolos/PycharmProjects/TestEnv/data/training_HGG/'):
    """Function to perform preprocessing and store the resulting 2D images in a new folder

        Inputs:
                src: Path for the files to be processed
                dest: Destination path for the processed files to be stored

        Output:
                Creation of a folder named training_data  containing numpy array images
    """
    # Get information for every patient
    files = glob.glob(src + '**/*2013*', recursive=True)

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

            # T1 modality information
            t1_mod = images3D['T1'][:, :, slice_idx]
            t1_mod_out = n4itk_bias_correction(t1_mod)
            np_t1 = sitk.GetArrayFromImage(t1_mod_out)
            imgs.append(np_t1)

            # T1c modality information
            t1c_mod = images3D['T1c'][:, :, slice_idx]
            t1c_mod_out = n4itk_bias_correction(t1c_mod)
            np_t1c = sitk.GetArrayFromImage(t1c_mod_out)
            imgs.append(np_t1c)

            # T2 modality information
            t2_mod = images3D['T2'][:, :, slice_idx]
            np_t2 = sitk.GetArrayFromImage(t2_mod)
            imgs.append(np_t2)

            # Flair modality information
            flair_mod = images3D['Flair'][:, :, slice_idx]
            np_flair = sitk.GetArrayFromImage(flair_mod)
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
    swt_coeffs2 = pywt.swt2(input_img, 'db9', level=1)
    swt_LL, (swt_LH, swt_HL, swt_HH) = swt_coeffs2[0]

    enh_est_LL = cv2.resize(input_img, None, fx=alpha, fy=alpha, interpolation=cv2.INTER_CUBIC)
    enh_est_LH = cv2.resize(swt_LH, None, fx=alpha, fy=alpha, interpolation=cv2.INTER_CUBIC)
    enh_est_HL = cv2.resize(swt_HL, None, fx=alpha, fy=alpha, interpolation=cv2.INTER_CUBIC)
    enh_est_HH = cv2.resize(swt_HH, None, fx=alpha, fy=alpha, interpolation=cv2.INTER_CUBIC)

    coeffs = enh_est_LL, (enh_est_LH, enh_est_HL, enh_est_HH)

    out_img = pywt.iswt2([coeffs], 'db9')
    bicubic_enh = cv2.resize(input_img, None, fx=alpha, fy=alpha, interpolation=cv2.INTER_CUBIC)

    plt.figure()
    plt.imshow(input_img, cmap='gray')
    plt.title('Original image')

    plt.figure()
    plt.imshow(out_img, cmap='gray')
    plt.title('Enhanced image')

    plt.figure()
    plt.imshow(bicubic_enh, cmap='gray')
    plt.title('Enhanced image using bicubic interpolation')

    plt.show()

    return out_img


def load_data(data_path):
    """Function to load the data form the path """

    samples = glob.glob(data_path + '*.npy')
    x_train = []
    y_train = []

    iter = 0
    # For every npy file in the path
    for spl in samples:
        iter = iter + 1
        img = np.load(spl)
        (x_data, y_data) = patch_extraction(img)

        for k in range(len(y_data)):
            x_train.append(np.asarray(x_data)[k, :, :])
            y_train.append(np.asarray(y_data)[k])

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    return x_train, y_train


def patch_extraction(img, dim=33):
    """Function to perform random patch extraction from an image"""

    # Get the dimensions of the image
    height = img.shape[1]
    width = img.shape[2]

    # Generate random numbers
    seed(time.time())
    x_data = []
    y_data = []
    max_iter = 20
    n_iter = 0

    # Get normal tissue cells by sampling randomlly
    while len(x_data) < 5 and n_iter < max_iter:

        n_iter = n_iter + 1
        x_idx = randint(0, height)
        y_idx = randint(0, width)

        if (x_idx < dim / 2) or (height - x_idx < dim / 2):
            continue

        if (y_idx < dim / 2) or (width - y_idx < dim / 2):
            continue

        patch = img[0:4, x_idx - int((dim - 1) / 2):x_idx + int((dim - 1) / 2 + 1),
                y_idx - int((dim - 1) / 2):y_idx + int((dim - 1) / 2 + 1)]

        if np.count_nonzero(patch[0, :, :]) < 0.25 * (dim) ** 2:
            continue

        x_data.append(patch)
        y_data.append(img[4, x_idx, y_idx])

    # Get the cancerous cell patches
    for x_idx in range(0, height, 13):
        for y_idx in range(0, width, 13):

            if img[4, x_idx, y_idx]:

                if (x_idx < dim / 2) or (height - x_idx < dim / 2):
                    continue

                if (y_idx < dim / 2) or (width - y_idx < dim / 2):
                    continue

                patch = img[0:4, x_idx - int((dim - 1) / 2):x_idx + int((dim - 1) / 2 + 1),
                        y_idx - int((dim - 1) / 2):y_idx + int((dim - 1) / 2 + 1)]

                if np.count_nonzero(patch[0, :, :]) < 0.25 * (dim) ** 2:
                    continue

                x_data.append(patch)
                y_data.append(img[4, x_idx, y_idx])

    return x_data, y_data


if __name__ == '__main__':
    (x_train, y_train) = load_data('/home/apostolos/PycharmProjects/TestEnv/data/training_HGG/')
    print(x_train.dtype)
    print('Data size in Gb:', getsizeof(x_train))
    percentages = [0, 0, 0, 0, 0]

    for k in range(len(y_train)):
        percentages[int(y_train[k])] = percentages[int(y_train[k])] + 1

    print(percentages)
