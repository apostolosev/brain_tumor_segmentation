from __future__ import print_function

import matplotlib.pyplot as plt
import SimpleITK as sitk
import glob
import os
import numpy as np
import time
import cv2
import pywt
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


if __name__ == '__main__':
    img = cv2.imread('/home/apostolos/Downloads/lena.bmp', cv2.IMREAD_GRAYSCALE)
    out_img = resolution_enhancement(img, alpha=4)



