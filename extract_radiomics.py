import os, re
import shutil
import time, six

import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk

from radiomics import featureextractor



def extract_features(tmppath, i, j, colsn, name, slc, mask, labels, bin_width, normalize, params):

    if os.path.exists(os.path.join(tmppath, 'tmp_{0:04d}_{1:03d}.csv'.format(i, j))):
        return True

    time_start = time.time()
    aux = pd.Series(index=colsn, data=np.zeros(len(colsn)))
    aux['id'] = os.path.basename(name)
    aux['slice'] = j+1
    aux['bin_width'] = bin_width
    aux['normalize'] = normalize
    print('Extracting radiomics for:')
    print(' - image: ', name)
    print(' - mask:  ', mask)
    mk = sitk.GetImageFromArray(nib.load(mask).get_fdata())
    for lb in labels:
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
        extractor.settings['binWidth'] = bin_width
        extractor.settings['normalize'] = normalize
        try:
            result = extractor.execute(slc, mk, label=int(lb))
            for key, val in six.iteritems(result):
                if key[:9] != 'original_':
                    continue
                aux[re.sub(r'original', 'lb{}'.format(int(lb)), key)] = val
        except ValueError as err:
            print(' extraction failed for this label: {}. Error:'.format(lb))
            print(err)
            continue

    aux.to_csv(os.path.join(tmppath, 'tmp_{0:04d}_{1:03d}.csv'.format(i, j)), header=True)
    time_end = time.time()

    print('Slice {0:03d} - Time {1:.2f} s'.format(j, time_end - time_start))


def extract(images, masks, output_path, bin_width=25, normalize=False):
    '''
    Extract radiomics features from a set of images
    TODO: Adapt script for an option with 4D masks.
    '''
    # ------------------
    # 1) Load settings for feature extractor and prepare variables
    # ------------------
    wd = os.path.realpath(os.path.dirname(__file__))
    params = os.path.join(wd, 'Params.yaml')
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    extractor.enableAllFeatures()
    extractor.settings['binWidth'] = bin_width
    extractor.settings['normalize'] = normalize

    # Temporary path to save features during the execution, in case the process
    # breaks, so it can be restarted.
    tmppath = os.path.join(output_path, 'tmp')
    try:
        # This folder should not exist, since the runXXX folder is always new.
        os.makedirs(tmppath)
    except FileExistsError:
        print('''
        ERROR: Trying to create folder "{}" but it was not possible. Check that
        the runXXX folder is a new one.'''.format(tmppath))
        return False

    # Get available labels in first mask (and consider them as the labels to
    # extract for the rest)
    labels = np.unique(nib.load(masks[0]).get_fdata()).astype(int)
    labels = labels[labels>0]

    assert len(images) == len(masks), \
        '''Found different number of images versus masks: {} vs.
        {}'''.format(len(images), len(masks))
    print('Found images and masks such as', images[0], masks[0])

    # ------------------
    # 2) Take a sample image and set column names for the radiomics dataframe
    # ------------------
    nii = nib.load(images[0])
    if len(nii.shape) == 4:
        auxim = nii.slicer[...,0].get_fdata()
        sample_image = sitk.GetImageFromArray(auxim)
    elif len(nii.shape) == 3:
        auxim = nii.get_fdata()
        sample_image = sitk.GetImageFromArray(auxim)
    else:
        raise Exception('''Image shape is {}. Supported shapes are in 3D or
                        4D formats'''.format(images[0].shape))

    mk = nib.load(masks[0]).get_fdata()
    mk = sitk.GetImageFromArray(mk)
    result = extractor.execute(sample_image, mk, label=int(labels[0]))
    aux = []
    for key, _ in six.iteritems(result):
        if key[:9] != 'original_':
            continue
        aux.append(re.sub(r'original', '', key))
    cols = []
    for lb in labels:
        cols.extend(['lb{}'.format(int(lb)) + s for s in aux])

    # ------------------
    # 3) Extract radiomics features for all images found
    # ------------------
    colsn = ['id', 'slice', 'bin_width', 'normalize'] + cols
    for i, image in enumerate(images):
        nii = nib.load(image)
        slc_num = 1 if len(nii.shape) == 3 else nii.shape[-1]

        # Iterate over each temporal slice in case it is available
        for j in range(slc_num):
            if slc_num > 1:
                auxim = nii.slicer[...,j].get_fdata()
                slc = sitk.GetImageFromArray(auxim)
            else:
                auxim = nii.get_fdata()
                slc = sitk.GetImageFromArray(auxim)

            extract_features(
                tmppath, i, j, colsn, image, slc, masks[i], 
                labels, bin_width, normalize, params
            )

    # ------------------
    # 4) Save results to a pandas DataFrame
    # ------------------
    df = pd.DataFrame(data=np.zeros((len(images), len(colsn))), columns=colsn)
    for i,f in enumerate(sorted(os.listdir(tmppath))):
        arr = pd.read_csv(os.path.join(tmppath, f), index_col=0)
        df.loc[i, arr.index.values] = arr.values.flatten()

    csv_file_path = os.path.join(output_path, 'radiomic_features.csv')
    df.to_csv(csv_file_path, index=True)
    shutil.rmtree(tmppath)

    return csv_file_path
