# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import modelDeployment
import funcs_ha_use
from PIL import Image
from nibabel import FileHolder, Nifti1Image
from io import BytesIO


# streamlit interface

st.sidebar.title('Organ Detection and Segmentation')
flag_Liver_Model = 0

# upload file
@st.cache
def loadData(dataAddress):
    img_vol = funcs_ha_use.readVolume4(dataAddress)
    return img_vol

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


#uploaded_nii_file = file_selector()

#st.write('You selected `%s`' % uploaded_nii_file)


uploaded_nii_file = st.sidebar.file_uploader("Select file:", type=['nii'])
# print (uploaded_nii_file)

if uploaded_nii_file is not None:
    rr = uploaded_nii_file.read()
    bb = BytesIO(rr)
    fh = FileHolder(fileobj=bb)
    img = Nifti1Image.from_file_map({'header': fh, 'image': fh})


    #img_vol = Image.open(uploaded_nii_file)
    #content = np.array(img_vol)  # pil to cv
    #print('yes')
    img_vol = loadData(img)
    # plot the data
    # plot the slider
    n_slices = img_vol.shape[2]
    slice_i = st.slider('Slice', 0, n_slices, int(n_slices / 2))

# plot volume
    fig, ax = plt.subplots()
    def plotImage(img_vol, slice_i):
        selected_slice = img_vol[:, :, slice_i, 1]

        ax.imshow(selected_slice, 'gray', interpolation='none')
        return fig
    fig = plotImage(img_vol, slice_i)
    #plot = st.pyplot(fig)

# select organ to segment
#     option = st.sidebar.selectbox('Select organ', ('Kidneys', 'Liver', 'Pancreas'))
#     segmentation = st.sidebar.button('Perform Segmentation')
    option = st.sidebar.radio('Select Organ to segment', ['None', 'Kidney', 'Liver', 'Pancreas', 'Psoas Muscles'], index=0)

    if option == 'Liver':
        # load segmentation model
        # perform segmentation
        maskSegment = modelDeployment.runDeepSegmentationModel('Liver', img)
        write('step1')
        # plot segmentation mask
        fig = funcs_ha_use.plotMask(fig, ax, img, maskSegment, slice_i)

# plot volume
    plot = st.pyplot(fig)

