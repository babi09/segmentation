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



# streamlit interface

st.sidebar.title('Organ Detection and Segmentation')
flag_Liver_Model = 0

# upload file
@st.cache
def loadData(dataAddress):
    img_vol = funcs_ha_use.readVolume4(uploaded_nii_files.name)
    return img_vol

uploaded_nii_files = st.sidebar.file_uploader("Select file:", type=['nii', 'gz'])
if uploaded_nii_files is not None:
    # read the data into an array
    img_vol = loadData(uploaded_nii_files.name)
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
        maskSegment = modelDeployment.runDeepSegmentationModel('Liver', uploaded_nii_files.name)
        # plot segmentation mask
        fig = funcs_ha_use.plotMask(fig, ax, uploaded_nii_files.name, maskSegment, slice_i)
        #plot = st.pyplot(fig2)


# plot volume
    plot = st.pyplot(fig)

