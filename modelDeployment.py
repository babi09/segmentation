import detectCroppedSeg3DKerasDR_predict_ha
import streamlit as st

@st.cache
def runDeepSegmentationModel(organTarget, img):

    # model parameters
    params = {};
    params['TestSetNum'] = 1;
    params['tpUsed'] = 50;
    params['tDim'] = params['tpUsed'];
    params['PcUsed'] = 1;
    # params['visualizeResults']= 0;
    # params['visSlider']= 0;
    params['deepReduction'] = 0;

    params['networkToUseDetect'] = 'rbUnet'  # 'denseNet'; #'tNet'; #'Unet' #meshNet
    params['networkToUseSegment'] = 'tNet'  # 'denseNet'; #'rbUnet' # 'Unet' #meshNet
    params['selectedEpochDetect'] = '30000';
    params['selectedEpochSegment'] = '31735';

    if params['PcUsed'] == 1:
        tDim = 5;
        params['tDim'] = tDim;

    #pName = imagesAddress
    baseline = '1';

    # call the model to detect and segment and return the mask
    maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri = detectCroppedSeg3DKerasDR_predict_ha.singlePatientDetection(img, int(baseline),
                                                                                              params, 'Liver');
    #maskSegment = detectCroppedSeg3DKerasDR_predict_ha.singlePatientSegmentation(params, img, maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs,
    #                                            zDimOri, 'Liver');

    #return maskSegment
    return 0
