import sys
import os
from os import path
#sys.path.insert(0, '/fileserver/abd/github_ha_editing/')

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  ##P4000=='0', P6000=='1'

import funcs_ha_use
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
# matplotlib.axes.Axes.plot
# matplotlib.pyplot.plot
# matplotlib.axes.Axes.legend
# matplotlib.pyplot.legend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import scipy
from scipy.ndimage import zoom
from scipy import signal
from scipy.interpolate import interp1d
import pickle
from sklearn.metrics import precision_recall_curve
from skimage import morphology
from keras.models import Model,load_model,Sequential
from networks_ah import get_unet2, get_rbunet, get_meshNet, get_denseNet, calculatedPerfMeasures
from networks_ah import get_unetCnnRnn
from networks_ah import get_denseNet103, get_unet3
#from selectTrainAndTestSubjects_ha_2_use import selectTrainAndTestSubjects

reconMethod = 'SCAN';

def singlePatientDetection(pName, baseline, params, organTarget):
    
    #TestSetNum=params['TestSetNum'];
    tDim = params['tDim'];
    #tpUsed = params['tpUsed'];
    deepRed = params['deepReduction'];
    PcUsed = params['PcUsed'];
    #visEnabled = params['visualizeResults'];
    #visSlider = params['visSlider']; 

    ##### extract input image data (vol4D00)
    vol4D00,_,_,_,_ = funcs_ha_use.readData4(pName,reconMethod,0);
    zDimOri = vol4D00.shape[2];
    
    # xDim = vol4D00.shape[0];
    # yDim = vol4D00.shape[1];
    
    # timeRes0=subjectInfo['timeRes'][pName];
    # if not isinstance(timeRes0, (int, float)): 
    #     timeRes=float(timeRes0.split("[")[1].split(",")[0]);
    # else:
    #     timeRes=np.copy(timeRes0); 
    
    # start from baseline      
    im = vol4D00[:,:,:,baseline:];

    # medianFind = np.median(im);
    # if medianFind == 0:
    #     medianFind = 1.0;
    # im = im/medianFind;
    
    im=im/np.nanmean(im);
    
    vol4D0 = np.copy(im);
    # origTimeResVec=np.arange(0,vol4D0.shape[3]*timeRes,timeRes);
    # resamTimeResVec=np.arange(0,50*6,6);   # resample to 50 data points #100
    
    # if origTimeResVec[-1]<resamTimeResVec[-1]:
    #     print(pName)

    # # interpolate to temporal dimension of 50
    # f_out = interp1d(origTimeResVec,vol4D0, axis=3,bounds_error=False,fill_value=0);     
    # vol4D0 = f_out(resamTimeResVec);

    # perform PCA to numPC 
    numPC = 5; #50
    pca = PCA(n_components=numPC);
    vol4Dvecs=np.reshape(vol4D0, (vol4D0.shape[0]*vol4D0.shape[1]*vol4D0.shape[2], vol4D0.shape[3]));
    PCs=pca.fit_transform(vol4Dvecs);
    vol4Dpcs=np.reshape(PCs, (vol4D0.shape[0],vol4D0.shape[1],vol4D0.shape[2], numPC));
        
    dpcs = np.copy(vol4Dpcs);
    dpcs=dpcs/dpcs.max();
    da = dpcs.T;

    # downsample to 64 x 64 x 64 in x-y-z-dimenions
    # dsFactor = 3.5; 
    zDim = 64; yDim = 64; zDim = 64;
    # im0 = zoom(da,(1,zDim/da.shape[1],1/dsFactor,1/dsFactor),order=0);
    im0 = zoom(da,(1,zDim/da.shape[1],yDim/da.shape[2],zDim/da.shape[3]),order=0);
    
    sx = 0; xyDim = 64; 
    DataTest=np.zeros((1,zDim,xyDim,xyDim,tDim));
    DataTest[sx,:,:,:,:]=np.swapaxes(im0.T,0,2);
    
    #initialise detection model
    n_channels = tDim; n_classes = 3;
    
    #address to detection model
    #address = '/static/Liver/'
    #address = '/fileserver/abd/github_ha/deepLearningModels/detect_ha_gh//NetrbUnet_time5_pcUsed1_tpUsed50_DR0_testSet2/'
    
    #choose relevant detection model
    networkToUse = params['networkToUseDetect'];
    if networkToUse == 'rbUnet':
        model = get_rbunet(xyDim,zDim,n_channels,n_classes,deepRed,0);
    elif networkToUse == 'Unet':
        model = get_unet3(xyDim,zDim,n_channels,n_classes,deepRed,0);
    elif networkToUse == 'denseNet':
        model = get_denseNet(xyDim,zDim,n_channels,n_classes,deepRed,0);
    elif networkToUse == 'tNet':
        model = get_denseNet103(xyDim,zDim,n_channels,n_classes,deepRed,0); 
    
    #load detection model weights
    selectedEpoch=params['selectedEpochDetect'];

    # select organ to segment
    if organTarget == 'Liver':
        model.load_weights('detect3D_30000.h5');
    # elif organTarget == 'Pancreas':
    #     #model for pancreas
    # elif organTarget == 'Psoas':
    #     #model for Psoas
    # elif organTarget == 'Kidneys':
    #     #model for kidneys;

    #### perform prediction ####
    imgs_mask_test= model.predict(DataTest, verbose=1);
    
    multiHead = 0;
    if multiHead:
        labels_pred=np.argmax(imgs_mask_test[0], axis=4)
    else:
        labels_pred=np.argmax(imgs_mask_test, axis=4)
                
    # ensure all detected labels for right kidney are on the right half of x-dimension
    # labels_pred[:,:,:,0:int(xyDim/2)][labels_pred[:,:,:,0:int(xyDim/2)]==2]=1;
    # labels_pred[:,:,:,int(xyDim/2):][labels_pred[:,:,:,int(xyDim/2):]==1]=2;            

    
    ##### generate bounding boxes from coarse segmentation #####
    si = 0;    
    
    left = labels_pred[si,:,:,:].T==2;
    left = left.astype(int);
    
    right = labels_pred[si,:,:,:].T==1;
    right = right.astype(int);

    ####### resample to original input image size dimensions
    
    # xyDimOri = 224;
    zDimm = vol4D00.shape[2];
    xDim = vol4D00.shape[0];
    yDim = vol4D00.shape[1];
    
    # KMR=zoom(right,(xyDimOri/np.size(right,0),xyDimOri/np.size(right,1),zDimOri/np.size(right,2)),order=0);
    # KML=zoom(left,(xyDimOri/np.size(left,0),xyDimOri/np.size(left,1),zDimOri/np.size(left,2)),order=0);
    KMR = zoom(right,(xDim/np.size(right,0),yDim/np.size(right,1),zDimm/np.size(right,2)),order=0);
    KML = zoom(left,(xDim/np.size(left,0),yDim/np.size(left,1),zDimm/np.size(left,2)),order=0);
            
    
    if np.sum(KMR) != 0:
        KMR=morphology.remove_small_objects(KMR.astype(bool), min_size=256,in_place=True).astype(int);
        KMR = KMR.astype(int);
    if np.sum(KML) != 0:
        KML=morphology.remove_small_objects(KML.astype(bool), min_size=256,in_place=True).astype(int);   
        KML = KML.astype(int);
        KML[KML>=1]=2;

    #full kidney mask
    maskDetect = KMR + KML;
    
    ### generate kidneys bounding box based on prediction
    boxDetect = [];

    aL=np.nonzero(KML==2);
    aR=np.nonzero(KMR==1);

    if aL[0].size!=0:
        boxL=np.array([int((min(aL[0])+max(aL[0]))/2),int((min(aL[1])+max(aL[1]))/2),int((min(aL[2])+max(aL[2]))/2),\
          (max(aL[0])-min(aL[0])),(max(aL[1])-min(aL[1])),(max(aL[2])-min(aL[2]))])
    else:
        boxL=np.zeros((6,));
        
    if aR[0].size!=0:
        boxR=np.array([int((min(aR[0])+max(aR[0]))/2),int((min(aR[1])+max(aR[1]))/2),int((min(aR[2])+max(aR[2]))/2),\
          (max(aR[0])-min(aR[0])),(max(aR[1])-min(aR[1])),(max(aR[2])-min(aR[2]))])
    else:
        boxR=np.zeros((6,));
    
    # bounding box for right (boxDetect[0,:]) and left kidney (boxDetect[1,:])
    boxDetect=np.vstack([np.array(boxR),np.array(boxL)]);

    # identify whether right kidney exists
    # identify whether left kidney exists
    kidneyNone=np.nonzero(np.sum(boxDetect,axis=1)==0); #right/left
    if kidneyNone[0].size!=0:
        kidneyNone=np.nonzero(np.sum(boxDetect,axis=1)==0)[0][0]; #right/left
    
    # add extra margins to minimise impact of false-negative predictions
    KM = np.copy(maskDetect); KM[KM>1]=1;
    xSafeMagin=10;ySafeMagin=10;zSafeMagin=3;
    if boxDetect[0,2]+boxDetect[0,5]+3 >= KM.shape[2] or boxDetect[0,2]+boxDetect[0,5]-3 <0:
        boxDetect[:,[3,4,5]]=boxDetect[:,[3,4,5]]+[xSafeMagin,ySafeMagin,0];
    else:
        boxDetect[:,[3,4,5]]=boxDetect[:,[3,4,5]]+[xSafeMagin,ySafeMagin,zSafeMagin];

    
    # if aL[0].size!=0:
    #     if boxDetect[0,2]+boxDetect[0,5]+3 >= KM.shape[2] or boxDetect[0,2]+boxDetect[0,5]-3 <0:
    #         boxDetect[0,[3,4,5]]=boxDetect[0,[3,4,5]]+[xSafeMagin,ySafeMagin,0];
    #     else:
    #         boxDetect[0,[3,4,5]]=boxDetect[0,[3,4,5]]+[xSafeMagin,ySafeMagin,zSafeMagin];
    
    # if aR[0].size!=0:
    #     if boxDetect[0,2]+boxDetect[0,5]+3 >= KM.shape[2] or boxDetect[0,2]+boxDetect[0,5]-3 <0:
    #         boxDetect[1,[3,4,5]]=boxDetect[1,[3,4,5]]+[xSafeMagin,ySafeMagin,0];
    #     else:
    #         boxDetect[1,[3,4,5]]=boxDetect[1,[3,4,5]]+[xSafeMagin,ySafeMagin,zSafeMagin];
    
    
    # xSafeMagin=12;ySafeMagin=12;zSafeMagin=3;
    # boxDetect[:,[3,4,5]]=boxDetect[:,[3,4,5]]+[xSafeMagin,ySafeMagin,zSafeMagin];
    
    # kidneyNone=np.nonzero(np.sum(boxDetect,axis=1)==0); #right/left
    # if kidneyNone[0].size!=0:
    #     kidneyNone=np.nonzero(np.sum(boxDetect,axis=1)==0)[0][0]; #right/left
    

    # predMaskR=np.zeros((1,xyDimOri,xyDimOri,zDimOri));
    # predMaskL=np.zeros((1,xyDimOri,xyDimOri,zDimOri));
    predMaskR=np.zeros((1,xDim,yDim,zDimm));
    predMaskL=np.zeros((1,xDim,yDim,zDimm));
    
    sc = 0;
    predMaskR[sc,:,:,:]=KMR; 
    predMaskL[sc,:,:,:]=KML;    

    Masks2Save={};

    predMaskR2=zoom(predMaskR[sc,:,:,:],(1,1,1),order=0);
    predMaskL2=zoom(predMaskL[sc,:,:,:],(1,1,1),order=0);
    
    Masks2Save['R']=np.copy(predMaskR2.astype(float));
    Masks2Save['L']=np.copy(predMaskL2.astype(float));
    
    #### write kidney masks to file ####    
    #funcs_ha_use.writeMasksDetect(pName,reconMethod,Masks2Save,1);
            
    return maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri

def singlePatientSegmentation(params, pName, maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri, organTarget):
    
    #TestSetNum=params['TestSetNum'];
    tDim = params['tDim'];
    #tpUsed = params['tpUsed'];
    deepRed = params['deepReduction'];
    PcUsed = params['PcUsed'];
    #visEnabled = params['visualizeResults'];
    #visSlider = params['visSlider']; 
    
    dx = 64; dy = 64; dz = 64;
    Box = np.copy(boxDetect);
    maskDetect[maskDetect>1]=1;
    
    # crop out kidney images using bounding boxes
    exv = 0; #Jennifer Nowlan (+5, L)
    if kidneyNone!=0:
        croppedData4DR_pcs=vol4Dpcs[int(Box[0,0]-int(Box[0,3]/2)+exv):int(Box[0,0]+int(Box[0,3]/2)+exv),\
                                int(Box[0,1]-int(Box[0,4]/2)+exv):int(Box[0,1]+int(Box[0,4]/2)+exv),\
                                int(Box[0,2]-int(Box[0,5]/2)+exv):int(Box[0,2]+int(Box[0,5]/2)+exv),:];
        croppedData4DR=vol4D0[int(Box[0,0]-int(Box[0,3]/2)+exv):int(Box[0,0]+int(Box[0,3]/2)+exv),\
                                int(Box[0,1]-int(Box[0,4]/2)+exv):int(Box[0,1]+int(Box[0,4]/2)+exv),\
                                int(Box[0,2]-int(Box[0,5]/2)+exv):int(Box[0,2]+int(Box[0,5]/2)+exv),:];
        
        croppedData4DR_pcs=zoom(croppedData4DR_pcs,(dx/np.size(croppedData4DR_pcs,0),dy/np.size(croppedData4DR_pcs,1),dz/np.size(croppedData4DR_pcs,2),1),order=0);
        croppedData4DR=zoom(croppedData4DR,(dx/np.size(croppedData4DR,0),dy/np.size(croppedData4DR,1),dz/np.size(croppedData4DR,2),1),order=0);
        
    # check01 = Box[1,0]-int(Box[1,3]/2)+exv
    # check11 = Box[1,0]+int(Box[1,3]/2)-exv
    # check02 = Box[1,1]-int(Box[1,4]/2)+exv
    # check12 = Box[1,1]+int(Box[1,4]/2)-exv
    # check03 = Box[1,2]-int(Box[1,5]/2)+exv
    # check13 = Box[1,2]+int(Box[1,5]/2)-exv
    # Box[1,2]-int(Box[1,5]/2)+exv:Box[1,2]+int(Box[1,5]/2)-exv
    
    if kidneyNone!=1:    
        croppedData4DL_pcs=vol4Dpcs[int(Box[1,0]-int(Box[1,3]/2)+exv):int(Box[1,0]+int(Box[1,3]/2)-exv),\
                                int(Box[1,1]-int(Box[1,4]/2)+exv):int(Box[1,1]+int(Box[1,4]/2)-exv),\
                                int(Box[1,2]-int(Box[1,5]/2)+exv):int(Box[1,2]+int(Box[1,5]/2)-exv),:]; 
            
        croppedData4DL=vol4D0[int(Box[1,0]-int(Box[1,3]/2)+exv):int(Box[1,0]+int(Box[1,3]/2)-exv),\
                                int(Box[1,1]-int(Box[1,4]/2)+exv):int(Box[1,1]+int(Box[1,4]/2)-exv),\
                                int(Box[1,2]-int(Box[1,5]/2)+exv):int(Box[1,2]+int(Box[1,5]/2)-exv),:];  
            
        croppedData4DL_pcs=zoom(croppedData4DL_pcs,(dx/np.size(croppedData4DL_pcs,0),dy/np.size(croppedData4DL_pcs,1),dz/np.size(croppedData4DL_pcs,2),1),order=0);
        croppedData4DL=zoom(croppedData4DL,(dx/np.size(croppedData4DL,0),dy/np.size(croppedData4DL,1),dz/np.size(croppedData4DL,2),1),order=0);
        
    if kidneyNone==0:
        d=np.concatenate((croppedData4DL[np.newaxis,:,:,:,:],croppedData4DL[np.newaxis,:,:,:,:]),axis=0);
        dpcs=np.concatenate((croppedData4DL_pcs[np.newaxis,:,:,:,:],croppedData4DL_pcs[np.newaxis,:,:,:,:]),axis=0);
    elif kidneyNone==1:
        d=np.concatenate((croppedData4DR[np.newaxis,:,:,:,:],croppedData4DR[np.newaxis,:,:,:,:]),axis=0);
        dpcs=np.concatenate((croppedData4DR_pcs[np.newaxis,:,:,:,:],croppedData4DR_pcs[np.newaxis,:,:,:,:]),axis=0);
    else:
        d=np.concatenate((croppedData4DR[np.newaxis,:,:,:,:],croppedData4DL[np.newaxis,:,:,:,:]),axis=0);
        dpcs=np.concatenate((croppedData4DR_pcs[np.newaxis,:,:,:,:],croppedData4DL_pcs[np.newaxis,:,:,:,:]),axis=0);
        
    d=d/d.max()
    dpcs=dpcs/dpcs.max();
    
    sc=0; n_channels = tDim;
    DataCroppedTest=np.zeros((2,dx,dy,dz,n_channels));
    DataCroppedTest[2*sc:2*sc+2,:,:,:,:]=dpcs;
    
    #address to segmentation model
    address = '/static/Liver/';
    #address = '/fileserver/abd/github_ha/deepLearningModels/segment_ha_gh/NettNet_time5_pcUsed1_tpUsed50_DR0_testSet2/';
    
    
    #choose relevant segmentation model
    n_classes = 2;
    networkToUse = params['networkToUseSegment'];

    if networkToUse == 'tNet':
        model = get_denseNet103(dx,dz,n_channels,n_classes,deepRed,0);
    elif networkToUse == 'rbUnet':
        model = get_rbunet(dx,dz,n_channels,n_classes,deepRed,0);
    elif networkToUse == 'Unet':
        model = get_unet3(dx,dz,n_channels,n_classes,deepRed,0);
    elif networkToUse == 'denseNet':
        model = get_denseNet(dx,dz,n_channels,n_classes,deepRed,0);
    
    #load segmentation model weights
    selectedEpoch=params['selectedEpochSegment'];
    # select organ to segment
    if organTarget == 'Liver':
        model.load_weights('croppedSeg3D_31735.h5');

    # perform prediction
    cropped_mask_test = model.predict(DataCroppedTest, verbose=1)
    if cropped_mask_test.min()<0:
        cropped_mask_test=abs(cropped_mask_test.min())+cropped_mask_test;
        
    imgs_mask_test2=np.copy(cropped_mask_test);
    imgs_mask_test2[:,:,:,:,0]=cropped_mask_test[:,:,:,:,0];
    imgs_mask_test2[:,:,:,:,1]=cropped_mask_test[:,:,:,:,1];
    labels_pred_2=np.argmax(imgs_mask_test2, axis=4);
    
    # insert predicted kidney masks into relevant positions in 
    # original image spatial dimensions
    # xyDim=224;
    
    # predMaskR=np.zeros((1,xyDim,xyDim,zDimOri));
    # predMaskL=np.zeros((1,xyDim,xyDim,zDimOri));
    
    xDim = vol4D0.shape[0]
    yDim = vol4D0.shape[1]
            
    predMaskR=np.zeros((1,xDim,yDim,zDimOri));
    predMaskL=np.zeros((1,xDim,yDim,zDimOri));
    
    if kidneyNone!=0:
        Rk=labels_pred_2[2*sc,:,:,:]
        croppedData4DR=signal.resample(Rk,int(Box[0,3]), t=None, axis=0);
        croppedData4DR=signal.resample(croppedData4DR,int(Box[0,4]), t=None, axis=1);
        croppedData4DR=signal.resample(croppedData4DR,int(Box[0,5]), t=None, axis=2);
        croppedData4DR[croppedData4DR>0.5]=2;croppedData4DR[croppedData4DR<0.5]=0
        croppedData4DR[croppedData4DR==0]=1;croppedData4DR[croppedData4DR==2]=0  
        
        predMaskR[sc,int(Box[0,0]-Box[0,3]/2):int(Box[0,0]+Box[0,3]/2),\
                            int(Box[0,1]-Box[0,4]/2):int(Box[0,1]+Box[0,4]/2),\
                            int(Box[0,2]-Box[0,5]/2):int(Box[0,2]+Box[0,5]/2)]=croppedData4DR;
                        

    if kidneyNone!=1:     
            Lk=labels_pred_2[2*sc+1,:,:,:]
            croppedData4DL=signal.resample(Lk,int(Box[1,3]), t=None, axis=0);
            croppedData4DL=signal.resample(croppedData4DL,int(Box[1,4]), t=None, axis=1);
            croppedData4DL=signal.resample(croppedData4DL,int(Box[1,5]), t=None, axis=2);
            croppedData4DL[croppedData4DL>0.5]=2; croppedData4DL[croppedData4DL<0.5]=0
            croppedData4DL[croppedData4DL==0]=1;croppedData4DL[croppedData4DL==2]=0    
            
            predMaskL[sc,int(Box[1,0]-Box[1,3]/2):int(Box[1,0]+Box[1,3]/2),\
                                int(Box[1,1]-Box[1,4]/2):int(Box[1,1]+Box[1,4]/2),\
                                int(Box[1,2]-Box[1,5]/2):int(Box[1,2]+Box[1,5]/2)]=croppedData4DL;    
        
        
    if np.sum(predMaskR) != 0:
            predMaskL=morphology.remove_small_objects(predMaskL.astype(bool), min_size=256,in_place=True).astype(int);
    if np.sum(predMaskL) != 0:
            predMaskR=morphology.remove_small_objects(predMaskR.astype(bool), min_size=256,in_place=True).astype(int);
    
    predMaskL2=np.copy(predMaskL);
    #predMaskL2[predMaskL2==1]=2;    
        
    Masks2Save={};
        
    predMaskR2=zoom(predMaskR[sc,:,:,:],(1,1,1),order=0);
    predMaskL2=zoom(predMaskL[sc,:,:,:],(1,1,1),order=0);
    maskSegment = predMaskR2 + predMaskL2;
        
    Masks2Save['R']=np.copy(predMaskR2.astype(float));
    Masks2Save['L']=np.copy(predMaskL2.astype(float));

    # write kidney segmentation masks to file
    #funcs_ha_use.writeMasks(pName,reconMethod,Masks2Save,1);
    
    return Masks2Save
    
# # path to .xls spreadsheet that contains temporal information about each
# # test subject (pName)
# ## fileAddress='/fileserver/external/rdynaspro4/abd/MRUcommon/subjectDicomInfo_ha.xls';
# ## subjectInfo=pd.read_excel(fileAddress, sheetname=0);
# reconMethod='SCAN';
#
# params={};
# params['TestSetNum']=1;
# params['tpUsed']= 50;
# params['tDim']= params['tpUsed'];
# params['PcUsed']= 1;
# #params['visualizeResults']= 0;
# #params['visSlider']= 0;
# params['deepReduction']= 0;
#
# params['networkToUseDetect']= 'rbUnet' #'denseNet'; #'tNet'; #'Unet' #meshNet
# params['networkToUseSegment']= 'tNet' #'denseNet'; #'rbUnet' # 'Unet' #meshNet
# params['selectedEpochDetect']='30000';
# params['selectedEpochSegment']='31735';
#
#
# if params['PcUsed']== 1:
#     tDim=5;
#     params['tDim']= tDim;
#
# # TestSetNum = 1; #2
# # _,subjectNamesNormalTest,_,testKidCondTest,subjectNamesNormalTestBaselines=selectTrainAndTestSubjects(TestSetNum);
# #subjectTrain,subjectTest,subjectTrainKidneyCondition,subjectTestKidneyCondition,subjectTestBaselines
#
# #for s in range(len(subjectNamesNormalTest)):
# for s in range(1):
#
#     #s = 1;
#     #pName = subjectNamesNormalTest[s];
#     pName = '6VFCD8yFmwJoPxpvZwf6il_LIVER.nii';
#
#     baseline = '1'; # 5, 8
#
#     # print(pName)
#     # pathToFolderD = '.\\detected'
#     # if not os.path.exists(pathToFolderD):
#     #     os.makedirs(pathToFolderD)
#     #
#     # pathToFolder = '.\\segmented'
#     # if not os.path.exists(pathToFolder):
#     #     os.makedirs(pathToFolder)
#
#     maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri = singlePatientDetection(pName, int(baseline),params, 'Liver');
#     maskSegment = singlePatientSegmentation(params, pName, maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri, 'Liver');
#
#     funcs_ha_use.plotMask(pName, maskSegment, 268)

