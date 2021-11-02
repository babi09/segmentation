#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import os, sys
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import zoom
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
#import SimpleITK as sitk
from matplotlib.widgets import Slider, Button, RadioButtons


# read input image data including kidney ground-truth masks
# def readData4(patientName,subjectInfo,reconMethod,genBoundBox):
def readData4(img, reconMethod, genBoundBox):
    
    # seqNum=subjectInfo['numSeq'][patientName];
    seqNum = 1

    # reconstruction Method
    # scanner ---> 'SCAN'
    # grasp   ---> 'GRASP'
    
    # dataAddress0='/fileserver/projects6/ha/psoas_nii_ori/'
    # dataAddress0='/fileserver/projects6/ha/LIVER_nii_ori/'
    # dataAddress0='/fileserver/projects6/ha/DUN_nii_ori/'
    # dataAddress0='/fileserver/projects6/ha/EXE_nii_ori/'
    ## dataAddress0='/fileserver/projects6/ha/CT_nii_ori/'
    
    # if reconMethod=='GRASP':
    #     dataAddress0='/fileserver/external/rdynaspro4/abd/GraspRecons/reconResults'+reconMethod+'/method2/';
    #     #dataAddress0='/fileserver/xxx/GraspRecons/reconResults'+reconMethod+'/method2/';
    # elif reconMethod=='SCAN':
    #     dataAddress0='/fileserver/projects6/ha/psoas_nii_ori/';
    #     #dataAddress0='/fileserver/xxx/reconResults'+reconMethod+'/';

    if seqNum==1:
        # dataAddress=dataAddress0+patientName+'/'+patientName+'_psoas.nii';
        # dataAddress=dataAddress0+patientName+'/'+patientName+'_LIVER.nii';
        # dataAddress=dataAddress0+patientName+'/'+patientName+'_DUN.nii';
        # dataAddress=dataAddress0+patientName+'/'+patientName+'_EXE.nii';
        ## dataAddress=dataAddress0+patientName+'/'+patientName+'_DCM.nii';
        #img= nib.load(dataAddress)
        im=img.get_data() 
        
        check0 = np.min(im[:])
        check1 = np.max(im[:])
        check2 = np.mean(im[:])
        check3 = np.median(im[:])
        
        im2 = np.zeros((im.shape[0],im.shape[1],im.shape[2],6))
        
        for ix in range(0,5):
            im2[:,:,:,ix]=im[:,:,:]
        # im2[:,:,:,1]=im[:,:,:]
        # im2[:,:,:,2]=im[:,:,:]
        # im2[:,:,:,3]=im[:,:,:]
        # im2[:,:,:,4]=im[:,:,:]
        
        check0 = np.min(im2[:])
        check1 = np.max(im2[:])
        check2 = np.mean(im2[:])
        check3 = np.median(im2[:])
        
        #maskAddress= 'currentMask';
        #maskAddress='/fileserver/xxx/GraspRecons/reconResultsSCAN/'+patientName+'/';
        maskAddress = 'leftCTMask_automatic.nii.gz'
        if seqNum==1:
            
            am = 1 
            
            # if os.path.isfile(maskAddress+'_Label.nii'):
            #     lkm1= nib.load(maskAddress+'_Label.nii');lkm=2*lkm1.get_data();
            #     lkm[lkm>2]=2;
            # else:
            #     lkm=np.zeros(np.shape(im));
            #
            # if os.path.isfile(maskAddress+'_Label.nii'):
            #     rkm1= nib.load(maskAddress+'_Label.nii');rkm=rkm1.get_data();
            #     rkm[rkm>1]=1;
            # else:
            #     rkm=np.zeros(np.shape(im));
            if os.path.isfile(maskAddress):
                lkm1 = nib.load(maskAddress);
                lkm = 2 * lkm1.get_data();
                lkm[lkm > 2] = 2;
            else:
                lkm = np.zeros(np.shape(im));

            if os.path.isfile(maskAddress):
                rkm1 = nib.load(maskAddress);
                rkm = rkm1.get_data();
                rkm[rkm > 1] = 1;
            else:
                rkm = np.zeros(np.shape(im));
        else:
            lkm=0;rkm=0;am=0;
        
             
    else:
        lkm=0;rkm=0;am=0;


    boxes=[];
    if genBoundBox:
        aL=np.nonzero(lkm==2);
        aR=np.nonzero(rkm==1);

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
        
        boxes=np.vstack([np.array(boxR),np.array(boxL)]);
        
    oriKM = rkm+lkm;
    im2=(im2/np.amax(im2))*100;
    
    check0 = np.min(im2[:])
    check1 = np.max(im2[:])
    check2 = np.mean(im2[:])
    check3 = np.median(im2[:])
    
    return im2, oriKM, boxes, rkm,lkm

# read nii file and return image volume
def readVolume4(img):

    # img = nib.load(dataAddress)
    im = img.get_data()

    check0 = np.min(im[:])
    check1 = np.max(im[:])
    check2 = np.mean(im[:])
    check3 = np.median(im[:])

    im2 = np.zeros((im.shape[0], im.shape[1], im.shape[2], 6))

    for ix in range(0, 5):
        im2[:, :, :, ix] = im[:, :, :]

    check0 = np.min(im2[:])
    check1 = np.max(im2[:])
    check2 = np.mean(im2[:])
    check3 = np.median(im2[:])

    im2 = (im2 / np.amax(im2)) * 100;

    check0 = np.min(im2[:])
    check1 = np.max(im2[:])
    check2 = np.mean(im2[:])
    check3 = np.median(im2[:])

    return im2

def plotMask(fig, ax, img, mask, slice_i):
    #img = nib.load(dataAddress)
    # img_vol = readVolume4(dataAddress)

    Masks2SaveL = mask['L'];
    Masks2Save1L = nib.Nifti1Image(Masks2SaveL, img.affine)

    lkm1 = Masks2Save1L
    lkm = 2 * lkm1.get_data()
    lkm[lkm > 2] = 2;

    tm90 = lkm[:, :, slice_i]
    tm90[tm90 >= 1] = 1
    masked = np.ma.masked_where(tm90 == 0, tm90)
    # colour map for ground-truth (red)
    cmapm = matplotlib.colors.ListedColormap(["red", "red", "red"], name='from_list', N=None)
    #selected_slice = img_vol[:, :, slice_i, 1]
    #fig, ax = plt.subplots()
    # ax.imshow(selected_slice, 'gray', interpolation='none')
    ax.imshow(masked, cmap=cmapm, interpolation='none', alpha=0.3)
    ax.contour(tm90, colors='red', linewidths=1.0)
    #
    return fig

    # ax.imshow(selected_slice, 'gray', interpolation='none')
    # ax.imshow(masked, cmap=cmapm, interpolation='none', alpha=0.3)
    # ax.contour(tm90, colors='red', linewidths=1.0)
   # plt.title('Slice ' + str(rr))

    # fig1 = plt.gcf()
    # # plt.gray()
    # plt.show()
    # plt.draw()
    #
    # return fig


# plot image volume
def plotImage(img_vol, slice_i):
    selected_slice = img_vol[:, :, slice_i,1]
    fig, ax = plt.subplots()
    ax.imshow(selected_slice, 'gray', interpolation='none')
    return fig




# read input image data
# read saved detection (coarse segmentation) kidney masks
# and generate respective kidney bounding masks
def readDetect(patientName,subjectInfo,reconMethod,genBoundBox):
    
    primaryAddress = '/fileserver/abd/github_ha/deepLearningModels/detect_ha_gh/detected/';
    seqNum=subjectInfo['numSeq'][patientName];

    if reconMethod=='GRASP':
        dataAddress0='/fileserver/external/rdynaspro4/abd/GraspRecons/reconResults'+reconMethod+'/method2/';
        #dataAddress0='/fileserver/xxxx/GraspRecons/reconResults'+reconMethod+'/method2/';
    elif reconMethod=='SCAN':
        dataAddress0='/fileserver/abd/GraspRecons/reconResults'+reconMethod+'/';
        #dataAddress0='/fileserver/xxx/GraspRecons/reconResults'+reconMethod+'/';

    if seqNum==1 or reconMethod=='GRASP':
        dataAddress=dataAddress0+patientName+'/recon'+reconMethod+'_4D.nii';
        img= nib.load(dataAddress)
        im=img.get_data()  
        
        maskAddress='/fileserver/abd/GraspRecons/reconResultsSCAN/'+patientName+'/';
        #maskAddress='/fileserver/xxx/GraspRecons/reconResultsSCAN/'+patientName+'/';
        if os.path.isfile(maskAddress+'aortaMask.nii'):
            am1= nib.load(maskAddress+'aortaMask.nii');am=am1.get_data();
            
            #maskAddressDetect='/fileserver/xxx/preprocessedData_ha/detect_singleSubjects_pc_gh/'+patientName+'_seq1'+'/';
            maskAddressDetect=primaryAddress+patientName+'_seq1'+'/'; 
           
            if os.path.isfile(maskAddressDetect+'leftKidneyMask_detected.nii.gz'):
                    lkm1= nib.load(maskAddressDetect+'leftKidneyMask_detected.nii.gz');lkm=2*lkm1.get_data();
                    lkm[lkm>2]=2;
            else:
                    lkm=np.zeros(np.shape(am));
                    
            if os.path.isfile(maskAddressDetect+'rightKidneyMask_detected.nii.gz'):
                    rkm1= nib.load(maskAddressDetect+'rightKidneyMask_detected.nii.gz');rkm=rkm1.get_data();
                    rkm[rkm>1]=1;
            else:
                    rkm=np.zeros(np.shape(am));

        else:
            lkm=0;rkm=0;am=0;
        
        
    elif seqNum==2:
        dataAddress=dataAddress0+patientName+'_seq1/recon'+reconMethod+'_4D.nii';  
        img1= nib.load(dataAddress)
        im1=img1.get_data()  
        dataAddress=dataAddress0+patientName+'_seq2/recon'+reconMethod+'_4D.nii';  
        img= nib.load(dataAddress)
        im2=img.get_data()  
        #im=np.concatenate((im1, im2), axis=3);

        x=subjectInfo['timeRes'][patientName];
        seq1tres=float(x.split("[")[1].split(",")[0]);
        seq2tres=float(x.split(",")[1].split("]")[0]);
        if seq2tres>seq1tres:
            #resample second to first
            num2=seq2tres/seq1tres;
            im3=zoom(im2,(1,1,1,num2),order=0);
            #num=int(np.round(im2.shape[3]*seq2tres/seq1tres))
            #im3=signal.resample(im2, num, t=None, axis=3)
            im=np.concatenate((im1, im3), axis=3);
        else:
            im=np.concatenate((im1, im2), axis=3);
            
        maskAddress='/fileserver/abd/GraspRecons/reconResultsSCAN/'+patientName+'_seq1/';
        if os.path.isfile(maskAddress+'aortaMask.nii.gz'):
            am1= nib.load(maskAddress+'aortaMask.nii.gz');am=am1.get_data() 
            
            maskAddressDetect = primaryAddress + patientName+'_seq1'+'/';   
            if os.path.isfile(maskAddressDetect+'leftKidneyMask_detected.nii.gz'):
                    lkm1= nib.load(maskAddressDetect+'leftKidneyMask_detected.nii.gz');lkm=2*lkm1.get_data();
                    lkm[lkm>2]=2;
            else:
                    lkm=np.zeros(np.shape(am));
                    
            if os.path.isfile(maskAddressDetect+'rightKidneyMask_detected.nii.gz'):
                    rkm1= nib.load(maskAddressDetect+'rightKidneyMask_detected.nii.gz');rkm=rkm1.get_data();
                    rkm[rkm>1]=1;
            else:
                    rkm=np.zeros(np.shape(am));    
        else:
            lkm=0;rkm=0;am=0;

    # generate kidney bounding boxes
    boxes=[];
    if genBoundBox:
        aL=np.nonzero(lkm==2);
        aR=np.nonzero(rkm==1);

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
        
        boxes=np.vstack([np.array(boxR),np.array(boxL)]);
        
        # add extra margins to minimise impact of false-negative predictions
        mask = rkm+lkm; mask[mask>1]=1;
        xSafeMagin=10;ySafeMagin=10;zSafeMagin=3;
        if boxes[0,2]+boxes[0,5]+3 >= mask.shape[2] or boxes[0,2]+boxes[0,5]-3 <0:
            boxes[:,[3,4,5]]=boxes[:,[3,4,5]]+[xSafeMagin,ySafeMagin,0];
        else:
            boxes[:,[3,4,5]]=boxes[:,[3,4,5]]+[xSafeMagin,ySafeMagin,zSafeMagin];
    
        # add extra margins to minimise impact of false-negative predictions
        #xSafeMagin=12;ySafeMagin=12;zSafeMagin=3;
        #Box[:,[3,4,5]]=Box[:,[3,4,5]]+[xSafeMagin,ySafeMagin,zSafeMagin];
        

    KM=rkm+lkm;
    im=(im/np.amax(im))*100;
    
    return im, KM, boxes, rkm,lkm

# save detection (coarse segmentation) masks to file
def writeMasksDetect(dataAddress,reconMethod,Masks2Save,overwrite):
    
   ## primaryAddress = '/fileserver/abd/github_ha_editing/deepLearningModels/detect_ha_gh/detected/';
    # seqNum=subjectInfo['numSeq'][patientName];
    seqNum = 1

    # dataAddress0='/fileserver/projects6/ha/psoas_nii_ori/'
    # dataAddress0='/fileserver/projects6/ha/LIVER_nii_ori/'
    # dataAddress0='/fileserver/projects6/ha/DUN_nii_ori/'
    # dataAddress0='/fileserver/projects6/ha/EXE_nii_ori/'
    ##dataAddress0='/fileserver/projects6/ha/CT_nii_ori/'
    
    # if reconMethod=='GRASP':
    #     dataAddress0='/fileserver/external/rdynaspro4/abd/GraspRecons/reconResults'+reconMethod+'/method2/';
    #     #dataAddress0='/fileserver/xxx/GraspRecons/reconResults'+reconMethod+'/method2/';
    # elif reconMethod=='SCAN':
    #     dataAddress0='/fileserver/projects6/ha/psoas_nii_ori/';
    #     #dataAddress0='/fileserver/xxx/reconResults'+reconMethod+'/';

    # reconstruction Method
    # scanner ---> 'SCAN'
    # grasp   ---> 'GRASP'
    
    # if reconMethod=='GRASP':
    #     dataAddress0='/fileserver/external/rdynaspro4/abd/GraspRecons/reconResults'+reconMethod+'/method2/';
    #     #dataAddress0='/fileserver/xxx/GraspRecons/reconResults'+reconMethod+'/method2/';
    # elif reconMethod=='SCAN':
    #     dataAddress0='/fileserver/abd/GraspRecons/reconResults'+reconMethod+'/';
    #     #dataAddress0='/fileserver/xxx/GraspRecons/reconResults'+reconMethod+'/';

    if seqNum==1:
        # dataAddress=dataAddress0+patientName+'/'+patientName+'_psoas.nii';
        # dataAddress=dataAddress0+patientName+'/'+patientName+'_LIVER.nii';
        # dataAddress=dataAddress0+patientName+'/'+patientName+'_DUN.nii';
        # dataAddress=dataAddress0+patientName+'/'+patientName+'_EXE.nii';
        ##dataAddress=dataAddress0+patientName+'/'+patientName+'_DCM.nii';
        img = nib.load(dataAddress)
        #im=img.get_data()  
        
        maskAddress = '.\\segment\\';
        #maskAddress='/fileserver/xxxx/preprocessedData_ha/detect_singleSubjects_pc_gh/'+patientName+'_seq1/';
        
        if os.path.isfile('leftCTMask_detected.nii') and not overwrite:
            print('Mask is already existant!');
        else:
            Masks2SaveR=Masks2Save['R'];Masks2SaveL=Masks2Save['L'];
            Masks2Save1R = nib.Nifti1Image(Masks2SaveR, img.affine)
            Masks2Save1L = nib.Nifti1Image(Masks2SaveL, img.affine)
            nib.save(Masks2Save1L,'leftCTMask_detected.nii.gz');
            nib.save(Masks2Save1R,'rightCTMask_detected.nii.gz');
            #oscommand='chmod -R 777 '+maskAddress;
            #os.system(oscommand);
    
    # elif seqNum==2:
    #     dataAddress=dataAddress0+patientName+'_seq1/recon'+reconMethod+'_T0.nii';  
    #     img1= nib.load(dataAddress)

    #     maskAddress = primaryAddress + patientName+'_seq1/';
    #     #maskAddress='/fileserver/xxxx/preprocessedData_ha/detect_singleSubjects_pc_gh/'+patientName+'_seq1/';
        
    #     if os.path.isfile(maskAddress+'leftKidneyMask_detected.nii') and not overwrite:    
    #         print('Mask is already existant!');
    #     else:
    #         Masks2SaveR=Masks2Save['R'];Masks2SaveL=Masks2Save['L'];
            
    #         summationL = np.sum(Masks2SaveL)
    #         if summationL == 0.0:
    #             #Masks2SaveL = Masks2SaveL.astype('int');
    #             Masks2SaveL[112,112,16] = 1.0;
                
    #         summationR = np.sum(Masks2SaveR)
    #         if summationR == 0.0:
    #             #Masks2SaveR = Masks2SaveR.astype('int');
    #             Masks2SaveR[112,112,16] = 1.0;
            
    #         Masks2Save1R = nib.Nifti1Image(Masks2SaveR, img1.affine)
    #         Masks2Save1L = nib.Nifti1Image(Masks2SaveL, img1.affine)
    #         nib.save(Masks2Save1L,maskAddress+'leftKidneyMask_detected.nii.gz');
    #         nib.save(Masks2Save1R,maskAddress+'rightKidneyMask_detected.nii.gz');
    #         oscommand='chmod -R 777 '+maskAddress;
    #         os.system(oscommand);
 
    return


# write kidney segmentation masks to file
def writeMasks(dataAddress,reconMethod,Masks2Save,overwrite):
    
    ##primaryAddress = '/fileserver/abd/github_ha_editing/deepLearningModels/segment_ha_gh/segmented/';
    # seqNum=subjectInfo['numSeq'][patientName];
    seqNum = 1

    # dataAddress0='/fileserver/projects6/ha/psoas_nii_ori/'
    # dataAddress0='/fileserver/projects6/ha/LIVER_nii_ori/'
    # dataAddress0='/fileserver/projects6/ha/DUN_nii_ori/'
    # dataAddress0='/fileserver/projects6/ha/EXE_nii_ori/'
    ##dataAddress0='/fileserver/projects6/ha/CT_nii_ori/'


    if seqNum==1:
        # dataAddress=dataAddress0+patientName+'/'+patientName+'_psoas.nii';
        # dataAddress=dataAddress0+patientName+'/'+patientName+'_LIVER.nii';
        # dataAddress=dataAddress0+patientName+'/'+patientName+'_DUN.nii';
        # dataAddress=dataAddress0+patientName+'/'+patientName+'_EXE.nii';
        ##dataAddress=dataAddress0+patientName+'/'+patientName+'_DCM.nii';
        
        img = nib.load(dataAddress)
        #im=img.get_data()  
        
        #maskAddress = 'segmentationMask/';
        #maskAddress='/fileserver/xxxx/preprocessedData_ha/detect_singleSubjects_pc_gh/'+patientName+'_seq1/';
        
        if os.path.isfile('leftCTMask_automatic.nii') and not overwrite:
            print('Mask is already existant!');
        else:
            Masks2SaveR=Masks2Save['R'];Masks2SaveL=Masks2Save['L'];
            Masks2Save1R = nib.Nifti1Image(Masks2SaveR, img.affine)
            Masks2Save1L = nib.Nifti1Image(Masks2SaveL, img.affine)
            nib.save(Masks2Save1L,'leftCTMask_automatic.nii.gz');
            nib.save(Masks2Save1R,'rightCTMask_automatic.nii.gz');
            
            #oscommand='chmod -R 777 '+maskAddress;
            #os.system(oscommand);

    # elif seqNum==2:
    #     dataAddress=dataAddress0+patientName+'_seq1/recon'+reconMethod+'_T0.nii';  
    #     img1= nib.load(dataAddress);
            
    #     maskAddress=primaryAddress+patientName+'_seq1/';
    #     #maskAddress='/fileserver/xxx/GraspRecons/reconResults_haSCAN_dense103_dax/'+patientName+'_seq1/';
    
    #     if os.path.isfile(maskAddress+'leftKidneyMask_automatic.nii') and not overwrite: 
    #         print('Mask is already existant!');
    #     else:
    #         Masks2SaveR=Masks2Save['R'];Masks2SaveL=Masks2Save['L'];
            
    #         summationL = np.sum(Masks2SaveL)
    #         if summationL == 0.0:
    #             #Masks2SaveL = Masks2SaveL.astype('int');
    #             Masks2SaveL[112,112,16] = 1.0;
                
    #         summationR = np.sum(Masks2SaveR)
    #         if summationR == 0.0:
    #             #Masks2SaveR = Masks2SaveR.astype('int');
    #             Masks2SaveR[112,112,16] = 1.0;
            
    #         Masks2Save1R = nib.Nifti1Image(Masks2SaveR, img1.affine)
    #         Masks2Save1L = nib.Nifti1Image(Masks2SaveL, img1.affine)
    #         nib.save(Masks2Save1L,maskAddress+'leftKidneyMask_automatic.nii.gz');
    #         nib.save(Masks2Save1R,maskAddress+'rightKidneyMask_automatic.nii.gz');
            
    #         oscommand='chmod -R 777 '+maskAddress;
    #         os.system(oscommand);
    
    return

# compute baseline from 4D image volume
def baselineFinder(im):
    aortaPotentialTimesIM=im[75:150,:,0:20,0:50]; # keep first 15 dataPoints
    #aortaPotentialTimesIM=im[:,:,:,0:1];
    x=(aortaPotentialTimesIM>.8*np.max(aortaPotentialTimesIM)).nonzero()    
    
    x=(np.max(aortaPotentialTimesIM,axis=3)-np.min(aortaPotentialTimesIM,axis=3)>.6*np.max(aortaPotentialTimesIM)).nonzero()    
    #b = Counter(x[2]);
    #mostOccInZ=b.most_common(1)[0][0];
    
    medianOfValsInXaxisNdx=(abs(x[0]-np.median(x[0]))<10).nonzero()[0];
    medianOfValsInYaxisNdx=(abs(x[1]-np.median(x[1]))<10).nonzero()[0];
    medianOfValsInZaxisNdx=(abs(x[2]-np.median(x[2]))<10).nonzero()[0];
    commonXYconstraint=np.intersect1d(medianOfValsInXaxisNdx,medianOfValsInYaxisNdx);
    commonXYconstraint=np.intersect1d(medianOfValsInZaxisNdx,commonXYconstraint)
    allAortaPotentials=aortaPotentialTimesIM[x[0][commonXYconstraint],x[1][commonXYconstraint],x[2][commonXYconstraint],:]
    
    v,timeNdx = allAortaPotentials.max(1),allAortaPotentials.argmax(1)
    maxA=np.median(timeNdx)
    y=np.mean(allAortaPotentials[:,0:int(maxA)+1],0);
    #plt.figure();plt.plot(y,'-*');
    #plt.figure();plt.plot(np.mean(allAortaPotentials,0),'-*');
    y2=(y*400)/y.max();y3=y2-y2.min()+50;
    baseLine=(y3<0.5*(max(y3)-min(y3))).nonzero()[0][-1];
    #baseLine=(y<0.5*(max(y)-min(y))).nonzero()[0][-1];
    #print('Automatically detected injection timePoint: '+str(baseLine))
    return baseLine

# reduce time-to-image 'gap' between from seq1-to-seq2 4D image volume (im)
# and return new 4D image volume (newVol4D)
def computeNew4D(im):    
    
    vol4D = np.array(im);
    xl = vol4D.shape[0];
    yl = vol4D.shape[1];
    sl = vol4D.shape[2];
    tl = vol4D.shape[3];
    
    reshapedVol4D = np.reshape(vol4D,[xl*yl*sl,tl]);
    fid =  np.mean(reshapedVol4D,axis=0)
    
    diffFid = np.diff(fid,n=1,axis=0) 
    absDiffFid = np.absolute(diffFid)
        
    maxAbsDiffFid = max(absDiffFid)
    
    jumpTimeSample, = np.where(absDiffFid == maxAbsDiffFid)
    jumpTimeSample = max(jumpTimeSample)
        
    difference = fid[jumpTimeSample+1]-fid[jumpTimeSample];
    
    newVol4D = np.copy(vol4D);
    oldVol4D = np.copy(vol4D);
   
    newVol4D[:,:,:,jumpTimeSample+1:] = np.subtract(oldVol4D[:,:,:,jumpTimeSample+1:],difference)

    return newVol4D

# compute baseline from 4D image volume
def baselineFinder2(patientName,subjectInfo):
    
    maskAddress='/fileserver/abd/GraspRecons/reconResultsSCAN/'+patientName;
    path1 = maskAddress+'/aortaMask.nii.gz';
    path2 = maskAddress+'_seq1/aortaMask.nii.gz';
    
    if os.path.isfile(path1):
        am1= nib.load(path1);am=am1.get_data() 
    elif os.path.isfile(path2): 
        am1= nib.load(path2);am=am1.get_data() 
        
    im, _,_,_,_= readData4(patientName,subjectInfo,'SCAN',0);
    im = computeNew4D(im);
        
    time = im.shape[3];
    
    aortaMask = np.zeros(np.shape(im));
    aortaInt = im;
    for xx in range (time):
        aortaMask[:,:,:,xx] = am;
    
    aortaInt[aortaMask==0]=0
    aortaMask = np.copy(aortaInt);
    
    xs,ys,zs,as1 = np.where(aortaMask!=0) 
    aortaMask2 = aortaMask[min(xs):max(xs)+1,min(ys):max(ys)+1,min(zs):max(zs)+1,min(as1):max(as1)+1]
    aortaMask2 = np.array(aortaMask2)
    aortaMask3 = np.reshape(aortaMask2,[aortaMask2.shape[0]*aortaMask2.shape[1]*aortaMask2.shape[2],time]);
    
    xs1,ys1 = np.where(aortaMask3 !=0) 
    aortaMask4 = aortaMask3[min(xs1):max(xs1)+1,min(ys1):max(ys1)+1]
    aortaMask4 = np.array(aortaMask4)
    
    whereIs = np.amax(aortaMask4, axis=1)
    indicesWhereIs = np.where(whereIs !=0)
    aortaMask4 = aortaMask4[indicesWhereIs, :]
    aortaMask4 = np.reshape(aortaMask4,[aortaMask4.shape[1],time])

    midWay = int(time/2)
    tWay = int(time/3)
    lookAtM = aortaMask4[:,1:midWay]
    lookAtT = aortaMask4[:,1:tWay]
    
    ranMax = np.amax(lookAtM, axis=1)   # Maxima along the second axis
    ranMin = np.amin(lookAtT, axis=1)   # Minima along the second axis
    rangeA = ranMax - ranMin
    
    maxKeepPercent = 40;
    condition = rangeA > (1-maxKeepPercent/100)*max(rangeA)
    valuesOfInterestINDICES = np.where(condition)
    
    aortaMask5 = aortaMask4[valuesOfInterestINDICES,:]
    aortaMask5 = np.reshape(aortaMask5,[aortaMask5.shape[1],time])
    
    checkLength = aortaMask5.shape[0];
    powerOfHF = []

    for xx in range(checkLength): 
        arrayVI = np.array(aortaMask5[xx,:]);
        maximumVI = max(arrayVI)
        peak, = np.where(arrayVI == maximumVI)
        peak = max(peak)
        diffS = np.diff(aortaMask5[xx,peak:],n=1,axis=0) 
        squaredDiffs = np.square(diffS);
        sumSquaredDiffs = np.sum(squaredDiffs)
        powerOfHF.append(sumSquaredDiffs)
        
    powerOfHF = np.array(powerOfHF)
    lenHF = powerOfHF.shape[0]
    
    sortedSmoothestVoxels = sorted(range(lenHF), key=lambda k:  powerOfHF[k])
    smoothestVoxels = sortedSmoothestVoxels[1:int(len(sortedSmoothestVoxels)/2)];
    selectedVoxels =  np.array(aortaMask5[smoothestVoxels,:]); 
    aif = np.median(selectedVoxels,axis=0); 
    
    aifMAX = max(aif)
    aifMndx, = np.where(aif == aifMAX)
    aifMndx = max(aifMndx)    
    aifDif = np.diff(aif[1:aifMndx-1],n=1,axis=0) 
    
    aifDif = np.array(aifDif.tolist());
    baseLine2 = [index for index, item in enumerate(aifDif) if item != 0][-1] 
    
    ### constraint (?)
    if baseLine2 > 15:
       baseLine2 = 10; 

    
    return baseLine2 
