import numpy as np
import os, io, cv2
import time
import matplotlib.pyplot as plt
import scipy.spatial.distance
from scipy import ndimage
from skimage import morphology, filters
from skimage.measure import label
from osgeo import gdal
import skimage.filters.rank as sfr
from sklearn.preprocessing import MinMaxScaler
import multiprocessing as mp
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, cohen_kappa_score
def read_img(filename):
    dataset = gdal.Open(filename) 
    im_width = dataset.RasterXSize 
    im_height = dataset.RasterYSize  
    im_geotrans = dataset.GetGeoTransform() 
    im_proj = dataset.GetProjection() 
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height) 
    del dataset
    return im_proj, im_geotrans, im_data

def write_img(filename, im_proj, im_geotrans, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    
    if len(im_data.shape) == 4:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
   
    driver = gdal.GetDriverByName("GTiff")  
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(im_geotrans)  
    dataset.SetProjection(im_proj) 
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset
Reference = '...\Sample\SZU-RE.tif'
TFile = '...\Sample\SZU-WithNoise&Error.tif'
AP, AG, RRRR = read_img(Reference)
AP, AG, CONE = read_img(TFile)
start_time = time.time()
LAB = RRRR
size = 3
cishu = 1

def EROSION(XXX, soo):
    SSUMM = np.zeros_like(XXX)
    SUMMM = np.ones_like(XXX)
    SUMMM[XXX==0] = 0
    PLUS = np.zeros_like(XXX)
    AAAA = np.unique(XXX)[1:]
    tit = 1
    IDRR = XXX
    while np.sum(XXX):
          kernel = morphology.square(soo)
          EEEE = morphology.erosion(XXX, kernel)
          TIME = np.zeros_like(EEEE)
          TIME[EEEE>0] = 1
          NINDEX = np.unique(TIME*XXX)[1:]
          SSUMM = SSUMM + TIME
          DII = np.setdiff1d(AAAA, NINDEX)
          for d in DII:
              SUMMM[IDRR == d] = 0
          PLUS = PLUS + SUMMM
          XXX = EEEE
          # Detect the Erosion Process Step by Step :)
          # plt.subplot(1, 3, 1)
          # plt.imshow(EEEE)
          # plt.subplot(1, 3, 2)
          # plt.imshow(SUMMM)
          # plt.subplot(1, 3, 3)
          # plt.imshow(PLUS)
          # plt.show()
          tit += 1
          soo += 2 # Change the window size of filter
    NORR = np.divide(SSUMM, PLUS)
    NORR[IDRR == 0] = 0
    return SSUMM, NORR, PLUS

label_image = label(LAB, background=0)
label_image[CONE == 0] = 0
print("Step 1>>>SINGLE PATCHES LABELING COMPLETE>>>")
BEIFEN = label_image * np.ones_like(label_image)
PPAA = morphology.dilation(BEIFEN)
DDFF = np.abs(PPAA-BEIFEN)
IA = BEIFEN * np.ones_like(BEIFEN)
ID = np.zeros_like(BEIFEN)
IA[(BEIFEN > 0) & (DDFF>0)] = 0
print("Step 2>>>NEIGHBOR PATCHES SEGMENTATION>>>")
SODSUMM, SODNOR, COUNT = EROSION(IA, size)
plt.subplot(1, 2, 1)
plt.imshow(SODSUMM)
plt.subplot(1, 2, 2)
plt.imshow(SODNOR)
plt.show()
print("Step 3>>>EROSION NORMALIZE COMPLETE!!!!")
OD = np.zeros_like(BEIFEN)
OD[(BEIFEN > 0) & (IA==0)] = 1
ODD = ndimage.binary_fill_holes(OD).astype('int')
ODD[IA == 0] = 0
SOILD = ODD*BEIFEN
UNSOD = BEIFEN - SOILD
SOILD = BEIFEN - UNSOD
UNSOD[IA == 0] = 0
print("Step 4>>>Embeded and Non-Embeded Patches Extraction")

UNSOD[UNSOD>0] = 1
UNID = UNSOD*label_image
UNDERO = UNSOD*SODNOR
SOILD[SOILD>0] = 1
SOID = SOILD*label_image
SOILDO = SOILD*SODNOR
time_use = format((time.time() - start_time), '.2f')
print(time_use)
write_img('...\Results\\SZU-OUTSUM.tif', AP, AG, UNDERO) # The ENOR of NEMPs
write_img('...\Results\\SZU-EMDSUM.tif', AP, AG, SOILDO) # The ENOR of EMPs
write_img('...\Results\\SZU-ERONOR.tif', AP, AG, SODNOR)
######### Calculate IA
# the number of elements of WEIGHT list = the landcover types in target image segmentation results
def iou_score(gt, pred, num_classes):
    iou = np.zeros(num_classes)
    for c in range(num_classes):
        intersection = np.logical_and(gt == c, pred == c).sum()
        union = np.logical_or(gt == c, pred == c).sum()
        if union == 0:
            iou[c] = float('nan')
        else:
            iou[c] = intersection / union
    return iou
# mIOU Calculation
def miou_score(gt, pred, num_classes):
    iou = iou_score(gt, pred, num_classes)
    m_iou = np.nanmean(iou)
    return m_iou
WEIGHT = [0, 0.03, 0.05, 0.18, 0.18, 0.2, 0.05, 0.05, 0.11, 0.1, 0.05, 0] # Weights of each Land cover type
AA = 0.7   #The weight set for embeded patches
BB = 1 - AA
ACCRR = []
CLASSTYPE = np.unique(LAB)
TOTAI = 0
TRUEIMG = np.zeros_like(LAB)
for tnum in range(len(WEIGHT)):
    print('WEIGHT>>>' + str(WEIGHT[int(tnum)]))
    OOIMG = np.zeros_like(LAB)
    OOIMG[RRRR == tnum] = 1
    RRIMG = np.zeros_like(LAB)
    RRIMG[(RRRR == tnum) & (CONE == tnum)] = 1
    TFEN = np.zeros_like(LAB)
    TFEN[(RRRR == tnum) & (CONE == tnum)] = 1
    TRUEIMG = TRUEIMG + TFEN
    print("Total Number of Pixels>>>>" + str(np.sum(OOIMG)))
    print("ERROR Number of Pixels>>>>" + str(np.sum(OOIMG) - np.sum(TFEN)))
    TSO = RRIMG * SOILDO
    TDO = RRIMG * UNDERO
    SREAL = len(np.argwhere(OOIMG * SOILDO == 1))
    DREAL = len(np.argwhere(OOIMG * UNDERO == 1))
    SONE = len(np.argwhere(TSO == 1))
    DONE = len(np.argwhere(TDO == 1))
    print(SONE, SREAL, DONE, DREAL)
    PP = 1
    if SREAL and DREAL:
        WSO = SONE / SREAL
        WDO = DONE / DREAL
        PP = AA * WSO + BB * WDO
    if DREAL and SREAL == 0:
        WDO = DONE / DREAL
        PP = WDO
    if SREAL and DREAL == 0:
        WSO = SONE / SREAL
        PP = WSO
    TOTAI += WEIGHT[int(tnum)] * PP
IA = np.round(TOTAI, 2)
TFF = RRRR.flatten()
OFF = CONE.flatten()
ORP = round(precision_score(OFF, TFF, average='macro'), 2)
ORR = round(recall_score(OFF, TFF, average='macro'), 2)
ORF = round(f1_score(OFF, TFF, average='macro'), 2)
ORA = round(accuracy_score(OFF, TFF), 2)
ORK = round(cohen_kappa_score(OFF, TFF), 2)
MIOU = round(miou_score(OFF, TFF, len(WEIGHT)), 2)
print('IA>>>' + str(IA) + ">>>mIOU>>>"+str(MIOU)+"  >>>Overall Accuracy>>" + str(ORA) + "  >>Percision>>" + str(ORP) + "  >>Recall>>" + str(ORR)+ "  >>F1-score>>" + str(ORF)+ "  >>Kappa>>" + str(ORK))




