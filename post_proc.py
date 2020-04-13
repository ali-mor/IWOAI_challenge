"""This script do postprocessing and generate the final binary mask"""
import numpy as np 
#from medpy.io import save
import sys
import os
path_img = str(sys.argv[1])
A1 = np.load(path_img[:-3] + '_A1.npy')
A2 = np.load(path_img[:-3] + '_A2.npy')
A3 = np.load(path_img[:-3] + '_A3.npy')
tt = (A1 + A2 + A3)/3.
tt[tt>=0.5] = 1
tt[tt<0.5] = 0
tt2 = np.zeros((384, 384, 160, 4))
tt2[tt[..., 0]>0, 0] = 1
tt3 = tt[..., 1] + tt[..., 2]
tt2[tt3>0, 1] = 1
tt2[tt[..., 3]>0, 2] = 1
tt3 = tt[..., 4] + tt[..., 5]
tt2[tt3>0, 3] = 1
tt2 = np.array(tt2, dtype=np.int8)
#save(tt2, path_img[:-3] + '_bin.nii')
np.save(path_img[:-3] + '_bin.npy',tt2)
os.remove(path_img[:-3] + '_A1.npy')
os.remove(path_img[:-3] + '_A2.npy')
os.remove(path_img[:-3] + '_A3.npy')
