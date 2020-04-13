import h5py    
import numpy as np 
import SimpleITK as sitk
from medpy import filter
from medpy.io import save
import os

def hist_matcher(ref,source):
	hist_match = sitk.HistogramMatchingImageFilter()
	hist_match.SetNumberOfHistogramLevels( 500)
	hist_match.SetNumberOfMatchPoints( 100 )
	image_hist_match = hist_match.Execute ( source,ref )
	return image_hist_match
	
def main():
	Num_training = 60
	directory = "train_npy/"
	if not os.path.exists(directory):
		os.makedirs(directory)
	for i in range(Num_training):  
		print(i) 
		img = h5py.File('train/train_'+str(i+1).zfill(3)+'_V00.im','r+')
		img = np.array(img['data'])
		img = filter.smoothing.anisotropic_diffusion(img, niter=4, kappa=50, gamma=0.1)
		if i==0:
			ref = sitk.GetImageFromArray(img)
		img = sitk.GetImageFromArray(img)
		img = hist_matcher(ref, img)
		img = sitk.GetArrayFromImage(img)
		img = (img-np.min(img))/(np.max(img)-np.min(img))
		np.save('train_npy/train_'+str(i+1).zfill(3)+'_V00.npy', img)
		gt = h5py.File('train/train_'+str(i+1).zfill(3)+'_V00.seg','r+')
		gt = np.array(gt['data'], dtype=np.int8)
		np.save('train_npy/train_'+str(i+1).zfill(3)+'_V00_seg.npy', gt)
		img = h5py.File('train/train_'+str(i+1).zfill(3)+'_V01.im','r+')
		img = np.array(img['data'])
		img = filter.smoothing.anisotropic_diffusion(img, niter=4, kappa=50, gamma=0.1)
		img = sitk.GetImageFromArray(img)
		img = hist_matcher(ref, img)
		img = sitk.GetArrayFromImage(img)
		img = (img-np.min(img))/(np.max(img)-np.min(img))
		np.save('train_npy/train_'+str(i+1).zfill(3)+'_V01.npy', img)
		gt = h5py.File('train/train_'+str(i+1).zfill(3)+'_V01.seg','r+')
		gt = np.array(gt['data'], dtype=np.int8)
		np.save('train_npy/train_'+str(i+1).zfill(3)+'_V01_seg.npy', gt)   

	Num_valid = 14
	directory = "valid_npy/"
	if not os.path.exists(directory):
		os.makedirs(directory)
	for i in range(Num_valid):  
		print(i) 
		img = h5py.File('valid/valid_'+str(i+1).zfill(3)+'_V00.im','r+')
		img = np.array(img['data'])
		img = filter.smoothing.anisotropic_diffusion(img, niter=4, kappa=50, gamma=0.1)
		img = sitk.GetImageFromArray(img)
		img = hist_matcher(ref, img)
		img = sitk.GetArrayFromImage(img)
		img = (img-np.min(img))/(np.max(img)-np.min(img))
		np.save('valid_npy/valid_'+str(i+1).zfill(3)+'_V00.npy', img)
		gt = h5py.File('valid/valid_'+str(i+1).zfill(3)+'_V00.seg','r+')
		gt = np.array(gt['data'], dtype=np.int8)
		np.save('valid_npy/valid_'+str(i+1).zfill(3)+'_V00_seg.npy', gt)
		img = h5py.File('valid/valid_'+str(i+1).zfill(3)+'_V01.im','r+')
		img = np.array(img['data'])
		img = filter.smoothing.anisotropic_diffusion(img, niter=4, kappa=50, gamma=0.1)
		img = sitk.GetImageFromArray(img)
		img = hist_matcher(ref, img)
		img = sitk.GetArrayFromImage(img)
		img = (img-np.min(img))/(np.max(img)-np.min(img))
		np.save('valid_npy/valid_'+str(i+1).zfill(3)+'_V01.npy', img)
		gt = h5py.File('valid/valid_'+str(i+1).zfill(3)+'_V01.seg','r+')
		gt = np.array(gt['data'], dtype=np.int8)
		np.save('valid_npy/valid_'+str(i+1).zfill(3)+'_V01_seg.npy', gt)
main()
