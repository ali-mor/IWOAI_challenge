"""This script do pre-processing for the input image and return """
""" a prepared image for network for each plane"""
import h5py    
import numpy as np 
import SimpleITK as sitk

def swap_A1(image):
    """ Swap dimensions for plane A1"""
    image = np.swapaxes(image, 2, 1)
    image = np.swapaxes(image, 0, 1)
    return image
    
def swap_A2(image):
    """ Swap dimensions for plane A2"""
    image = np.swapaxes(image, 0, 1)
    return image

def hist_matcher(ref, source):
    """Apply histogram matching to input image"""
    hist_match = sitk.HistogramMatchingImageFilter()
    hist_match.SetNumberOfHistogramLevels(500)
    hist_match.SetNumberOfMatchPoints(100)
    image_hist_match = hist_match.Execute (source, ref)
    return image_hist_match

def main(path_img, plane):
    """ main function for data preparation
    input image of shape [d1, d2, d3]:
    Returns:
        For plane A1- image with shape of [d3, d2, d1, 1]
        For plane A2- image with shape of [d2, d3, d1, 1]
        For plane A3- image with shape of [d1, d2, d3, 1]
    """
    ######### Reading ref. image for histogram matching #########
    ref = h5py.File('ref_img/ref_img.im', 'r+')
    ref = np.array(ref['data'])
    ref = filter.smoothing.anisotropic_diffusion(ref, niter=4, kappa=50, gamma=0.1)
    ref = sitk.GetImageFromArray(ref)
    ######### Apply post processign steps to given image #########
    img = h5py.File(path_img, 'r+')
    img = np.array(img['data'])
    img = filter.smoothing.anisotropic_diffusion(img, niter=4, kappa=50, gamma=0.1)
    img = sitk.GetImageFromArray(img)
    img = hist_matcher(ref, img)
    img = sitk.GetArrayFromImage(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    ####### Prepare the input image for specific plane ########
    if plane=='A1':
        img = swap_A1(img)
    if plane=='A2':
        img = swap_A2(img)
    img = np.expand_dims(img, 3)
    #####################################
    return img
