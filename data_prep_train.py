""" This scripts prepare data for each plane """
""" It is called with train.py """
import numpy as np 
def add_bg(img):
	img2 = np.sum(img, axis=3, keepdims=True)
	img2[img2>0] = 5
	img2[img2==0] = 1
	img2[img2==5] = 0
	img = np.concatenate((img2, img), axis=3)
	img = np.array(img, dtype=np.int8)
	return img
def swap_A1(image):
    """ Swap dimensions for plane A1"""
    image = np.swapaxes(image, 2, 1)
    image = np.swapaxes(image, 0, 1)
    return image
def swap_A2(image):
    """ Swap dimensions for plane A2"""
    image = np.swapaxes(image, 0, 1)
    return image

def main(path_data, plane):
    """ main function for data preparation
    input image of shape [d1, d2, d3]:
    Returns:
        For plane A1- img_tr_all with shape of [Num_trainingxd3, d2, d1, 1]
                      gt_tr_all with shape of [Num_trainingxd3, d2, d1, 7]
        For plane A2- img_tr_all with shape of [Num_trainingxd2, d3, d1, 1]
                      gt_tr_all with shape of [Num_trainingxd2, d3, d1, 7]
        For plane A3- img_tr_all with shape of [Num_trainingxd1, d2, d3, 1]
                      gt_tr_all with shape of [Num_trainingxd1, d2, d3, 7]
        img_va_all as dict{}
        gt_va_all as dict{}
    """
    path_tr = path_data + '/train_npy/'
    path_va = path_data + '/valid_npy/'
    if plane=='A1': axis_n = 2
    if plane=='A2': axis_n = 1
    if plane=='A3': axis_n = 0
    Num_training = 60
    for i in range(Num_training):  
        print(i) 
        img_tr = np.load(path_tr+'/train_'+str(i+1).zfill(3)+'_V00.npy')
        if i==0:
            img_tr_all = img_tr
        else:
            img_tr_all = np.concatenate((img_tr_all, img_tr), axis=axis_n)
        img_tr = np.load(path_tr+'/train_'+str(i+1).zfill(3)+'_V01.npy')
        img_tr_all = np.concatenate((img_tr_all, img_tr), axis=axis_n)
        
        gt_tr = np.load(path_tr+'/train_'+str(i+1).zfill(3)+'_V00_seg.npy')
        gt_tr = add_bg(gt_tr)
        if i==0:
            gt_tr_all = gt_tr
        else:
            gt_tr_all = np.concatenate((gt_tr_all, gt_tr), axis=axis_n)
        gt_tr = np.load(path_tr+'/train_'+str(i+1).zfill(3)+'_V01_seg.npy')
        gt_tr = add_bg(gt_tr)
        gt_tr_all = np.concatenate((gt_tr_all, gt_tr), axis=axis_n)
    #####################################
    if plane=='A1': 
        img_tr_all = swap_A1(img_tr_all)
        gt_tr_all = swap_A1(gt_tr_all)
    if plane=='A2': 
        img_tr_all = swap_A2(img_tr_all)
        gt_tr_all = swap_A2(gt_tr_all)
    img_tr_all = np.expand_dims(img_tr_all, 3)
    print(img_tr_all.shape)
    print(gt_tr_all.shape)
    #####################################	
    Num_valid = 14
    img_va_all = {}
    gt_va_all = {}
    for i in range(Num_valid):  
        print(i) 
        img_va = np.load(path_va+'/valid_'+str(i+1).zfill(3)+'_V00.npy')
        if plane=='A1': img_va = swap_A1(img_va)
        if plane=='A2': img_va = swap_A2(img_va)
        img_va_all[str(i+1)+'_00'] = np.expand_dims(img_va, 3)
        img_va = np.load(path_va+'/valid_'+str(i+1).zfill(3)+'_V01.npy')
        if plane=='A1': img_va = swap_A1(img_va)
        if plane=='A2': img_va = swap_A2(img_va)
        img_va_all[str(i+1)+'_01'] = np.expand_dims(img_va, 3)
        
        gt_va = np.load(path_va+'/valid_'+str(i+1).zfill(3)+'_V00_seg.npy')
        if plane=='A1': gt_va = swap_A1(gt_va)
        if plane=='A2': gt_va = swap_A2(gt_va)
        gt_va_all[str(i+1)+'_00'] = add_bg(gt_va)
        gt_va = np.load(path_va+'/valid_'+str(i+1).zfill(3)+'_V01_seg.npy')
        if plane=='A1': gt_va = swap_A1(gt_va)
        if plane=='A2': gt_va = swap_A2(gt_va)
        gt_va_all[str(i+1)+'_01'] = add_bg(gt_va)
    print(img_va_all['1_00'].shape)
    print(gt_va_all['1_00'].shape)
    return img_tr_all, gt_tr_all, img_va_all, gt_va_all
