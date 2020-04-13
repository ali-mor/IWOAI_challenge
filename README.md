Readme
All the codes are implemented and tested in Ubuntu 18.04 LTS. The deep learning part is implemented using Tensorflow 1.11 in Nvidia GeForce GTX 1080 GPU. The other Python 2.7 libraries used are as follows: Numpy 1.13.3, h5py 2.8, sys, os, random, SimpleITK. 
All the codes and the folders (models and ref_img) should be in the same directory before running the codes. 

Running the code in test mode:
The bach shell called run_test.sh can be used to run the method for a given image. It takes two arguments as inputs: first one is GPU number (for a system with multi-GPUs), and second one is path and name of the test image. For example following command is running the method in GPU 1 for the image in /home/usr/test.im from Ubuntu terminal:
./run_test_.sh 1  /home/usr/test.im
The input image format file should be ‘im’. Then it will generate the npy binary file in the same directory as the input image. This npy file is the same format as challenge asked for.   

Running the code in train mode:
For running code in train mode the train.py code should be used. This code gets four input arguments as follows: 1) GPU number, 2) A directory to save training outputs (such as models and etc) 3) A path to training data 4) the plane for training 
Arguments 1 and 2 are clear. 
The third argument is a path to the directory which includes the training folder (named ‘training_npy’) and validation folder (named ‘validation_npy’). These two folders can be generated by running “conv_to_npy_train.py” code in the same folder as the training folder (the folder includes ‘im’ files). “conv_to_npy_train.py” will consider 60 of the images as training and 14 as validation and convert ‘im’ files to ‘npy’. Then, it will create the ‘training_npy’ and ‘validation_npy’ folders.  
The fourth argument is specifying which network should be trained and it can be one of these three inputs: ‘A1’ (for axial plane), ‘A2’ (for sagittal plane), and ‘A3’ (for coronal plane). By choosing each of these plane, the corresponding network will be trained.

