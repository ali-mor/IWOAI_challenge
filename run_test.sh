##### bash shell to generate a binary npy file for the input image
##### The binary file will be generated in same path as test file
##### "S1" is a GPU number. As an example: 1 
##### "S2" is a path+file name with "im" suffix. Asa an example: /home/usr/test.im
python test.py "$1" "$2" A1
python test.py "$1" "$2" A2
python test.py "$1" "$2" A3
python post_proc.py "$2"

