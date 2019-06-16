import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

nro= 0.5  ## height or width factoy to try minimize black field when rotating images
height = int((1+nro) * 480)
width = int((1+nro) * 640)

folder='../my_dataset/Fruit_legume_rgb/'
obj_name='Aubergine'

#Number of images in folder
file_count = sum(len(files) for _, _, files in 
	os.walk(os.getcwd() + '/' + folder + obj_name))

#rotation matrixes
aug_factor=8   ## number of output images per input image 
rot_m=np.zeros((2,3,aug_factor),dtype=np.float)
a_step=360/aug_factor

for i in range (1,aug_factor):

	rot_m[:,:,i-1]=cv2.getRotationMatrix2D((int(width / 2) , 
		int(height / 2)), i * a_step , 1)

for i in range(0, file_count):

	img=cv2.imread(folder + obj_name + '/' + obj_name + str(i) + '.png',
		cv2.IMREAD_UNCHANGED)
	ar=np.array(img[:,:,0:3],dtype=np.uint8)

	pad_ar = np.zeros((width,height), dtype = np.uint8)
	top = bottom = int(640* nro/ 2)
	right = left = int(480* nro/ 2)
	pad_ar = cv2.copyMakeBorder(ar, top , bottom , left , right ,
		cv2.BORDER_REPLICATE)


	for j in range(0,rot_m.shape[2]):
		img_aug= cv2.warpAffine(pad_ar,rot_m[:,:,j],(width , height))
		# cv2.imwrite(folder + obj_name + '/' + obj_name 
		# 	+ str(i) +'_'+str(j) + '.png', img_aug)
		img_aug = img_aug[right: right+ 480, top: top+ 640]
		cv2.namedWindow('Olo1', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('Olo1',img_aug)
		cv2.waitKey(1)

	cv2.namedWindow('Olo', cv2.WINDOW_AUTOSIZE)
	cv2.imshow('Olo', ar)
	cv2.waitKey(1)

cv2.destroyAllWindows()

