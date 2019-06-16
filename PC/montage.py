import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

out_folder = '../my_dataset/Fruit_legume_rgb'
train_folder = '../my_dataset/Fruit_legume_rgbd/'
back_folder = '../my_dataset/table/'
back_obj='table'
obj_category = 'Aubergine'
path = os.getcwd()


n_back= sum(len(files) for _, _, files in 
  os.walk(back_folder))


file_count = sum(len(files) for _, _, files in 
        os.walk(train_folder))

def add_background(frame, depth_image):
  
  background=cv2.imread(back_folder + '/'+ back_obj + 
      str(np.random.randint(1,n_back)) +'.jpg')
  background=cv2.resize(background,(frame.shape[1],frame.shape[0]))

  #depth_image = undistort(depth_image) 
  depht=cv2.convertScaleAbs(depth_image, alpha=0.06)

  th2 = np.zeros((depht.shape[0],depht.shape[1]), dtype=np.uint8)

  # Split image in six regins  ## The pictures were taken in isometrical view, then the background has an increasing deph from bottom to top
  _ , th2[0:81,:] = cv2.threshold(depht[0:81,:] , 230,255,cv2.THRESH_BINARY_INV)
  _ , th2[81:161,:] = cv2.threshold(depht[81:161,:] , 204,255,cv2.THRESH_BINARY_INV)
  _ , th2[161:241,:] = cv2.threshold(depht[161:241,:] , 185,255,cv2.THRESH_BINARY_INV)
  _ , th2[241:321,:] = cv2.threshold(depht[241:321,:] , 164,255,cv2.THRESH_BINARY_INV)
  _ , th2[321:401,:] = cv2.threshold(depht[321:401,:] , 148,255,cv2.THRESH_BINARY_INV)
  _ , th2[401:481,:] = cv2.threshold(depht[401:481,:] , 130,255,cv2.THRESH_BINARY_INV)

  for j in range(0,background.shape[2]):

    background[:,:,j] = cv2.bitwise_and(background[:,:,j],
      cv2.bitwise_not(th2))
    frame[:,:,j]=cv2.bitwise_and(frame[:,:,j],th2)
    frame[:,:,j]=cv2.bitwise_or(frame[:,:,j], background[:,:,j])

  return frame


def main():
	
	file_count = sum(len(files) for _, _, files in 
        os.walk(os.getcwd() + '/' + train_folder + obj_category))

	for i in range(0, file_count):

		img=cv2.imread(train_folder + obj_category +'/' + obj_category +
			str(i)+'.png', cv2.IMREAD_UNCHANGED)
		background=cv2.imread(back_folder + '/'+ back_obj + 
			str(np.random.randint(1,n_back)) +'.jpg')
		background=cv2.resize(background,(img.shape[1],img.shape[0]))

		depth_image=img[:,:,3]
		img=img[:,:,0:3].astype(np.uint8)
		
		img=add_background(img , depth_image)

		# cv2.imwrite(path +'/' + out_folder +'/' + obj_category + 
		# 	'/' + obj_category + str(i) + '.png',
		# 	img[:,:,0:3])

		cv2.namedWindow('montaje', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('montaje', img)
		cv2.waitKey(1)

if __name__ == '__main__':
    main()
