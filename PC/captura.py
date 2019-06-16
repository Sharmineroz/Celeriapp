import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys

# Working on SR300 realsense camera

# Parameters for fisheye calibration
DIM = (640,480)
K = np.array([[475.53765869140625, 0.0, 309.2130126953125], 
            [0.0, 475.5375061035156, 245.9634552001953], 
            [0.0, 0.0, 1.0]]) 

D = np.array([[0.14829276502132416], 
            [0.06511270254850388], 
            [0.004560006316751242],
            [0.0025975550524890423]])

# Camera width, height, framerate
camx, camy, fps = 640, 480, 30 

# Current dir path 
path = os.getcwd()

def undistort(depth_img):

    h,w = depth_img.shape[:2]

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    # Remove fisheye distortion
    undistorted_img = cv2.remap(depth_img, map1, map2, 
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
    # Remove focal opening distortion
    undistorted_img = cv2.resize(undistorted_img[56:420,44:540],(camx,camy))

    return(undistorted_img)

def make_RGBD(color_image,depth_image):

    capture=np.zeros((depth_image.shape[0],depth_image.shape[1],4),
                             dtype = np.uint16)

    b,g,r = cv2.split(color_image)

    b = b.astype(np.uint16)
    g = g.astype(np.uint16)
    r = r.astype(np.uint16)

    depth_image = undistort(depth_image)

    capture = cv2.merge((b,g,r,depth_image))

    return capture

def create_dir(train_folder, obj_category):

    if not os.path.exists(path + '/' + train_folder
        + obj_category):
        print('Creating folder')
        os.makedirs(train_folder + obj_category)
    else:
        print('Adding pics to class')


def main():

    #Write folder names without accentns or special characters,
    #cv2.imwrite() does not accept it 
    train_folder = '../my_dataset/Fruit_legume_rgbd'
    obj_category = '/thing'

    #Flag
    make_pics = False

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    #realsense camera configuration
    config.enable_stream(rs.stream.depth, camx, camy, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, camx, camy, rs.format.bgr8, fps)
    # Start streaming
    pipeline.start(config)

    #Number of pictures per class to take (how_many_pics)
    how_many_pics=1
    n_pics = 0
    
    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            capture=make_RGBD(color_image , depth_image)

            #shown images, color image lef side, color maped depht image right side
            to_show=np.hstack((capture[:,:,0:3],
                              cv2.applyColorMap(cv2.convertScaleAbs(capture[:,:,3], alpha=0.07),
                              cv2.COLORMAP_JET))).astype(np.uint8)
            cv2.circle(to_show,(int(camx/2),int(camy/2)), 10, (0,0,255), 2)
            
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense',to_show)
            cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                make_pics=True
                sys.stdout.write('\n')

            if cv2.waitKey(1) & 0xFF == ord('w'):
                cv2.destroyAllWindows()
                break    

            if n_pics < how_many_pics:

                if make_pics:
                    # Ensure directory exists 
                    if n_pics == 0:
                        create_dir(train_folder,obj_category)

                    #Save RGBD image
                    cv2.imwrite(os.path.join(path , train_folder + 
                        obj_category + obj_category + str(n_pics) + '.png'),
                         capture)  

                    sys.stdout.write('\rNumber of pics: '+ str(n_pics))

                    n_pics = sum(len(files) for _, _, files in 
                        os.walk(os.getcwd() + '/' + train_folder + obj_category))
                    
                    ## Pause capturing to reorganize the object
                    if n_pics % 200 == 0:
                        make_pics = False
                        sys.stdout.write('\n')

                    if n_pics == how_many_pics-1:
                        break
                else:
                    if n_pics > 0:
                        sys.stdout.write('\r('+ str(n_pics) + ') Change the object position then press q')
                        
                        

                       
    finally:

        # Stop streaming
        pipeline.stop()

if __name__ == '__main__':
    main()
    