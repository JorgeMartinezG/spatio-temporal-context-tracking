import cv2
import math
import numpy as np
import os
import sys

#def get_initbox(video_name)

if __name__ == '__main__':

    dataset_folder = 'Datasets'
    video_name = sys.argv[1]
    frames_list = os.listdir(os.path.join(dataset_folder, video_name, 'img'))
    frames_list.sort()

    # Get initial rectangle [x, y, width, height].
    initstate = [161, 65, 75, 95]

    # Center target.
    pos = [initstate[1] + initstate[3]/2, initstate[0] + initstate[2]/2]
    # Initial target size.
    target_sz = np.array([initstate[3], initstate[2]])
    
    # Parameters according to the paper.
    padding = 1.0                               # Extra area.
    rho = 0.075                                 # Learning parameter rho.
    sz = target_sz * (1. + padding)             # Context region size.

    # Parameters of scale update - scale ratio, lambda, average frames.
    scale, lambada, num = 1, 0.25, 5
    # Pre-computed confidence map.
    alapha = 2.25
    #rs, cs.
    #dist = rs^2 + cs^2
    conf = math.exp(-0.5 / (alapha) * math.sqrt(642))

    # Confidence map normalization
    #conf =  

    # Loop reading frames
    for frame in frames_list:
        print os.path.join(dataset_folder, video_name, frame)       
        img = cv2.imread(os.path.join(dataset_folder, video_name, 'img', frame))
        cv2.imshow('image', img)
        key = cv2.waitKey(30)
    
        if key & 0xFF == ord('q'):
            break

    print "Loop finished!"
