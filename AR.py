# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:42:57 2021

@author: devon
"""

import cv2
import numpy as np

#load the predefined dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Generate the markers
n = 4

temp = np.zeros((200,200), dtype=np.uint8)
markerImage = [temp]*n

for i in range(0,n):
    picture_number = i
    temp = np.zeros((200,200), dtype=np.uint8)
    markerImage[i] = cv2.aruco.drawMarker(dictionary,picture_number,200,temp,1)

# #print images
# cv2.imshow('marker 1', markerImage[0])
# cv2.imshow('marker 2', markerImage[1])
# cv2.imshow('marker 3', markerImage[2])
# cv2.imshow('marker 4', markerImage[3])

# #save images
# cv2.imwrite("marker0.jpg", markerImage[0])
# cv2.imwrite("marker1.jpg", markerImage[1])
# cv2.imwrite("marker2.jpg", markerImage[2])
# cv2.imwrite("marker3.jpg", markerImage[3])

#import mask for test purposes
mask = cv2.imread('test_img_mask.jpg')
im_src = mask

cap = cv2.VideoCapture(0)

while True: 
    #obtain camera image
    
    # # used for testing
    # img = cv2.imread('test_img.jpg')
    
    ret0, img = cap.read()                                                                                                                          
    
    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()
    
    #detect the markers in the image
    markerCorners, markerIds, rejecteCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)
    
    # put rectangles on markers. For testing purposes. 
    # for i in range(0, n):
    #     (x, y, w, h) = cv2.boundingRect(markerCorners[i])
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    im_out = img
    
    for k in range(0, 4):
        #calculate Homography
        index = np.squeeze(np.where(markerIds==k))
        refPt1 = np.squeeze(markerCorners[index[0]])[2]
        
        index = np.squeeze(np.where(markerIds==k))
        refPt2 = np.squeeze(markerCorners[index[0]])[3]
        
        distance = np.linalg.norm(refPt1-refPt2)
        
        scalingFac = 0.02
        pts_dst = [[refPt1[0] - round(scalingFac*distance), refPt1[1] - round(scalingFac*distance)]]
        pts_dst = pts_dst + [[refPt2[0] + round(scalingFac*distance), refPt2[1] - round(scalingFac*distance)]]
        
        index = np.squeeze(np.where(markerIds==k));
        refPt3 = np.squeeze(markerCorners[index[0]])[0];
        pts_dst = pts_dst + [[refPt3[0] + round(scalingFac*distance), refPt3[1] + round(scalingFac*distance)]];
        
        index = np.squeeze(np.where(markerIds==k));
        refPt4 = np.squeeze(markerCorners[index[0]])[1];
        pts_dst = pts_dst + [[refPt4[0] - round(scalingFac*distance), refPt4[1] + round(scalingFac*distance)]];
        
        pts_src = [[0,0], [im_src.shape[1], 0], [im_src.shape[1], im_src.shape[0]], [0, im_src.shape[0]]];
        
        pts_src_m = np.asarray(pts_src)
        pts_dst_m = np.asarray(pts_dst)
        
        h, status = cv2.findHomography(pts_src_m, pts_dst_m)
        
        # Warp source image to destination based on homography
        warped_image = cv2.warpPerspective(im_src, h, (img.shape[1],img.shape[0]))
        
        # Prepare a mask representing region to copy from the warped image into the original frame.
        mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8);
        cv2.fillConvexPoly(mask, np.int32([pts_dst_m]), (255, 255, 255), cv2.LINE_AA);
        
        # Erode the mask to not copy the boundary effects from the warping
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3));
        mask = cv2.erode(mask, element, iterations=3);
        
        # Copy the mask into 3 channels.
        warped_image = warped_image.astype(float)
        mask3 = np.zeros_like(warped_image)
        for i in range(0, 3):
            mask3[:,:,i] = mask/255
        
        # Copy the warped image into the original frame in the mask region.
        warped_image_masked = cv2.multiply(warped_image, mask3)
        frame_masked = cv2.multiply(im_out.astype(float), 1-mask3)
        im_out = cv2.add(warped_image_masked, frame_masked)
    
    # Showing the original image and the new output image side by side
    concatenatedOutput = cv2.hconcat([img.astype(float), im_out]);
    cv2.imshow("AR using Aruco markers", concatenatedOutput.astype(np.uint8))
    
    # cv2.imshow('img', im_src)
    
    # while True:
    #     #print("got into while loop")
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         #print("read q")
    #         break
    
    key = cv2.waitKey(100)
    if key == ord('q'):
        break

cv2.destroyAllWindows()