# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:42:57 2021

@author: devon
"""

import cv2
import numpy as np
from objloader_simple import *
import os
from tqdm import tqdm

def render(img, obj, r, t, mtx, dist, color=False):
    vertices = obj.vertices
    # offset to middle of card
    h, w = 0, 0

    for face in tqdm(obj.faces):
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])

        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])

        imgpts, jac = cv2.projectPoints(points, r, t, mtx, dist)
        imgpts = np.int32(imgpts)

        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1] # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

obj = OBJ('models/fox.obj', swapyz=True)  

#load the predefined dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

# load params
params = np.load('models/logitech_c270.npz')
camera_matrix = params['mtx']
dist_coeffs = params['dist']

cap = cv2.VideoCapture(2)

while True: 
    #obtain camera image
    ret0, img = cap.read()                                                                                                                          
    # img = cv2.imread('test_img.jpg')
        
    #detect the markers in the image
    markerCorners, markerIds, rejecteCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

    if markerIds is not None:
        for corners, id in zip(markerCorners, markerIds):
            r, t, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 50, camera_matrix, dist_coeffs)
            r = cv2.Rodrigues(r)[0]
            # cv2.aruco.drawAxis(img, camera_matrix, dist_coeffs, r, t, 0.4)

            img = render(img, obj, r, t, camera_matrix, dist_coeffs)
    
    # Showing the original image and the new output image side by side
    cv2.imshow("AR using Aruco markers", img)

    key = cv2.waitKey(20)
    if key == ord('q'):
        break

cv2.destroyAllWindows()