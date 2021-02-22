# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:42:57 2021

@author: devon, easton
"""

import cv2
import numpy as np
from objloader_simple import *

def render(img, obj, r, t, mtx, dist, color=False):
    vertices = obj.vertices
    # offset to middle of card
    h, w = 0, 0

    for face in (obj.faces):
        # get points from obj file, shift them as needed
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])

        # project them onto the image
        imgpts, jac = cv2.projectPoints(points, r, t, mtx, dist)
        imgpts = np.int32(imgpts)

        # fill each polygon
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            print(color)
            cv2.fillConvexPoly(img, imgpts, color)

    return img

# load all models and set color of each
objs = [OBJ('models/squirtle.obj'),
        OBJ('models/pikachu.obj'),
        OBJ('models/charmander.obj'),
        OBJ('models/bulbasaur.obj')]
colors = [(91,163,181),
            (244,220,38),
            (229,56,0),
            (91,143,99)]
colors = [(c[2], c[1], c[0]) for c in colors]

#load the predefined dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

# load params
params = np.load('params/logitech_c270.npz')
camera_matrix = params['mtx']
dist_coeffs = params['dist']

cap = cv2.VideoCapture(2)

while True: 
    #obtain camera image
    ret0, img = cap.read()                                                                                                                          
        
    #detect the markers in the image
    markerCorners, markerIds, rejecteCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

    if markerIds is not None:
        for corners, id in zip(markerCorners, markerIds.flatten()):
            # estimate pose for each marker
            r, t, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 50, camera_matrix, dist_coeffs)
            r = cv2.Rodrigues(r)[0]

            # put stuff on image
            img = render(img, objs[id], r, t, camera_matrix, dist_coeffs, color=colors[id])
    
    # Showing the original image and the new output image side by side
    cv2.imshow("AR using Aruco markers", img)

    key = cv2.waitKey(20)
    if key == ord('q'):
        break

cv2.destroyAllWindows()