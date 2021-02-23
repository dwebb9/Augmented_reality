# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:42:57 2021

@author: devon, easton
"""

import cv2
import numpy as np
from objloader_simple import *
import argparse

def save_markers(dictionary, n):
    for i in range(n):
        img = cv2.aruco.drawMarker(dictionary, i, 200)
        cv2.imwrite(f'markers/marker{i}.jpg', img)

def render(img, obj, r, t, mtx, dist, offset, color=(137, 27, 211)): 
    vertices = obj.vertices
    h, w = offset

    for face in (obj.faces):
        # get points from obj file, shift them as needed
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])

        # project them onto the image
        imgpts, jac = cv2.projectPoints(points, r, t, mtx, dist)
        imgpts = np.int32(imgpts)

        # fill each polygon
        cv2.fillConvexPoly(img, imgpts, color)

    return img

def main(big, idx, params, video):
    # clarify between small and big versions
    if big:
        size = 45
        offset = 0, -15
    else:
        size = 45/3
        offset = 25, -15+50

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
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters_create()
    # these parameters seemed to help with detection
    parameters.adaptiveThreshWinSizeMax = 30
    parameters.adaptiveThreshWinSizeStep = 2
    parameters.adaptiveThreshConstant = 10
    parameters.minDistanceToBorder = 0

    # load params
    params = np.load(params)
    camera_matrix = params['mtx']
    dist_coeffs = params['dist']

    # setup camera and video writer
    cap = cv2.VideoCapture(idx)
    height, width, _ = cap.read()[1].shape
    vout = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while True: 
        #obtain camera image
        ret0, img = cap.read()                                                                                                                          

        #detect the markers in the image
        markerCorners, markerIds, rejecteCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

        if markerIds is not None:
            for corners, id in zip(markerCorners, markerIds.flatten()):
                if id < 4:
                    # estimate pose for each marker
                    r, t, _ = cv2.aruco.estimatePoseSingleMarkers(corners, size, camera_matrix, dist_coeffs)

                    # put stuff on image
                    img = render(img, objs[id], r, t, camera_matrix, dist_coeffs, offset, color=colors[id])
        
        # Showing the original image and the new output image side by side
        cv2.imshow("AR using Aruco markers", img)
        vout.write(img)

        key = cv2.waitKey(20)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file", default='ran.avi')
    ap.add_argument("-b", "--big", action="store_true", help="If big cards should be used")
    ap.add_argument("-i", "--idx", type=int, default=0, help="Index of video camera")
    ap.add_argument("-p", "--params", help="path to the instrinsic params", default='params/logitech_c270.npz')
    args = vars(ap.parse_args())
    main(**args)