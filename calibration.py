import cv2
import numpy as np

objp = np.zeros((5*8,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:8].T.reshape(-1,2)

objpoints = []
imgpoints = []

for i in range(1,21):
    img = cv2.imread('Chessboard{}.jpg'.format(i))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (5,8), None)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        #cv2.drawChessboardCorners(img, (5,8), corners2, ret)
        #cv2.imshow('img', img)
        #cv2.waitKey(1000)

#cv2.destroyAllWindows()