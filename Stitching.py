import cv2
import numpy as np
from calibration import objpoints, imgpoints
import random


#Part 1: Undistort image
img1 = cv2.imread('Panorama1.jpg')
img2 = cv2.imread('Panorama2.jpg')
size = (img1.shape[1], img1.shape[0])
camera_matrix, dist_coeffs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)[1:3]
rect_camera_matrix = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, size, 0)[0]
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3), rect_camera_matrix, size, cv2.CV_32FC1)

rect_img1 = cv2.remap(img1, map1, map2, cv2.INTER_LINEAR)
cv2.imwrite("Undistorted1.jpg", rect_img1)
#cv2.imshow('imgs',cv2.hconcat([img1, rect_img1]))
#cv2.waitKey(500)

rect_img2 = cv2.remap(img2, map1, map2, cv2.INTER_LINEAR)
cv2.imwrite("Undistorted2.jpg", rect_img2)
#cv2.imshow('imgs',cv2.hconcat([img2, rect_img2]))
#cv2.waitKey(500)

#Part 2:
def apply_transformation_to_pixel(x,y,H):
    pom = np.dot(H, np.array([x, y, 1]))
    xd = pom[0]/pom[2]
    yd = pom[1]/pom[2]
    return round(xd), round(yd)

def apply_transformation_to_point(x,y,H):
    pom = np.dot(H, np.array([x, y, 1]))
    xd = pom[0]/pom[2]
    yd = pom[1]/pom[2]
    return xd, yd

def transformation(img, H):
    H_1 = np.linalg.inv(H)
    h_img, w_img = img.shape[0:2]
    corners_dest = []
    corners_dest.append(apply_transformation_to_pixel(0,0,H))
    corners_dest.append(apply_transformation_to_pixel(w_img,0,H))
    corners_dest.append(apply_transformation_to_pixel(w_img, h_img, H))
    corners_dest.append(apply_transformation_to_pixel(0, h_img, H))
    min_x = min(round(corners_dest[0][0]), round(corners_dest[3][0]))
    min_y = min(round(corners_dest[0][1]), round(corners_dest[1][1]))
    max_x = max(round(corners_dest[1][0]), round(corners_dest[2][0]))
    max_y = max(round(corners_dest[3][1]), round(corners_dest[2][1]))
    img_dest = np.zeros((max_y-min_y+1, max_x-min_x+1, 3), dtype=np.uint8)
    for i in range(img_dest.shape[0]):
        for j in range(img_dest.shape[1]):
            x_pom, y_pom = apply_transformation_to_pixel(j+min_x, i+min_y, H_1)
            if (x_pom >= 0)&(x_pom < w_img)&(y_pom >= 0)&(y_pom < h_img):
                img_dest[i][j] = img[y_pom][x_pom]
    return img_dest, (min_x, min_y, max_x, max_y)


#Part 3:
def equation(point, point_dest):
    return np.array([point[0], point[1], 1, 0, 0, 0, -point[0]*point_dest[0], -point_dest[0]*point[1], -point_dest[0]]),\
           np.array([0, 0, 0, point[0], point[1], 1, -point[0]*point_dest[1], -point_dest[1]*point[1], -point_dest[1]])

def homography(points, points_dest):
    A = []
    for i in range(len(points)):
        A.append(equation(points[i],points_dest[i])[0])
        A.append(equation(points[i],points_dest[i])[1])
    _, _, V = np.linalg.svd(A)
    eingenvector = V[-1, :]
    return np.array([eingenvector[0:3], eingenvector[3:6], eingenvector[6:]])

def test_homography():
    for i in range(10):
        H = np.random.rand(3,3)
        points = np.random.rand(4,2)
        points_dest = [np.array([apply_transformation_to_point(point[0],point[1],H)[0],\
                                apply_transformation_to_point(point[0],point[1],H)[1]]) for point in points]
        H_est = homography(points, points_dest)
        H_squares = H**2
        H_norm = H/np.sqrt(H_squares.sum())
        print(((H_norm.round(4)==H_est.round(4)).all()) | ((H_norm.round(4)==-H_est.round(4)).all()))


#Part 4:
points_dest = np.array([[741,344],[549,404],[750,68],[655,130],[500,586],[484,393]]) #points from Panorama1
points = np.array([[358,320],[176,386],[364,63],[283,111],[139,578],[118,381]]) #points from Panorama2
H = homography(points, points_dest)  # homography from Panorama2 to Panorama


#Part 5:
def stitch(img1,img2,H,name):
    transformed_img2, (min_x, min_y, max_x, max_y) = transformation(img2, H)
    #cv2.imshow('img',transformed_img2)
    #cv2.waitKey(500)
    y, x = img1.shape[:2]
    size_final = (max(max_x + 1, max_x - min_x + 1, x, x - min_x + 1), max(max_y + 1, max_y - min_y + 1, y, y - min_y + 1))
    img_final = np.zeros((size_final[1], size_final[0], 3), dtype=np.uint8)
    left1 = max(0, -min_x)
    upper1 = max(0, -min_y)
    left2 = max(0, min_x)
    upper2 = max(0, min_y)
    img_final[upper2:(max_y - min_y + 1 + upper2), left2:(max_x + 1 - min_x + left2), :] = transformed_img2
    for i in range(y):
        for j in range(x):
            if (img_final[i + upper1, j + left1, :] == np.zeros(3)).all():
                img_final[i + upper1, j + left1, :] = img1[i, j, :]
            else:
                weigth = 1 - (1 - abs(2 * j - x)/x)**2
                img_final[i + upper1, j + left1, :] = weigth * img_final[i + upper1, j + left1, :] + (
                            1 - weigth) * img1[i, j, :]
    cv2.imshow('img', img_final)
    cv2.waitKey(5000)
    cv2.imwrite(name, img_final)

stitch(rect_img1,rect_img2,H,"Stitched.jpg")


#Part 6:
def get_matches(img1, img2, visualize=False, lowe_ratio=0.6):
    # Find the keypoints and descriptors with SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # Ratio test as per Lowe's paper
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < lowe_ratio * n.distance:
            matchesMask[i] = [1, 0]
            good_matches.append((kp1[m.queryIdx].pt, kp2[m.trainIdx].pt))

    if visualize:
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(0, 0, 255),
            matchesMask=matchesMask,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        cv2.imshow("vis", img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return good_matches

#matches = get_matches(rect_img1,rect_img2)


#Part 7:
def stitch_with_RANSAC(img1, img2):
    matches = get_matches(img1,img2)
    best_score = 0
    best_H = None
    for _ in range(500):
        score = 0
        sample = random.sample(matches, 4)
        points = [pair[1] for pair in sample]
        points_dest = [pair[0] for pair in sample]
        H = homography(points, points_dest)
        for pair in matches:
            estimated_x, estimated_y = apply_transformation_to_point(pair[1][0], pair[1][1], H)
            if (abs(estimated_x - pair[0][0]) < 1) & (abs(estimated_y - pair[0][1]) < 1):
                score += 1
        if score > best_score:
            best_score = score
            best_H = H
    stitch(img1, img2, best_H, "Stitched_with_RANSAC.jpg")

stitch_with_RANSAC(rect_img1,rect_img2)