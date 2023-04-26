# ImageStitching
In this project, I show how to combine images using point matching algorithms and algebraic transformations

Part 1:
I took two photos ("Panorama1.jpg" and "Panorama2.jpg") and undistorted them (saved as
"Undistorted1.jpg" and "Undistorted2.jpg"). To undistort them I needed camera calibration,
so I also upload "calibration.py". If you want to see comparison of Panoramas before and after undistortion, you and uncomment
lines on the bottom.

Part 2:
The main part of this task is function transformation(), which takes image and matrix and
returns transformed image and information about its placement on the destination plane.

Part 3:
Function homography() finds homography from points to points_dest. You can test this
function, by using test_homography(), which generates random 3x3 matrix and 4 points,
then transform this points and compare estimated homography to the real one. This test is
repeated 10 times.

Part 4:
I found 6 pairs of points and used them to calculate homography from Panorama2 to Panorama1

Part 5:
Function stitch() takes two images, homography from second image to the first one and the
name of the file where the result will be saved. I saved my stitched photo as "Stitched.jpg"

Part 6:
I modified this function to take images (after imread), not files. You can see pairs of
matching points, by uncommenting the line on the bottom and changing parameter visualize
to True.

Part 7:
Function stitch_with_RANSAC() takes two images, stitch them using RANSAC on pairs of points
from get_matches() and save result as "Stitched_with_RANSAC.jpg".
