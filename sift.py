import cv2
import numpy as np
from sift_detection import sift_detector

image1_path = "BestSenseiBlack.png"
image2_path = "BestSenseiBlack.png"

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)
keypoints1, descriptors1 = sift_detector(image1)
keypoints2, descriptors2 = sift_detector(image2)

# Draw the key points on the image
image_with_keypoints = cv2.drawKeypoints(image1, keypoints1, None)

# Display the images
cv2.imshow('Original Image', image1)
cv2.imshow('Image with Keypoints', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create a Brute Force Matcher
bf = cv2.BFMatcher()

# Match descriptors using KNN
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract keypoints from the good matches
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Use RANSAC to find the best homography
homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Apply the best homography to the first image
result = cv2.warpPerspective(image1, homography, (image2.shape[1], image2.shape[0]))

# Draw the matches on the images
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matched image and the result after applying the homography
cv2.imshow('Matched Image', matched_image)
cv2.imshow('Result with RANSAC', result)
cv2.waitKey(0)
cv2.destroyAllWindows()