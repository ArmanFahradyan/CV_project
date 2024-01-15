import cv2
import numpy as np

def orb(image1, image2):
    # Convert the training image to RGB
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    # Convert the training image to gray scale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    #Image2 turning Gray
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create()

    train_keypoints, train_descriptor = orb.detectAndCompute(image1_gray, None)
    test_keypoints, test_descriptor = orb.detectAndCompute(image2_gray, None)

    keypoints_without_size = np.copy(image1)
    keypoints_with_size = np.copy(image2)

    # Create a Brute Force Matcher object.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    # Perform the matching between the ORB descriptors of the training image and the image2
    matches = bf.match(train_descriptor, test_descriptor)

    # The matches with shorter distance are the ones we want.
    matches = sorted(matches, key = lambda x : x.distance)

    result = cv2.drawMatches(image1, train_keypoints, image2_gray, test_keypoints, matches, image2_gray, flags = 2)

    return result
