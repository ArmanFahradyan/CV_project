import cv2
import matplotlib.pyplot as plt
import numpy as np


def surf(image1, image2):
    # Convert the image1 to RGB
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    # Convert the image1 to gray scale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    # ## Detect keypoints and Create Descriptor

    surf = cv2.xfeatures2d.SURF_create(800)

    image1_keypoints, image1_descriptor = surf.detectAndCompute(image1_gray, None)
    image2_keypoints, image2_descriptor = surf.detectAndCompute(image2_gray, None)

    # Display image with and without keypoints size

    # ## Matching Keypoints

    # Create a Brute Force Matcher object.
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

    # Perform the matching between the SURF descriptors of image 1 and image2
    matches = bf.match(image1_descriptor, image2_descriptor)

    # The matches with shorter distance are the ones we want.
    matches = sorted(matches, key=lambda x: x.distance)

    result = cv2.drawMatches(image1, image1_keypoints, image2_gray, image2_keypoints, matches, image2_gray, flags=2)

    return result
