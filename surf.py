import cv2
import matplotlib.pyplot as plt
import numpy as np


def surf(image1, image2):
    # Convert the image1 to RGB
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    # Convert the image1 to gray scale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    # Display image1 and image2
    fx, plots = plt.subplots(1, 2, figsize=(20, 10))

    plots[0].set_title("Image 1")
    plots[0].imshow(image1)

    plots[1].set_title("Image 2")
    plots[1].imshow(image2)

    # ## Detect keypoints and Create Descriptor

    surf = cv2.xfeatures2d.SURF_create(800)

    image1_keypoints, image1_descriptor = surf.detectAndCompute(image1_gray, None)
    image2_keypoints, image2_descriptor = surf.detectAndCompute(image2_gray, None)

    keypoints_without_size = np.copy(image1)
    keypoints_with_size = np.copy(image1)

    cv2.drawKeypoints(image1, image1_keypoints, keypoints_without_size, color=(0, 255, 0))

    cv2.drawKeypoints(image1, image1_keypoints, keypoints_with_size, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display image with and without keypoints size
    fx, plots = plt.subplots(1, 2, figsize=(20, 10))

    plots[0].set_title("Image 1 keypoints With Size")
    plots[0].imshow(keypoints_with_size, cmap='gray')

    plots[1].set_title("Image 1 keypoints Without Size")
    plots[1].imshow(keypoints_without_size, cmap='gray')

    # Print the number of keypoints detected in Image 1
    print("Number of Keypoints Detected In Image 1: ", len(image1_keypoints))

    # Print the number of keypoints detected in Image 2
    print("Number of Keypoints Detected In The Image 2: ", len(image2_keypoints))

    # ## Matching Keypoints

    # Create a Brute Force Matcher object.
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

    # Perform the matching between the SURF descriptors of image 1 and image2
    matches = bf.match(image1_descriptor, image2_descriptor)

    # The matches with shorter distance are the ones we want.
    matches = sorted(matches, key=lambda x: x.distance)

    return matches
