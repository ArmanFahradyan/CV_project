import cv2
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

def orb(image1, image2):
    # Convert the training image to RGB
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    # Convert the training image to gray scale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    #Image2 turning Gray
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    # Display traning image and testing image --> image1/image2
    fx, plots = plt.subplots(1, 2, figsize=(20,10))

    plots[0].set_title("Training Image")
    plots[0].imshow(image1)

    plots[1].set_title("Testing Image")
    plots[1].imshow(image2)

    orb = cv2.ORB_create()

    train_keypoints, train_descriptor = orb.detectAndCompute(image1_gray, None)
    test_keypoints, test_descriptor = orb.detectAndCompute(image2_gray, None)

    keypoints_without_size = np.copy(image1)
    keypoints_with_size = np.copy(image2)

    cv2.drawKeypoints(image1, train_keypoints, keypoints_without_size, color = (0, 255, 0))

    cv2.drawKeypoints(image1, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display image with and without keypoints size
    fx, plots = plt.subplots(1, 2, figsize=(20,10))

    plots[0].set_title("Train keypoints With Size")
    plots[0].imshow(keypoints_with_size, cmap='gray')

    plots[1].set_title("Train keypoints Without Size")
    plots[1].imshow(keypoints_without_size, cmap='gray')

    # Print the number of keypoints detected in the training image
    print("Number of Keypoints Detected In The Training Image: ", len(train_keypoints))

    # Print the number of keypoints detected in the query image
    print("Number of Keypoints Detected In The Query Image: ", len(test_keypoints))

    # Create a Brute Force Matcher object.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    # Perform the matching between the ORB descriptors of the training image and the test image
    matches = bf.match(train_descriptor, test_descriptor)

    # The matches with shorter distance are the ones we want.
    matches = sorted(matches, key = lambda x : x.distance)

    return matches