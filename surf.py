import cv2
import numpy as np


# def surf(image1, image2):
#     # Convert the image1 to RGB
#     image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
#
#     # Convert the image1 to gray scale
#     image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
#
#     image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
#
#     # ## Detect keypoints and Create Descriptor
#
#     surf = cv2.xfeatures2d.SURF_create(800)
#
#     image1_keypoints, image1_descriptor = surf.detectAndCompute(image1_gray, None)
#     image2_keypoints, image2_descriptor = surf.detectAndCompute(image2_gray, None)
#
#     # Display image with and without keypoints size
#
#     # ## Matching Keypoints
#
#     # Create a Brute Force Matcher object.
#     bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
#
#     # Perform the matching between the SURF descriptors of image 1 and image2
#     matches = bf.match(image1_descriptor, image2_descriptor)
#
#     # The matches with shorter distance are the ones we want.
#     matches = sorted(matches, key=lambda x: x.distance)
#
#     result = cv2.drawMatches(image1, image1_keypoints, image2_gray, image2_keypoints, matches, image2_gray, flags=2)
#
#     return result


def get_integral_image(image):
    integral_image = np.cumsum(np.cumsum(image, axis=0), axis=1)
    integral_image = np.insert(integral_image, 0, 0, axis=0)
    integral_image = np.insert(integral_image, 0, 0, axis=1)
    return integral_image


def calculate_box_integral(integral_image, x1, y1, x2, y2):
    A = integral_image[y1, x1]
    B = integral_image[y2, x1]
    C = integral_image[y1, x2]
    D = integral_image[y2, x2]
    return D - C - B + A


def calculate_hessian_response(integral_image, x, y, size):
    Dxx = calculate_box_integral(integral_image, x, y, x + size, y + size)
    Dyy = calculate_box_integral(integral_image, x, y, x + size, y + size)
    Dxy = calculate_box_integral(integral_image, x, y, x + size, y) - calculate_box_integral(integral_image, x, y,
                                                                                             x, y + size)
    k = 0.04  # Empirical constant
    det = Dxx * Dyy - Dxy ** 2
    trace = Dxx + Dyy
    return det - k * trace ** 2


def non_max_suppression(keypoints, threshold=0.8):
    keypoints.sort(key=lambda x: x.response, reverse=True)
    selected_keypoints = []
    for keypoint in keypoints:
        x, y = keypoint.pt
        if all(abs(x - kp.pt[0]) > 10 or abs(y - kp.pt[1]) > 10 for kp in selected_keypoints):
            selected_keypoints.append(keypoint)
    return selected_keypoints


def detect_surf(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    integral_image = get_integral_image(gray)

    keypoints = []

    for y in range(5, gray.shape[0] - 5):
        for x in range(5, gray.shape[1] - 5):
            response = calculate_hessian_response(integral_image, x, y, 5)
            if response > 1000:  # Adjust this threshold based on your requirements
                keypoints.append(cv2.KeyPoint(x, y, size=5, response=response))

    keypoints = non_max_suppression(keypoints)

    return keypoints


def surf_match_images(img1, img2):
    keypoints1 = detect_surf(img1)
    keypoints2 = detect_surf(img2)

    # Perform matching using simple Euclidean distance
    matches = []
    for kp1 in keypoints1:
        best_match = None
        min_distance = float('inf')

        for kp2 in keypoints2:
            distance = np.sqrt((kp1.pt[0] - kp2.pt[0]) ** 2 + (kp1.pt[1] - kp2.pt[1]) ** 2)
            if distance < min_distance:
                best_match = kp2
                min_distance = distance

        matches.append(cv2.DMatch(len(matches), len(matches), 0, min_distance))

    # Draw matches on the images
    matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None,
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return matched_image
