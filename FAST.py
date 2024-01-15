import cv2


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def fast(image1, image2):

    gray1 = convert_to_gray(image1)
    gray2 = convert_to_gray(image2)

    fast_ = cv2.FastFeatureDetector_create()

    keypoints1 = fast_.detect(gray1, None)
    keypoints2 = fast_.detect(gray2, None)

    orb = cv2.ORB_create()

    # Compute descriptors
    keypoints1, descriptors1 = orb.compute(gray1, keypoints1)
    keypoints2, descriptors2 = orb.compute(gray2, keypoints2)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return matched_image
