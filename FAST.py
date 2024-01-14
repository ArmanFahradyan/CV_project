import cv2
import matplotlib.pyplot as plt
import numpy as np

def read_image(file_path):
    image = cv2.imread(file_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def detect_keypoints(image, nonmax_suppression=True):
    fast = cv2.FastFeatureDetector_create()
    fast.setNonmaxSuppression(nonmax_suppression)
    return fast.detect(image, None)

def draw_keypoints(image, keypoints):
    result_image = np.copy(image)
    cv2.drawKeypoints(image, keypoints, result_image, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return result_image

def display_images(images, titles):
    num_images = len(images)
    fig, plots = plt.subplots(1, num_images, figsize=(20, 10))
    for i in range(num_images):
        plots[i].set_title(titles[i])
        plots[i].imshow(images[i])

def main():
    file_path = 'C:\\Users\\serik\\fastImage\\test.jpeg'

    original_image = read_image(file_path)
    gray_image = convert_to_gray(original_image)

    keypoints_with_nonmax = detect_keypoints(gray_image, nonmax_suppression=True)
    keypoints_without_nonmax = detect_keypoints(gray_image, nonmax_suppression=False)

    image_with_nonmax = draw_keypoints(original_image, keypoints_with_nonmax)
    image_without_nonmax = draw_keypoints(original_image, keypoints_without_nonmax)

    display_images([original_image, gray_image], ["Original Image", "Gray Image"])
    display_images([image_with_nonmax, image_without_nonmax], ["With non max suppression", "Without non max suppression"])


if name == "__main__":
    main()
