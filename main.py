import os
import cv2

from sift import sift_matching
from superglue import superglue
from fast import fast
from surf import surf
from orb import orb

RESULTS_DIR = "results"
DATASET_DIR = "dataset"
FIRST_DIRNAME = 'A'
SECOND_DIRNAME = 'B'


def main():
    models = {"sift": sift_matching,
              "superglue": superglue,
              "fast": fast,
              "surf": surf,
              "orb": orb}

    image_names = [file for file in os.listdir(os.path.join(DATASET_DIR, FIRST_DIRNAME)) if not file.startswith('.')]

    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    for model_name in models.keys():
        if not os.path.exists(os.path.join(RESULTS_DIR, model_name)):
            os.mkdir(os.path.join(RESULTS_DIR, model_name))

        model = models[model_name]

        for name in image_names:
            image1 = cv2.imread(os.path.join(DATASET_DIR, FIRST_DIRNAME, name))
            image2 = cv2.imread(os.path.join(DATASET_DIR, SECOND_DIRNAME, name))

            result = model(image1, image2)

            cv2.imwrite(os.path.join(RESULTS_DIR, model_name, name), result)


if __name__ == "__main__":
    main()
