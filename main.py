import os
import cv2

from sift import sift
from superglue import superglue
from fast import fast

RESULTS_DIR = "results"
DATASET_DIR = "dataset"
FIRST_DIRNAME = 'A'
SECOND_DIRNAME = 'B'


def main():
    models = {"sift": sift,
              "superglue": superglue,
              "fast": fast}

    image_names = [file for file in os.listdir(os.path.join(DATASET_DIR, FIRST_DIRNAME)) if not file.startswith('.')]

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
