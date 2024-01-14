import os
import cv2
import argparse


class DataCreator:
    def __init__(self, shift=50, destination="", first_dirname='A', second_dirname='B', create_destination=False):
        self.shift = shift
        self.destination = destination
        self.first_dirname = first_dirname
        self.second_dirname = second_dirname
        if create_destination:
            self.create_destination()

    def create_destination(self):
        if not os.path.exists(self.destination):
            os.mkdir(self.destination)
        if not os.path.exists(os.path.join(self.destination, self.first_dirname)):
            os.mkdir(os.path.join(self.destination, self.first_dirname))
        if not os.path.exists(os.path.join(self.destination, self.second_dirname)):
            os.mkdir(os.path.join(self.destination, self.second_dirname))

    def create_image_pair(self, image):
        image1 = image[:-self.shift, :-self.shift]
        image2 = image[self.shift:, self.shift:]
        return image1, image2

    @staticmethod
    def name_from_path(path):
        return path.split('\\')[-1]

    def read_and_store_pair(self, image_path):
        image = cv2.imread(image_path)
        image1, image2 = self.create_image_pair(image)
        image_name = self.name_from_path(image_path)
        cv2.imwrite(os.path.join(self.destination, self.first_dirname, image_name), image1)
        cv2.imwrite(os.path.join(self.destination, self.second_dirname, image_name), image2)
        # dot_pos = image_path.rfind('.')
        # ext = image_path[dot_pos:]
        # cv2.imwrite(f'{image_path[:dot_pos]}1{ext}', image1)
        # cv2.imwrite(f'{image_path[:dot_pos]}2{ext}', image2)

    def create_from_directory(self, dirpath):
        image_names = [file for file in os.listdir(dirpath) if not file.startswith('.')]
        for name in image_names:
            self.read_and_store_pair(os.path.join(dirpath, name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--source_dirpath")
    parser.add_argument('-d', "--destination_dirpath")
    args = parser.parse_args()

    data_creator = DataCreator(destination=args.destination_dirpath, create_destination=True)
    data_creator.create_from_directory(args.source_dirpath)
