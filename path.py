import cv2
import numpy as np
import os

def get_jpg_file_paths(folder_path):
    jpg_file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg'):
                jpg_file_paths.append(os.path.join(root, file))
    return jpg_file_paths

img_path = "/data/xx/CrackTree/imgs"
imgs = get_jpg_file_paths(img_path)
f = open("./data/test_example_CrackTree.txt","w")
for item in imgs:
    # masks_name = item[:-4] + ".bmp"
    f.write(os.path.join(img_path,item) + " " + os.path.join(img_path,item.replace("jpg","png"))+"\n")

