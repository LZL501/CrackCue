from scipy import io
import os
import cv2
import numpy as np

def get_jpg_file_paths(folder_path):
    jpg_file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.png'):
                jpg_file_paths.append(os.path.join(root, file))
    return jpg_file_paths

savedir = "./gt_mat"
source = "/data/arima/GAPS384/croppedgt"
masks = get_jpg_file_paths(source)
os.makedirs("./gt_mat", exist_ok=True)
for item in masks:
    # total_item = os.path.join(source, item)
    total_item = item
    print(total_item)
    label = cv2.imread(total_item, 0)
    # print(item)
    # print(total_item)
    # print(label.shape)
    label = cv2.resize(label, (512, 512))
    # print(label.shape)
    label[label == 0] = 0
    label[label > 0] =1
    # if "6300" in item:
    #     print(np.unique(label,return_counts=True))
    item = os.path.basename(item)
    io.savemat(os.path.join(savedir, item.replace("png", "mat")), {'groundTruth':[{'Boundaries':label}]})