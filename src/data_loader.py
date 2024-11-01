import os
import cv2
import numpy as np

def load_data(data_dir, categories, img_size=(100, 100)):
    data = []
    labels = []
    for category in categories:
        category_path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(category_path):
            try:
                img_array = cv2.imread(os.path.join(category_path, img))
                img_resized = cv2.resize(img_array, img_size)
                data.append(img_resized)
                labels.append(class_num)
            except Exception as e:
                pass
    return np.array(data), np.array(labels)

def load_test_data(data_dir, img_size=(100, 100)):
    data = []
    for img in os.listdir(data_dir):
        try:
            img_path = os.path.join(data_dir, img)
            img_array = cv2.imread(img_path)
            img_resized = cv2.resize(img_array, img_size)
            data.append(img_resized)
        except Exception as e:
            pass
    return np.array(data)