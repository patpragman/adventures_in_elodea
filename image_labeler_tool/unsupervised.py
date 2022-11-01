from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil
# most of this code comes from:  https://github.com/shubham7169/Projects/blob/master/Image_Clustering.ipynb

import os
import pathlib

IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".JPG"}


def get_all_image_files(directory) -> list:
    # returns list of all image files in a directory

    all_file_tuples = os.walk(directory, topdown=True)
    img_files = []
    for root, dir, files in all_file_tuples:
        if files:
            for file in files:
                path_to_str = f"{root}/{file}"
                path = pathlib.Path(path_to_str)
                suffixes = set(path.suffixes)

                if IMAGE_FILE_EXTENSIONS.intersection(suffixes):
                    relative_path = os.path.relpath(path_to_str, directory)
                    img_files.append(path)

    return img_files


# Function to Extract features from the images
def get_files_features(file_list):
    model = InceptionV3(weights='imagenet', include_top=False)
    features = [];
    img_name = [];
    for file in tqdm(file_list):
        img = image.load_img(file, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = model.predict(x)
        feat = feat.flatten()
        features.append(feat)
        img_name.append(file)

    return features

def cluster_features(img_features, k=2, random_state=40) -> KMeans:
    #Creating Clusters
    clusters = KMeans(k, random_state=random_state)
    clusters.fit(img_features)
    return clusters


def generate_dataframe(file_list) -> pd.DataFrame:
    img_features = get_files_features(file_list)
    kmeans = cluster_features(img_features, k=25).labels_

    data_lists = [file_list, kmeans]
    col_names = ['path', 'label']
    df = pd.concat([pd.Series(x) for x in data_lists], axis=1)
    df.columns = col_names
    return df


if __name__ == "__main__":
    from pprint import pprint

    all_images = get_all_image_files("../ElodeaGoProPics_2018/Starboard9_14")
    df = generate_dataframe(all_images)
    df.to_csv('test.csv')