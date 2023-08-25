"""
look at all the images, and resize them to a more manageable size

I don't actually need the "real size images" to do this analysis, and it

"""
import pandas as pd
import os
from PIL import Image

data_scores = pd.read_csv("../data_scores.csv")


sizes = [1024, 512, 256, 128, 64]

for size in sizes:
    folder_name = f"data_{size}"
    os.system(f"rm -rf {folder_name}")
    os.system(f"mkdir {folder_name}")

    for score in [0, 3]:
        os.system(f"mkdir {folder_name}/score_{score}")

    file_names = []
    for index, row in data_scores.iterrows():
        path, score = row
        file_name = os.path.basename(path)

        if file_name in file_names:
            file_name = f"another_{file_name}"

        file_names.append(file_name)

        if score in [1, 2]:
            # I don't care about the ambiguous cases
            continue
        else:
            # resize the image and save
            img = Image.open(path).resize((size, size))
            img.save(f"{folder_name}/score_{score}/{file_name}")