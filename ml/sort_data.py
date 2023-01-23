import pandas as pd
import shutil
import os

data_scores = pd.read_csv("../data_scores.csv")


os.system("rm -rf data")
os.system("mkdir data")
for score in [0, 3]:
    os.system(f"mkdir data/score_{score}")

file_names = []
for index, row in data_scores.iterrows():
    path, score = row
    file_name = os.path.basename(path)

    if file_name in file_names:
        file_name = f"another_{file_name}"

    file_names.append(file_name)

    if score in [1, 2]:
        continue
    else:
        shutil.copyfile(path, f"data/score_{score}/{file_name}")  # commented out for safety

print(len(file_names))