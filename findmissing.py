import pandas as pd
import os
from pprint import pprint

df = pd.read_csv('data_scores.csv')
file_names = df['path'].tolist()

bogies = []

def endswith(filename:str, extension:str) -> str:
    if filename.split(".")[1].upper()[0:] == extension.upper():
        return True
    else:
        return False


for root, _, files in os.walk("Pics2022", topdown=True):
    for file in files:
        if endswith(file, "jpg"):
            bogies.append(os.path.join(os.getcwd(), os.path.join(root, file)))


missed_list = []
for potential in bogies:
    if potential in file_names:
        pass
    else:
        missed_list.append(potential)

pprint(missed_list)
