import pandas as pd
import os
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

from sklearn.metrics import classification_report

df = pd.read_csv("summary.csv", header=0)
sizes = [1024, 512, 256, 128, 64]

for size in sizes:
    frame_of_same_size = df[df['size'] == size]
    frame_of_same_size['guess'] = 0

    try:
        files_guessed_true = os.listdir(f"data_{size}/output/Entangled")
    except FileNotFoundError:
        files_guessed_true = []

    frame_of_same_size['guess'].loc[
        frame_of_same_size['file'].isin(files_guessed_true)] = 1

    y_true = frame_of_same_size['vegetation'].to_list()
    y_pred = frame_of_same_size['guess'].to_list()


    print(f'Classification report for {size}x{size} images')
    print(classification_report(y_true, y_pred,
                                ))




