import pandas as pd
import os
import json

if __name__ == "__main__":
    target_folder = os.path.join(os.getcwd(), "Pics2022")

    data = {"path": [],
            "score": []}

    for root, _, filenames in os.walk(target_folder, topdown=True):

        for filename in filenames:
            if filename == "scores.json":
                with open(os.path.join(root, filename), "r") as json_file:
                    json_obj = json.load(json_file)

                    if json_obj['images']:
                        for image in json_obj['images']:
                            path = image['path']
                            score = image['score']

                            data["path"].append(path)
                            data["score"].append(score)


    df = pd.DataFrame(data)

    df.to_csv('data_scores.csv', index=False)
