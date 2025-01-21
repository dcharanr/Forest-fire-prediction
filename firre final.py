import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# MAIN PATH
Fire_Dataset_Path = Path(r"C:\project 4-2")  # Adjust the directory path

# PATH PROCESS
PNG_Path = list(Fire_Dataset_Path.glob(r"*/*.png"))

# LABEL PROCESS
PNG_Labels = [os.path.split(os.path.split(x)[0])[1] for x in PNG_Path]

print("FIRE: ", PNG_Labels.count("__results___35_1.png"))
print("NO_FIRE: ", PNG_Labels.count("__results___36_1.png"))

# TRANSFORMATION TO SERIES STRUCTURE
PNG_Path_Series = pd.Series(PNG_Path, name="PNG").astype(str)
PNG_Labels_Series = pd.Series(PNG_Labels, name="CATEGORY")

print(PNG_Path_Series)
print(PNG_Labels_Series)

PNG_Labels_Series.replace({"non_fire_images": "NO_FIRE", "fire_images": "FIRE"}, inplace=True)

print(PNG_Labels_Series)

# TRANSFORMATION TO DATAFRAME STRUCTURE
Main_Train_Data = pd.concat([PNG_Path_Series, PNG_Labels_Series], axis=1)

print(Main_Train_Data.head(-1))

# SHUFFLING
Main_Train_Data = Main_Train_Data.sample(frac=1).reset_index(drop=True)

print(Main_Train_Data.head(-1))

# VISUALIZATION
plt.style.use("dark_background")

# GENERAL
sns.countplot(Main_Train_Data["CATEGORY"])
plt.show()

Main_Train_Data['CATEGORY'].value_counts().plot.pie(figsize=(5, 5))
plt.show()

# IMAGES
try:
    if not Main_Train_Data.empty:
        figure = plt.figure(figsize=(10, 10))
        x = cv2.imread(Main_Train_Data["PNG"][0])
        plt.imshow(x)
        plt.xlabel(x.shape)
        plt.title(Main_Train_Data["CATEGORY"][0])
    else:
        print("DataFrame is empty.")
except KeyError as e:
    print(f"Error: {e}. Index 0 is not in range.")

try:
    if not Main_Train_Data.empty:
        fig, axes = plt.subplots(nrows=5,
                                 ncols=5,
                                 figsize=(10, 10),
                                 subplot_kw={"xticks": [], "yticks": []})

        for i, ax in enumerate(axes.flat):
            ax.imshow(cv2.imread(Main_Train_Data["PNG"][i]))
            ax.set_title(Main_Train_Data["CATEGORY"][i])
    else:
        print("DataFrame is empty.")
except KeyError as e:
    print(f"Error: {e}. Index is out of range.")
