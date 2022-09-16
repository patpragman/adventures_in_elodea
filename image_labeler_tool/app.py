"""
This file contains a super simple tool that allows you to get all the images in a directory and the subdirectory
iterate through them and label them as "contains object" or not.

this creates a pandas dataframe (which can then be exported as a .csv file) that contains the following columns
filename
path to file relative to the top directory
whether it contains the desired thing or not
"""


import tkinter as tk
import os
import pathlib
import atexit
import pandas as pd
from operator import add, sub
from PIL import Image, ImageTk  # libraries for handling images in tkinter
import cv2

from tkinter.filedialog import askdirectory

STARTING_IMAGE_MAX_WIDTH = 640
STARTING_IMAGE_MAX_HEIGHT = 480
SCALE_FACTOR = 1.2
IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".JPG"}



class App(tk.Frame):

    def __init__(self):
        self.top_root = tk.Tk()  # root window of the app
        # now set the size to a max size times the scale factor and lock the window size
        self.top_root.geometry(f"{int(STARTING_IMAGE_MAX_WIDTH*SCALE_FACTOR)}x{int(STARTING_IMAGE_MAX_HEIGHT*SCALE_FACTOR)}")
        self.top_root.resizable(False, False)

        # image size pane
        self.image_pane = tk.Frame(width=640, height=480)
        self.file_name_tk_var = tk.StringVar(value="None")
        self.image_label = tk.Label(self.image_pane, textvariable=self.file_name_tk_var)
        self.image_label.pack(side=tk.TOP)
        self.image_pane.pack(side=tk.TOP)

        # controller pane
        self.controller_pane = tk.Frame()
        self.controller_pane.pack(side=tk.BOTTOM)

        # choose directory button
        self.previous_button = tk.Button(self.controller_pane, text="<-",
                                         command=lambda: self.change_index(sub))
        self.choose_button = tk.Button(self.controller_pane,
                                       text="Click to select a root folder",
                                       command=self.set_folder)
        self.next_button = tk.Button(self.controller_pane, text="->",
                                     command=lambda: self.change_index(add))


        # label the directory you're working on
        self.dir_tk_var = tk.StringVar(value="None")
        self.dir_label = tk.Label(textvariable=self.dir_tk_var)

        # now layout for those buttons...
        self.previous_button.pack(side=tk.LEFT)
        self.choose_button.pack(side=tk.LEFT)
        self.next_button.pack(side=tk.LEFT)
        self.dir_label.pack(side=tk.BOTTOM)

        # variable that we hold all the image file names in
        self.image_file_names = []

        # dataframe to contain the current information in the top directory
        self.df = pd.DataFrame(data={})

        # index for image
        self.image_index = 0

        # at exit cleanup stuff
        atexit.register(self.clean_up)

    def set_folder(self):
        working_directory = askdirectory(
            title="select the folder where your data lives",
            mustexist=True
        )
        self.dir_tk_var.set(working_directory)
        all_file_tuples = os.walk(working_directory, topdown=True)
        img_files = []
        for root, dir, files in all_file_tuples:
            if files:
                for file in files:
                    path_to_str = f"{root}/{file}"
                    path = pathlib.Path(path_to_str)
                    suffixes = set(path.suffixes)

                    if IMAGE_FILE_EXTENSIONS.intersection(suffixes):
                        relative_path = os.path.relpath(path_to_str, working_directory)
                        img_files.append(relative_path)

        # check to see if it's already been looked at

        if os.path.isfile(f"{working_directory}/labeling.csv"):
            self.df = pd.read_csv(f"{working_directory}/labeling.csv")
        else:
            # we need to populate the df in this half of the conditional

            self.df = pd.DataFrame(img_files, columns=["relative_path"])
            self.df['condition'] = 0

        # you're loading a new image, set the index back to zero and load
        self.image_index = 0
        self.load_image()


    def load_image(self):
        # get the relative path of the image and the working directory
        image_relative_path = self.df['relative_path'].iloc[self.image_index]
        working_directory = self.dir_tk_var.get()
        image_path = f"{working_directory}/{image_relative_path}"

        # load up the image, resize to fit, then update the label that lives in the appropriate tk.Frame
        image = Image.open(image_path)
        image_widget = ImageTk.PhotoImage(
            image.resize(
                (STARTING_IMAGE_MAX_WIDTH, STARTING_IMAGE_MAX_HEIGHT),
                Image.ANTIALIAS))
        self.image_label.configure(image=image_widget)
        self.image_label.image = image_widget

    def change_index(self, change_function):
        self.image_index = change_function(self.image_index, 1)
        if self.image_index > len(self.df) - 1:
            self.image_index = 0
        elif self.image_index < 0:
            self.image_index = len(self.df) - 1

        self.load_image()

    def run(self):
        self.top_root.mainloop()

    def clean_up(self):
        working_directory = self.dir_tk_var.get()
        self.df.to_csv(f"{working_directory}/labeling.csv")

