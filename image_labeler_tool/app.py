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

import pandas as pd
import cv2

from tkinter.filedialog import askdirectory

STARTING_IMAGE_MAX_WIDTH = 640
STARTING_IMAGE_MAX_HEIGHT = 480
IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".JPG"}



class App(tk.Frame):

    def __init__(self):
        self.top_root = tk.Tk()
        self.top_root.geometry(f"{int(STARTING_IMAGE_MAX_WIDTH*1.4)}x{int(STARTING_IMAGE_MAX_HEIGHT*1.4)}")
        self.top_root.resizable(False, False)

        # image size pane
        self.image_pane = tk.Frame(width=640, height=480)
        self.file_name_tk_var = tk.StringVar(value="None")
        self.image_label = tk.Label(textvariable=self.file_name_tk_var)
        self.image_label.pack(side=tk.TOP)
        self.image_pane.pack(side=tk.TOP)

        # controller pane
        self.controller_pane = tk.Frame()
        self.controller_pane.pack(side=tk.BOTTOM)

        # choose directory button
        self.previous_button = tk.Button(self.controller_pane, text="<-")
        self.choose_button = tk.Button(self.controller_pane,
                                       text="Click to select a root folder",
                                       command=self.set_folder)
        self.next_button = tk.Button(self.controller_pane, text="->")

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
                        img_files.append(path_to_str)

        if path.exists()


    def run(self):
        self.top_root.mainloop()


