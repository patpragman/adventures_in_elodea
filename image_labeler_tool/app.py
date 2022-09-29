"""
This file contains a super simple tool that allows you to get all the images in a directory and the subdirectory
iterate through them and label them as "contains object" or not.

this creates a pandas dataframe (which can then be exported as a .csv file) that contains the following columns
filename
path to file relative to the top directory
whether it contains the desired thing or not
"""
from unsupervised import get_all_image_files
import tkinter as tk
import os
import pathlib
import atexit
import pandas as pd
from operator import add, sub
from PIL import Image, ImageTk  # libraries for handling images in tkinter
import toml

from tkinter.filedialog import askdirectory

STARTING_IMAGE_MAX_WIDTH = 640
STARTING_IMAGE_MAX_HEIGHT = 480
SCALE_FACTOR = 1.2

TRUE_COLOR = 'green'
FALSE_COLOR = 'red'

IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".JPG"}

class ZoomWindow:

    def __init__(self, x, y, image, flipped):
        root = tk.Tk()
        root.title("Popup")

        image_widget = ImageTk.PhotoImage(
            image.resize(
                (STARTING_IMAGE_MAX_WIDTH, STARTING_IMAGE_MAX_HEIGHT),
                Image.ANTIALIAS).rotate(
                180 * flipped))

    def run(self):
        pass


class App:

    def __init__(self):
        self.top_root = tk.Tk()  # root window of the app
        # now set the size to a max size times the scale factor and lock the window size
        self.top_root.geometry(
            f"{int(STARTING_IMAGE_MAX_WIDTH * SCALE_FACTOR)}x{int(STARTING_IMAGE_MAX_HEIGHT * SCALE_FACTOR)}")
        self.top_root.resizable(False, False)
        self.top_root.title("Picture Categorizer")

        # image size pane
        self.x = 0
        self.y = 0

        self.image_pane = tk.Frame(width=640, height=480)
        self.file_name_tk_var = tk.StringVar(value="None")
        self.image_label = tk.Label(self.image_pane, textvariable=self.file_name_tk_var)
        self.image_label.pack(side=tk.TOP)
        self.image_pane.pack(side=tk.TOP, expand=True)
        self.top_root.bind("<Motion>", self.mouse_image_pane_callback)
        self.image_pane.bind("<Button-1>", self.spawn_popup)

        # controller pane
        self.controller_pane = tk.Frame()
        self.controller_pane.pack(side=tk.BOTTOM)

        # choose directory and image buttons
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

        # status label - this variable / label combo gives you the current status of an image
        self.status_label_var = tk.StringVar(value="None Selected")
        self.status_label = tk.Label(self.controller_pane, textvariable=self.status_label_var)
        self.status_int = tk.IntVar()
        self.status_label_checkbox = tk.Checkbutton(self.controller_pane,
                                                    text="Status",
                                                    variable=self.status_int,
                                                    command=lambda:
                                                    self.toggle())

        # checkbox to flip the image if it's upside down
        self.flip_image_var = tk.IntVar()
        self.flip_checker = tk.Checkbutton(self.controller_pane,
                                           text='Flip?',
                                           variable=self.flip_image_var,
                                           onvalue=1, offvalue=0,
                                           command=self.load_image)

        # now layout for those buttons...
        self.status_label.pack(side=tk.TOP)
        self.status_label_checkbox.pack(side=tk.LEFT)
        self.previous_button.pack(side=tk.LEFT)
        self.choose_button.pack(side=tk.LEFT)
        self.next_button.pack(side=tk.LEFT)
        self.dir_label.pack(side=tk.BOTTOM)
        self.flip_checker.pack(side=tk.LEFT)

        # keyboard bindings
        self.top_root.bind("<space>", lambda _: self.toggle())
        self.top_root.bind("<Left>", lambda _: self.change_index(sub))
        self.top_root.bind("<Right>", lambda _: self.change_index(add))

        # variable that we hold all the image file names in
        self.image_file_names = []

        # dataframe to contain the current information in the top directory
        self.df = pd.DataFrame(data={})

        # index for image
        self.image_index = 0

        # load previous state
        self._load_on_start()

        # at exit cleanup stuff
        atexit.register(self._clean_up)

    def mouse_image_pane_callback(self, event):
        actual_x = event.x
        actual_y = event.y
        frame_x = actual_x - self.image_label.winfo_x()
        frame_y = actual_y - self.image_label.winfo_y()
        self.x, self.y = frame_x, frame_y

    def spawn_popup(self, event):
        pass

    def set_folder(self):
        working_directory = askdirectory(
            title="select the folder where your data lives",
            mustexist=True
        )
        self.dir_tk_var.set(f"{working_directory}")
        all_file_tuples = os.walk(working_directory, topdown=True)
        img_files = get_all_image_files(working_directory)


        # check to see if it's already been looked at

        if os.path.isfile(f"{working_directory}/labeling.csv"):
            self.df = pd.read_csv(f"{working_directory}/labeling.csv")
        else:
            # we need to populate the df in this half of the conditional

            self.df = pd.DataFrame(img_files, columns=["relative_path"])
            self.df['condition'] = 0

        # you're loading a new image folder, set the index back to zero and load
        self.image_index = 0
        self.load_image()

    def _load_on_start(self):
        if os.path.isfile("state.toml"):
            with open("state.toml") as toml_file:
                save_state = toml.load(toml_file)
                self.flip_image_var.set(save_state['flip_image_var'])
                self.image_index = save_state['image_index']
                self.dir_tk_var.set(save_state['dir_tk_var'])
                self.df = pd.read_csv(f"{self.dir_tk_var.get()}/labeling.csv")
                self.load_image()

    def toggle(self):
        if self.df.at[self.image_index, "condition"]:
            self.mark_false()
        else:
            self.mark_true()

    def mark_true(self):
        # set the marking on that image to 1
        self.df.at[self.image_index, "condition"] = 1
        self.load_image()

    def mark_false(self):
        # set the marking on the current image to 0
        self.df.at[self.image_index, "condition"] = 0
        self.load_image()

    def load_image(self):
        # get the relative path of the image and the working directory
        image_relative_path = self.df['relative_path'].iloc[self.image_index]

        # now get the current status of the image
        current_status = self.df['condition'].iloc[self.image_index]

        """
        we need to set the color of the status label and make sure the switch is set properly
        
        tkinter is weird though - it was extremely buggy if you didn't clear the status label checkbox first before
        reselecting it.
        """
        self.status_label_checkbox.selection_clear()
        if current_status:
            self.status_label_checkbox.select()
            color = TRUE_COLOR
        else:
            self.status_label_checkbox.deselect()
            color = FALSE_COLOR


        # get information about where you're at
        working_directory = self.dir_tk_var.get()
        image_path = f"{working_directory}/{image_relative_path}"

        # load up the image, resize to fit, then update the label that lives in the appropriate tk.Frame
        image = Image.open(image_path)
        image_widget = ImageTk.PhotoImage(
            image.resize(
                (STARTING_IMAGE_MAX_WIDTH, STARTING_IMAGE_MAX_HEIGHT),
                Image.ANTIALIAS).rotate(
                180 * self.flip_image_var.get()))
        # note the rotation method tacked onto that - if the "flip" switch is toggled, you'll rotate the image at load

        # load the image
        self.image_label.configure(image=image_widget)
        self.image_label.image = image_widget

        # finally, change the color of the status label and the text
        self.status_label_var.set(f"{image_relative_path} {self.image_index}/{len(self.df)}")
        self.status_label.configure(bg=color)
        self.status_label.bg = color

    def change_index(self, change_function):
        """
        we pass a change function to this function where we can increment or decrement by one.

        we change the index by change_function(index, 1) which will move it up one or down one
        """
        self.image_index = change_function(self.image_index, 1)
        # handle limits of the image
        if self.image_index > len(self.df) - 1:
            self.image_index = 0
        elif self.image_index < 0:
            self.image_index = len(self.df) - 1

        self.load_image()

    def run(self) -> None:
        """
        run the app
        """
        self.top_root.mainloop()

    def _clean_up(self) -> None:
        """
        store all the information we want to stay persistence from session to session in a toml file
        """

        working_directory = self.dir_tk_var.get()
        self.df.to_csv(f"{working_directory}/labeling.csv", index=False)  # save the labels you've made

        # we need to save all the stuff we care about in a toml dictionary
        if self.dir_tk_var.get():
            with open("state.toml", "w") as toml_file:
                save_state = {}
                save_state['image_index'] = self.image_index
                save_state['flip_image_var'] = self.flip_image_var.get()
                save_state['dir_tk_var'] = self.dir_tk_var.get()
                toml.dump(save_state, toml_file)
