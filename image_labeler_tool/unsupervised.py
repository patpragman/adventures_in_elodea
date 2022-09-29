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
                    relative_path = os.path.relpath(path_to_str, working_directory)
                    img_files.append(relative_path)

    return img_files