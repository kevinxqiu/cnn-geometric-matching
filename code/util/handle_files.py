import os
import glob 
from natsort import natsorted
import natsort
"""
Coded by R.Wan. The file is located here instead of functions because I initially created it here,
then didn't want to move the file to functions and have to update the imports in every file, risking to miss some.
  """


def makedir(relative_path):
    """
    Checks whether a directory exists, and creates it if not.
    Used when extracting, augmenting, saving data
    """
    dirname = os.getcwd()
    target_path = os.path.join(dirname, relative_path) #Get absolute target path
    test = not os.path.isdir(target_path)
    if test:
        os.mkdir(target_path)
        print("\nTarget directory doesn't exist. Creating directory under {}".format(target_path))
        return target_path
    if test == False:
        return target_path
    

def list_files(relative_path, extension, reverse_sort=False):
    """
    input : relative_path (str), ex : '../folder1/folder2/'
            extension (str), with or without dot
    For a given relative folder path and extension as strings, returns an iterator that contains all the filenames with a given extension.
    Ex : A folder contains 1000 images, from frame0.jpg to frame999.jpg, extension = jpg
    Returns a iglob containing ['frame0.jpg', ... , 'frame999.jpg']
    Uses natsort to sort the strings, otherwise it will be ordered as ['frame0.jpg, frame1.jpg, frame10.jpg, frame100.jpg ...']
    """
    
    if '.' not in extension[0] : #adds a dot if not specified. 'jpg' --> '.jpg'
        extension = '.'+extension
    extension = '*'+extension
    if relative_path[-1] != '/':
        relative_path += '/' #adds slash if not already here
        
    temp = glob.iglob(relative_path+extension)
    files_list = natsort.natsorted(temp, reverse=reverse_sort)
    if len(files_list) == 1:
        return files_list[0]
    
    else:
        return files_list
        