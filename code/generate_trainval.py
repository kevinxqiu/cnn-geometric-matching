"""
Script to generate a train and validation set as well as the associated csv files, and can augment if needed.

Ex : From a folder containing 100 jpg frames, can generate augmented sets 
"""

import torch
import torch.nn as nn
from functions.model_alignment import * #load models, save_warped, etc.
from functions.data_extraction import * 
from functions.data_generation import*
from util.handle_files import *
import h5py as h5
import argparse
import os 
import shutil
from options.options import ArgumentParser
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 



def main ():
    args, arg_groups = ArgumentParser(mode='generate').parse()
    
    #Creating the target directory name
    name = ''
    if args.sourcepath[-1]=='/':
        name = os.path.basename(args.sourcepath[0:-1])+'/'
    elif args.sourcepath[-1]!= '/':
        name = os.path.basename(args.sourcepath)+'/' 
        
    data_directory = './datasets/'+name+'TrainVal/'
    
    #Scenario depending on filetype (img/h5/vid)
    if args.datatype == 'h5':
        h5_files = list_files(args.sourcepath, 'h5')
        extract_frames_h5(h5_files[0], args.sourcepath+'frames/', channel = args.h5_channels)
        source = args.sourcepath+'frames/'
        extension = '.jpg'
        
    elif args.datatype == 'video':

        extract_frames_video(args.sourcepath, args.extension)
        extension = '.jpg'
        source = args.sourcepath+'frames/'
    
    elif args.datatype == 'image':
        extension = args.extension
        source = args.sourcepath
    
    shutil.copytree(source, data_directory)
    
    if args.augment != 0:
        augment_images(data_directory, someAug(), times_target = args.augment, source = True)

    create_csv_split(data_directory)
    
if __name__ == '__main__':
    main()