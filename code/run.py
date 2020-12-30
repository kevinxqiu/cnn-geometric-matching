"""
Script to load a frame-set (2D, rgb frames) and apply the alignment.
By default : Aligns to previous, otherwise, specify a frame to align to.
Can take as input image, video, or h5.

See README.md for details.
"""
import torch
import torch.nn as nn
from functions.model_alignment import * #load models, save_warped, etc.
from functions.data_extraction import * 
from util.handle_files import *
import h5py as h5
import argparse
import os 
from options.options import ArgumentParser
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

def main():
    #Parsing args
    args, arg_groups = ArgumentParser(mode='run').parse()
    print("Running script")
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    
    ### Loading the model ###
    
    #Parse args for network params (weights and models) and sets the feature regressor accordingly for the pretrained weights.
    aff_split = os.path.dirname(args.aff_weight).split('/trained_models/')[-1].split('_')
    tps_split = os.path.dirname(args.tps_weight).split('/trained_models/')[-1].split('_')
    feat_reg = [args.aff_fr, args.tps_fr]
    
    #Checking if the pathnames contains keyword to infer the model type (FE + FR)
    for index, split in enumerate([aff_split, tps_split]):
        if split[0] == 'baseline' or split[1] == 'basefr':
            feat_reg[index] = 'base'
            
        elif split[1] == 'deepfr':
            feat_reg[index] = 'deep'
            
        elif split[1] == 'simplerfr':
            feat_reg[index] = 'simpler'
            
    #Loading network
    model_aff, model_tps = load_model(args.aff_weight, args.aff_model, feat_reg[0],
                                     args.tps_weight, args.tps_model, feat_reg[1])
    
    #Image warping
    #In the case we specify datatype = video, but leave the default extension which is jpg:
    ext = args.extension
    if args.datatype == 'video' and args.extension == '.jpg':
    	ext = input("Please enter the extension of your video file. Ex : .mp4, .avi, .. \ninput :")
    
    if args.datatype=='image':
        savepath=batch_align(args.datapath, args.alignment_target, model_aff, model_tps, args.extension)
        save_video(savepath)
    
    elif args.datatype =='video': #first extracts all frames, then align
        extract_frames_video(args.datapath, ext)
        frames_path = args.datapath+'frames/'
        if args.alignment_target != '' and args.alignment_target != 'previous':
            target = frames_path+args.alignment_target+'.jpg'
        else:
            target = args.alignment_target
        savepath = batch_align(frames_path, target, model_aff, model_tps, '.jpg')
        save_video(savepath)
    
    elif args.datatype == 'h5': #assumes red and green channels with no blue channel, and datatype = int16 
        if args.datapath[-1]=='/':
            h5_file = h5.File(list_files(args.datapath,'h5'))
            frames_path = args.datapath+'/frames/'
            
        elif args.datapath[-3:] == '.h5':
            h5_file = h5.File(args.datapath)
            frames_path = os.path.dirname(args.datapath)+'/frames/'
            
        target = frames_path+args.alignment_target+'.jpg'
        extract_frames_h5(h5_file, frames_path, channel = args.h5_channels)
        savepath = batch_align(frames_path, target, model_aff, model_tps, '.jpg')
        save_video(savepath)
        
if __name__ == '__main__':
    main()