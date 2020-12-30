### IMPORTS ###
from __future__ import print_function, division
from util.handle_files import * #Contains makedir() and list_files()
import os
import argparse
import torch
import torch.nn as nn
import cv2
import shutil
from torch.utils.data import Dataset, DataLoader
from model.cnn_geometric_model import CNNGeometric
from data.pf_dataset import PFDataset
from data.download_datasets import download_PF_willow
from image.normalization import NormalizeImageDict, normalize_image
from util.torch_util import BatchTensorToVars, str_to_bool
from geotnf.transformation import GeometricTnf
from geotnf.point_tnf import *
import matplotlib.pyplot as plt
from skimage import io
import warnings
from torchvision.transforms import Normalize
from collections import OrderedDict
warnings.filterwarnings('ignore')

###Base fcts used by methods 
use_cuda = torch.cuda.is_available()
tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)
affTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)
resizeCNN = GeometricTnf(out_h=240, out_w=240, use_cuda = False) 
means = [0.0743, 0.0755, 0.0070]
stds = [0.0198, 0.0320, 0.0155]
normalizeTnf = Normalize(mean = means, std=stds)

###
def load_model(aff_params_path = '', aff_feat_ext = 'wormbrain_1', aff_feat_reg = 'simpler',
               tps_params_path = '', tps_feat_ext = 'wormbrain_1', tps_feat_reg = 'simpler'):
    """
    Loads a model. Assumes that each model (Affine and Thin-Plate Spline)
    have been trained separately. Must specify the architecture used for feature_extraction
    By default, it is resnet101
    """
    use_cuda = torch.cuda.is_available()
    #Only create a model for which weights have been provided
    do_aff = not aff_params_path==''
    do_tps = not tps_params_path==''
    if not do_aff and not do_tps: 
        print("No weights found. Models not created, exiting.")
        return
    
    print("Creating CNN model.")
    if do_aff:
        model_aff = CNNGeometric(output_dim=6,use_cuda=use_cuda,
                             feature_extraction_cnn= aff_feat_ext,
                                feature_regression=aff_feat_reg)
    if do_tps:
        model_tps = CNNGeometric(output_dim=18,use_cuda=use_cuda,
                             feature_extraction_cnn= tps_feat_ext,
                                feature_regression = tps_feat_reg)
    print("Loading trained model weights.")
    
    if do_aff: #Loading affine model    
        if aff_feat_ext == 'resnet101': aff_feat_ext = 'resnet'
            
        checkpoint = torch.load(aff_params_path, map_location=lambda storage, loc: storage)
        checkpoint['state_dict'] = OrderedDict([(k.replace(aff_feat_ext, 'model'), v) for k, v in checkpoint['state_dict'].items()])
        model_aff.load_state_dict(checkpoint['state_dict'])
        print('Weights for Affine model loaded.')
    
    if do_tps: #Loading thin plate spline model
        if tps_feat_ext == 'resnet101': aff_feat_ext = 'resnet'
    
        checkpoint = torch.load(tps_params_path, map_location=lambda storage, loc: storage)
        checkpoint['state_dict'] = OrderedDict([(k.replace(aff_feat_ext, 'model'), v) for k, v in checkpoint['state_dict'].items()])
        model_tps.load_state_dict(checkpoint['state_dict'])
        print('Weights for Thin-Plate Spline model loaded.')
    
    print('Returning model(s).')
    
    if do_aff and not do_tps:
        return model_aff
    if do_tps and not do_aff:
        return model_tps
    if do_aff and do_tps:
        return model_aff, model_tps
    

def preprocess_image(image,means,stds):
    """
    Preprocesses the image for warping
    """
    # convert to torch Variable
    image = np.expand_dims(image.transpose((2,0,1)),0)
    image = torch.Tensor(image.astype(np.float32)/255.0)
    image_var = Variable(image,requires_grad=False)
    # Resize image using bilinear sampling with identity affine tnf
    image_var = resizeCNN(image_var)
    # Normalize image
    image_var = normalize_image(image_var,mean=means,std=stds)
    return image_var

def save_warped(source_name, target_name, savename, model_aff, model_tps, demo=False):
    """
    Aligns a source image to a target image, and saves it.
    """
    do_aff, do_tps = not model_aff == '', not model_tps == ''
    if not (do_aff and do_tps):
        print("No model found. Exiting.")
        return
    source_image = io.imread(source_name)
    target_image = io.imread(target_name)
    #Use the preprocess method declared above to resize, 
    #normalize using the means and stds. Here, by default uses the means/stds of ImageNet,
    #Otherwise causes some weird issues we don't understand why.
    source_image_var = preprocess_image(source_image,means=means, stds=stds)
    target_image_var = preprocess_image(target_image,means=means, stds=stds)
    
    if use_cuda:
        source_image_var = source_image_var.cuda()
        target_image_var = target_image_var.cuda()
    #Create a "batch" (i.e. a pair) for the next cell below
    batch = {'source_image': source_image_var, 'target_image':target_image_var}
    #Resize target: create a function that will resize a given input into the target_image's dimension
    resizeTgt = GeometricTnf(out_h=target_image.shape[0], out_w=target_image.shape[1], use_cuda = use_cuda) 
    #Set the models to eval mode
    if do_aff:
        model_aff.eval()
    if do_tps:
        model_tps.eval()      
    # Evaluate models and get the thetas
    if do_aff:
        theta_aff=model_aff(batch)
        warped_image_aff = affTnf(batch['source_image'],theta_aff.view(-1,2,3))
    
    if do_tps:
        theta_tps=model_tps(batch)
        warped_image_tps = tpsTnf(batch['source_image'],theta_tps)
    
    if do_aff and do_tps:
        theta_aff_tps=model_tps({'source_image': warped_image_aff, 'target_image': batch['target_image']})        
        warped_image_aff_tps = tpsTnf(warped_image_aff,theta_aff_tps)
    if do_aff:
        warped_image_aff_np = normalize_image(resizeTgt(warped_image_aff),forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
    
    if do_tps:
        warped_image_tps_np = normalize_image(resizeTgt(warped_image_tps),forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
    
    if do_aff and do_tps:
        warped_image_aff_tps_np = normalize_image(resizeTgt(warped_image_aff_tps),forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
        x = np.clip(warped_image_aff_tps_np,0,1)

    if demo==False:
        plt.imsave(savename, x)
    else:
        N_subplots = 2+int(do_aff)+int(do_tps)+int(do_aff and do_tps)
        fig, axs = plt.subplots(1,N_subplots)
        axs[0].imshow(source_image)
        axs[0].set_title('src')
        axs[1].imshow(target_image)
        axs[1].set_title('tgt')
        subplot_idx = 2
        if do_aff:
            axs[subplot_idx].imshow(warped_image_aff_np)
            axs[subplot_idx].set_title('aff')
            subplot_idx +=1 
        if do_tps:
            axs[subplot_idx].imshow(warped_image_tps_np)
            axs[subplot_idx].set_title('tps')
            subplot_idx +=1 
        if do_aff and do_tps:
            axs[subplot_idx].imshow(warped_image_aff_tps_np)
            axs[subplot_idx].set_title('aff+tps')
        
        for i in range(N_subplots):
            axs[i].axis('off')
        
        fig.set_dpi(330)
        plt.show()
    
def batch_align_to_previous(sourcepath, ext='jpg', model_aff ='', model_tps ='', iterations=1):
    """
    For a folder of images, align each frame to the previous one.
    """
    input_reversed = list_files(sourcepath, ext,reverse_sort=True)
    #getting directories
    savepath = makedir(sourcepath+'alignedprevious/')
    makedir(savepath+'/iteration_0/')
    savepath = sourcepath+'alignedprevious/iteration_0/'
    
    for iteration in range(iterations):
        #CREATING DIRECTORY TO RECEIVE RESULTS
        shutil.copy2(input_reversed[-1], savepath)    
        makedir(savepath)
        print("\n ### Aligning ###") 
        #Looping and doing the warp etc. 
        for i in range(len(input_reversed)-1):
            source = input_reversed[i]
            target = input_reversed[i+1]
            outname = savepath+'frame_'+str(len(input_reversed)-i-1)+'_to_previous.jpg'
            #print("\n #### ALIGNING : \n {} TO \n {}, \n SAVING AS : {}##### ".format(source,target,outname))
            save_warped(source,target,outname, model_aff, model_tps)
        
        if iterations > 1:
            #Updates the sets for the next iteration
            input_reversed = list_files(savepath, 'jpg', reverse_sort = True)        
            savepath = savepath.split('iteration')[0]+'iteration_'+str(iteration+1)+'/'
    print("Done aligning.")
    return savepath

    
def batch_align_to_target(sourcepath, target, model_aff ='', model_tps ='', ext = '.jpg'):
    """
    For a folder of images, align each frame to a target.
    Must provide full path to the target.
    For ex. : 
        Sourcepath = ./images/data123/frames/
        target = ./images/data123/frames/frame_15.jpg
    
    """
    input_list = list_files(sourcepath, ext,reverse_sort=False)
    #getting directories
    print(target)
    target_name = target.split(ext)[0].split('/')[-1]
    if target_name[-1]=='.':
        target_name = target_name[:-1]
    savepath = makedir(sourcepath+'aligned_to_'+target_name+'/')
    
    #CREATING DIRECTORY TO RECEIVE RESULTS
    shutil.copy2(target, savepath)    
    makedir(savepath)
    print("\n ### Aligning ###") 
    #Looping and doing the warp etc. 
    
    for index, file in enumerate(input_list):
        frame = 'frame_'+str(index)
        outname = savepath+frame+'aligned.jpg'
        save_warped(file, target, outname, model_aff, model_tps)
    
    print("Done aligning.")
    return savepath
    

def batch_align(sourcepath, target = '', model_aff = '', model_tps = '', ext = ''):
    """
    method to choose either to_previous or to_target (easier to use in run.py with options)
    """

    if target == '' or target == 'previous':
        print("\n###Aligning to previous frame.###\n")
        savepath = batch_align_to_previous(sourcepath, ext, model_aff, model_tps)
        return savepath 
    if target != '' and target != 'previous':
        print()
        savepath = batch_align_to_target(sourcepath, target, model_aff, model_tps)
        return savepath


def save_video(imagepath, reverse = False):
    """
    For a given set of frames, create a video and saves it.
    """
    frames = list_files(imagepath, 'jpg', reverse_sort = reverse)
    img_array = []
    print("Reading frames.")
    for index, frame in enumerate(frames):
        img = cv2.imread(frame)
        img_array.append(img)
    h,w,l = img_array[0].shape
    shape = (w,h)
    out = cv2.VideoWriter(imagepath+'aligned.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), 25, shape)
    print("Writing frames to video.")
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("Done")    