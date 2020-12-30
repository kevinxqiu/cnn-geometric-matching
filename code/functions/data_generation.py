"""
Contains functions necessary for generating a training dataset (with or without augmentation.)
"""

### IMPORTS ###
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np
import glob
from PIL import Image
from util.handle_files import *
import csv, os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
#Uniformly sampling the background value for which we use to fill black spots
def someAug():
    bg = iap.Uniform(16,18)
    #Creating a series of augmentations
    shearXY = iaa.Sequential([
                               iaa.ShearY(iap.Uniform(-10,10),cval=bg),
                               iaa.ShearX(iap.Uniform(-10,10),cval=bg)
                           ])
    rotate = iaa.Rotate(rotate = iap.Choice([-30,-15,15,30],
                                            p=[0.25,0.25,0.25,0.25]),
                        cval = bg)
    
    pwAff = iaa.PiecewiseAffine(scale=(0.01,0.06),
                                cval = bg)
    
    affine = iaa.Affine(scale={"x":iap.Uniform(1.1,1.2),
                                           "y":iap.Uniform(1.1,1.2)},
                        cval = bg)
    
    noise = iaa.AdditiveGaussianNoise(loc=0, scale=(0,0.025*255))
    #Using SomeOf to randomly select some augmentations
    someAug = iaa.SomeOf(iap.Choice([2,3,4], p = [1/3,1/3,1/3]),
                         [
                             affine,
                             shearXY,
                             pwAff,
                             rotate,
                             noise
                         ], random_order=True)
    return someAug
    
def augment_images(input_path, augs=someAug(), times_target=1, source = False):
    #Load input image
    imgs = []
    print("Loading images from :", input_path)
    for f in glob.iglob(input_path+'*.jpg'):
        imgs.append(np.asarray(Image.open(f)))
        
    #Initialize list to receive augmented target images
    #For each time we want to augment (since random augmentation)
    target_imgs = []
    #Create the directory to receive the images in
    if source == True:
        target_path = input_path
    elif source == False:
        target_path = input_path+'/augmented/'
        target_path = makedir(target_path)
    
    #Applying (times_target) times the augmentation
    for i in range(times_target):
        print("Augmenting", i, "/",times_target-1)
        target_imgs += augs.augment_images(imgs)
        
    print("Saving images")
    for idx, augmented in enumerate(target_imgs):
        img = Image.fromarray(augmented)
        img.save(target_path+'augmented_frame_'+str(idx)+'.jpg')
    print("Done")
    
def create_csv_split(image_dir, csv_dir='', ratio=0.2):
    """
    Create csv files containing the filenames to use in each of the 
    training and validation set.
    """
    data=[]
    csvdir = image_dir.split('datasets/')[1].split('/TrainVal')[0]
    csv_dir = makedir('./training_data/'+csvdir+'-random/')
    #else if csv_dir != '': 
    #    csvdir = makedir(csv_dir)
    #
    print("Wrinting CSV files.")
    with open(csv_dir+'data.csv', 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(['image'])
        for filename in os.listdir(image_dir):
            data.append('TrainVal/'+filename)
            writer.writerow(data)
            data=[]
    writeFile.close()
    
    df = pd.read_csv(csv_dir+'data.csv')
    train, val = train_test_split(df, test_size=ratio)
    print("Done")
    train.to_csv(csv_dir+'train.csv', index=False)
    val.to_csv(csv_dir+'val.csv', index=False)
