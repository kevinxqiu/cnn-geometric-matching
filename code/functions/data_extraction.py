"""
Used in the run.py scripts to extract data if needed (ex if data is a video or a .h5 file)
"""

### IMPORTS ###
from util.handle_files import * #Contains makedir() and list_files()
import cv2
import warnings
import h5py as h5
import numpy as np
import torch
import torchvision
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, Resize, ToPILImage
from PIL import Image

warnings.filterwarnings('ignore')

#Values for normalization to convert to float.
NORM_VALUES = [1543, 1760]

def extract_frames_video(relative_filepath, extension):
    """
    input : relative_filepath (str), folder containing the video
            extension (str), extension of the video 
            
    Extracts frames from a video and saves them.
    Frames will be saved in a folder called frame in the same folder as the video,
    So far, assumes that you have only one video in the folder.
    ex:
        extension is 'mp4'
        video is contained in /directory/video/
        frames will be saved in /directory/video/frames/
    """
    #Getting the video filepath
    video = list_files(relative_filepath, extension)
    print("\n",video, "\n")
    cap = cv2.VideoCapture(video)
    target_directory = makedir(relative_filepath+'/frames/')
    output_filename = target_directory+'frame_'
    print("Reading frames and saving to folder.")
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read() #Test whether a frame was read correctly
        if ret == False:
            break
        cv2.imwrite(output_filename+str(i)+'.jpg',frame)
        i+=1
    print("Done. A total of {} frames were saved.".format(i))
    cap.release()
    cv2.destroyAllWindows()
    
def extract_frames_h5(file, save_path, channel = 'rgb', resize=None):
    """
    reads an h5 file and extracts all the frames within
    Channel specify the channels to save
    can be 'red', 'green' or 'rgb'.
    paths are relative
    """
    if type(file) == str:
        file = h5.File(file)
    if resize: #using torch.transforms
        res = Resize((resize,resize))
    maxvalues = [1543, 1760]
    #due to the structure, we keep only the files that are numeric
    filenames = [x for x in file.keys() if x.isnumeric()]
    filenames.sort(key=int) #Sort with key = int
    
    nb_channels = file['0/frame'].shape[0]
    w_h = file['0/frame'].shape[1]
    z = file['0/frame'].shape[3]
    channel = channel.lower()
    if ((nb_channels==1) and (channel=='rgb')):
        print("No frame extracted. Exiting.\nRGB specified, but only one color channel was found.")
        return
    
    PATH = makedir(save_path)
    PATH = makedir(save_path+channel+'_frames')
    print("Extracting {} frames.".format(channel))
    #For loop to extract all the frames
    for index, number in enumerate(filenames):
        name = number+'/frame'
        frame = file[name]
        #max projecting
        
        #creating RGB images with artificial blue "background" channel
        if nb_channels==2 and channel=='rgb':
            temp = torch.as_tensor(np.max(frame,axis=3),dtype=torch.float32) 
            blue = torch.full((1,512,512),0, dtype=torch.float32)
            image_tensor = torch.cat((temp,blue))
            #Normalizing to range [0,1]
            image_tensor[0,...].div_(maxvalues[0])
            image_tensor[1,...].div_(maxvalues[1])
        
        if channel!='rgb':
            index = 0
            if channel == 'green':
                index = 1
            image_tensor = torch.as_tensor(np.max(frame,axis=3),dtype=torch.float32)
            if nb_channels!=1:
                image_tensor[index,...].div_(maxvalues[index])
                
            image_tensor = image_tensor[index:index+1,...]
    
        if resize:
            #Using this convoluted way due to some errors within pytorch
            #with how it handles tensors/PILImage for Resize.
            to_pili_image = ToPILImage()
            image_tensor = to_pili_image(image_tensor)
            image_tensor = ToTensor()(res(image_tensor))
            
        save_image(image_tensor, PATH+'/frame_'+number+'.jpg')
            
    print("Done. A total of {} frames were saved.".format(len(filenames)))
    return PATH