from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import numpy.matlib
from geotnf.transformation import GeometricTnf

def featureL2Norm(feature):
    epsilon = 1e-6
    #        print(feature.size())
    #        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature,norm)

class FeatureExtraction(torch.nn.Module):
    def __init__(self, train_fe=False, feature_extraction_cnn='vgg', normalization=True, 
                 last_layer='', use_cuda=True, pre_trained_bool=True):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        if feature_extraction_cnn == 'vgg':
            self.model = models.vgg16(pretrained=pre_trained_bool)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers=['conv1_1','relu1_1','conv1_2','relu1_2','pool1','conv2_1',
                         'relu2_1','conv2_2','relu2_2','pool2','conv3_1','relu3_1',
                         'conv3_2','relu3_2','conv3_3','relu3_3','pool3','conv4_1',
                         'relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','pool4',
                         'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','pool5']
            if last_layer=='':
                last_layer = 'pool4'
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx+1])

        if feature_extraction_cnn == 'resnet101':
            self.model = models.resnet101(pretrained=pre_trained_bool)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer=='':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]
            
            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx+1])

        if 'wormbrain' in feature_extraction_cnn:
            num = feature_extraction_cnn.split('_')[1]
            if num == '1' : 
                act = nn.LeakyReLU()
            elif num == '2' : 
                act = nn.ELU()
            self.model = nn.Sequential(nn.Conv2d(3,32, kernel_size = (5,5)),
                                       nn.BatchNorm2d(32),
                                       act,
                                       nn.Conv2d(32,64, kernel_size = (5,5)),
                                       nn.BatchNorm2d(64),
                                       nn.MaxPool2d(kernel_size=2),
                                       nn.ReLU(),
                                       nn.Conv2d(64,128, kernel_size = (5,5)),
                                       nn.BatchNorm2d(128),
                                       act,
                                       nn.Conv2d(128,256, kernel_size = (5,5)),
                                       nn.MaxPool2d(kernel_size=2),
                                       nn.ReLU(),
                                       nn.Conv2d(256,512, kernel_size = (5,5)),
                                       nn.BatchNorm2d(512),
                                       act,
                                       nn.Conv2d(512,512, kernel_size = (5,5)),
                                       nn.MaxPool2d(kernel_size=3),
                                       nn.ReLU())
        if not train_fe:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.model = self.model.cuda()
        
    def forward(self, image_batch):
        features = self.model(image_batch)
        if self.normalization:
            features = featureL2Norm(features)
        return features
    
class FeatureCorrelation(torch.nn.Module):
    def __init__(self,shape='3D',normalization=True,matching_type='correlation'):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.matching_type=matching_type
        self.shape=shape
        self.ReLU = nn.ReLU()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        if self.matching_type=='correlation':
            if self.shape=='3D':
                # reshape features for matrix multiplication
                feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
                feature_B = feature_B.view(b,c,h*w).transpose(1,2)
                # perform matrix mult.
                feature_mul = torch.bmm(feature_B,feature_A)
                # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
                correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
            elif self.shape=='4D':
                # reshape features for matrix multiplication
                feature_A = feature_A.view(b,c,h*w).transpose(1,2) # size [b,c,h*w]
                feature_B = feature_B.view(b,c,h*w) # size [b,c,h*w]
                # perform matrix mult.
                feature_mul = torch.bmm(feature_A,feature_B)
                # indexed [batch,row_A,col_A,row_B,col_B]
                correlation_tensor = feature_mul.view(b,h,w,h,w).unsqueeze(1)
            
            if self.normalization:
                correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))
        
            return correlation_tensor

        if self.matching_type=='subtraction':
            return feature_A.sub(feature_B)
        
        if self.matching_type=='concatenation':
            return torch.cat((feature_A,feature_B),1)

class FeatureRegression(nn.Module):
    def __init__(self, output_dim, use_cuda=True, batch_normalization=True, 
                 kernel_sizes=[7,5,5], channels=[225,128,64], 
                 feature_regression = 'base', p_dropout=0):
        super(FeatureRegression, self).__init__()
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers-1): # last layer is linear 
            k_size = kernel_sizes[i]
            ch_in = channels[i]
            ch_out = channels[i+1]            
            nn_modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=k_size, padding=0))
            if batch_normalization:
                nn_modules.append(nn.BatchNorm2d(ch_out))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)        
        
        #Old regressor
        if feature_regression == 'base':
            self.linear = nn.Linear(ch_out * kernel_sizes[-1] * kernel_sizes[-1], output_dim)
        
        #New things added by Richie for FR
        #I've found the old regressor to be too simple (I.e. it is literally a linear predictor)
        #I've tried adding some complexity as to try to better fit the data.
        #It is a standard 4 layers MLP with ReLU as activation, BatchNorm and Dropout for regularization, and a final linear layer with no activation functions for prediction
        elif feature_regression == 'deep':
            self.Dropout = nn.Dropout(p_dropout)
            self.linear = nn.Sequential(
                nn.Linear(ch_out * kernel_sizes[-1] * kernel_sizes[-1], 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024), 
                self.Dropout,
                nn.Linear(1024,512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                self.Dropout,
                nn.Linear(512,256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                self.Dropout,
                nn.Linear(256,128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                self.Dropout,
                nn.Linear(128,64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                self.Dropout,
                nn.Linear(64, output_dim)
            )
            
        #As I (qualitatively) noticed that the network was overfitting and was maybe not utilizing correlated features properly, I decided to simplify the feature regressor as well as 
        #using a Softmax in the first layer, to get the features into range (0,1) and compute a "probability that it will be useful" for geometric transformation features estimation
        elif feature_regression == 'simpler':
            self.Dropout = nn.Dropout(p_dropout)
            self.linear = nn.Sequential(
                nn.Linear(ch_out * kernel_sizes[-1] * kernel_sizes[-1], 512),
                nn.BatchNorm1d(512), 
                nn.Softmax(dim=1),
                self.Dropout,
                
                nn.Linear(512,128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                self.Dropout,
                
                nn.Linear(128,output_dim))
            ### changes stops here
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear(x)
        return x
    
    
class CNNGeometric(nn.Module):
    def __init__(self, output_dim, 
                 feature_extraction_cnn='vgg', 
                 feature_extraction_last_layer='',
                 return_correlation=False,  
                 fr_kernel_sizes=[7,5,5],
                 fr_channels=[225,128,64],
                 feature_regression = 'base',
                 fr_dropout = 0.2,
                 feature_self_matching=False,
                 normalize_features=True,
                 normalize_matches=True, 
                 batch_normalization=True, 
                 train_fe=False,
                 use_cuda=True,
                 pretrained=True,
                 matching_type='correlation'):
        
        super(CNNGeometric, self).__init__()
        self.use_cuda = use_cuda
        self.feature_self_matching = feature_self_matching
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.return_correlation = return_correlation
        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe,
                                                   feature_extraction_cnn=feature_extraction_cnn,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,
                                                   use_cuda=self.use_cuda,
                                                  pre_trained_bool=pretrained)
        if pretrained == False:
            print("Loading model : Pretrained = ",str(pretrained))
            print("Starting training from scratch.")
        self.FeatureCorrelation = FeatureCorrelation(shape='3D',normalization=normalize_matches,matching_type=matching_type)        
        

        self.FeatureRegression = FeatureRegression(output_dim,
                                                   use_cuda=self.use_cuda,
                                                   kernel_sizes=fr_kernel_sizes,
                                                   channels=fr_channels,
                                                   feature_regression = feature_regression,
                                                   p_dropout = fr_dropout,
                                                   batch_normalization=batch_normalization)


        self.ReLU = nn.ReLU(inplace=True)
    
    # used only for foward pass at eval and for training with strong supervision
    def forward(self, tnf_batch): 
        # feature extraction
        feature_A = self.FeatureExtraction(tnf_batch['source_image'])
        feature_B = self.FeatureExtraction(tnf_batch['target_image'])
        # feature correlation
        correlation = self.FeatureCorrelation(feature_A,feature_B)
        # regression to tnf parameters theta
        theta = self.FeatureRegression(correlation)
        
        if self.return_correlation:
            return (theta,correlation)
        else:
            return theta

