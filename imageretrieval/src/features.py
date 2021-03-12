import os
import torch

from config import DebugConfig, DeviceConfig

device = DeviceConfig.DEVICE
DEBUG = DebugConfig.DEBUG

# TODO: Move code from feature extraction and load features from precalculated features
class FeaturesManager:
    def __init__(self):
        self.models =   {  
                            'vgg16', # Documentation says input must be 224x224
                            'resnet50',
                            'inception_v3', # [batch_size, 3, 299, 299]
                            'inception_resnet_v2', #needs : [batch_size, 3, 299, 299]
                            'densenet161',
                            'efficient_net_b4'
                        }
                    
        self.device = device
        self.models_dir = models_dir

    # Returns the available models to use
    def get_model_names(self):
        return self.models

    def get_model_features(self, model_name):
        if model_name == 'vgg16':
            pass

        if model_name == 'resnet50':
            pass
            
        if model_name == 'inception_v3':
            pass

        if model_name == 'inception_resnet_v2':
            pass


        if model_name == 'densenet161':
            pass
        
        if model_name == 'efficient_net_b4':
            pass


def extract_features(config):
    pass
