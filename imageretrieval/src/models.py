import os
import torch
import torch.nn as nn

class Model:
    def __init__(self, device, model_name, model, is_pretrained):
        self.device = device
        self.model_name = model_name
        self.model = model
        self.is_pretrained = is_pretrained
    
    def get_model_name(self):
        return self.model_name
    
    def get_model(self):
        return self.model
    
    def is_pretrained(self):
        return self.is_pretrained
    
    def to_device(self):
        return self.model.to(self.device)
    
    def save_model(self, model_dir):
        os.path.join(model_dir, 'model.pt')
        torch.save(self.model.state_dict(), model_dir)

class ModelManager:
    def __init__(self, device, models_folder):
        self.models =   {  
                            'vgg16', # Documentation says input must be 224x224
                            'resnet50',
                            'inception_v3', # [batch_size, 3, 299, 299]
                            'inception_resnet_v2', #needs : [batch_size, 3, 299, 299]
                            'densenet161'
                        }
                    
        self.device = device
        self.models_folder = models_folder

    # Returns the available models to use
    def get_model_names(self):
        return self.models

    def get_model(self, model_name):
        if model_name == 'vgg16':
            from torchvision.models import vgg16
            # Input must be 224x224
            model = vgg16(pretrained=True)
            # Just use the output of feature extractor and ignore the classifier
            model.classifier = nn.Identity()

        if model_name == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(pretrained=True)
            # Remove FC layer
            model.fc = nn.Identity()
            # Remove RELU
            model.layer4[2].relu = nn.Identity()
            
        if model_name == 'inception_v3':
            from torchvision.models import inception_v3
            model = inception_v3(pretrained=True)
            # Remove FC Layer
            model.fc = nn.Identity()

        if model_name == 'inception_resnet_v2':
            from torch_inception_resnet_v2.model import InceptionResNetV2
            model = InceptionResNetV2(1000) #upper to PCA
            # Remove FC Layer
            model.dropout = nn.Identity()
            model.fc = nn.Identity()
            model.softmax = nn.Identity()

        if model_name == 'densenet161':
            from torchvision.models import densenet161
            model = densenet161(pretrained=True)
            #Just use the output of feature extractor and ignore the classifier
            model.classifier = nn.Identity()
        
        return Model(self.device, model_name, model)
    
    def load_model_checkpoint(self):
        pass

    def save_model(self, model):
        model_dir = os.path.join(self.models_folder, model.get_model_name())
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        model.save_model(model_dir)