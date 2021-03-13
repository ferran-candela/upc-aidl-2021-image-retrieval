import os
import torch
import torch.nn as nn

class Model:
    TRAIN_FILE_NAME = 'trained_model.pt'

    def __init__(self, device, model_name, models_dir, model, optimizer=None, is_pretrained=True, input_resize=224, output_features=0):
        self.device = device
        self.model_name = model_name
        self.models_dir = models_dir
        self.model = model
        self.optimizer = optimizer
        self.is_pretrained = is_pretrained
        self.input_resize = input_resize
        self.output_features = output_features
    
    def get_model_name(self):
        return self.model_name
    
    def get_model(self):
        return self.model

    def get_input_resize(self):
        return self.input_resize
    
    def get_output_features(self):
        return self.output_features
    
    def get_device(self):
        return self.device

    def to_device(self):
        return self.model.to(self.device)
    
    def save_model(self):
        model_dir = self.get_model_dir(self.models_dir, self.model_name)

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        model_file_path = self.get_model_file_path(self.models_dir, self.model_name)

        optimizer_state_dict = None

        if(self.optimizer != None):
            optimizer_state_dict = self.optimizer.cpu().state_dict()

        model_checkpoint = {
            "model_name": self.model_name,
            "model_state_dict": self.model.cpu().state_dict(),
            "optimizer_state_dict": optimizer_state_dict,
            "is_pretrained": self.is_pretrained,
            "input_resize": self.input_resize
        }

        torch.save(model_checkpoint, model_file_path)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad) 

    @staticmethod
    def get_model_dir(models_dir, model_name):
        return os.path.join(models_dir, model_name)

    @staticmethod
    def get_model_file_path(models_dir, model_name):
        return os.path.join(Model.get_model_dir(models_dir, model_name), Model.TRAIN_FILE_NAME)

class ModelManager:
    def __init__(self, device, models_dir):
        self.models =   {  
                            'vgg16', # Documentation says input must be 224x224
                            'resnet50',
                            # 'inception_v3', # [batch_size, 3, 299, 299]
                            'inception_resnet_v2', #needs : [batch_size, 3, 299, 299]
                            'densenet161',
                            'efficient_net_b4'
                        }
                    
        self.device = device
        self.models_dir = models_dir

    # Returns the available models to use
    def get_model_names(self):
        return self.models

    def get_raw_model(self, model_name):
        device = self.device
        input_resize = 299
        is_pretrained = True
        output_features = 0

        if model_name == 'vgg16':
            from torchvision.models import vgg16
            # Input must be 224x224
            pretrained_model = vgg16(pretrained=True)
            # Just use the output of feature extractor and a globalAveragePooling
            model = nn.Sequential(
                pretrained_model.features,
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten()
            )
            
            input_resize = 224
            output_features = 512

        if model_name == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(pretrained=True)
            # Remove FC layer
            model.fc = nn.Identity()
            # Remove RELU in order to not lose negative features information
            model.layer4[2].relu = nn.Identity()

            input_resize = 299
            output_features = 2048
            
        if model_name == 'inception_v3':
            from torchvision.models import inception_v3
            model = inception_v3(pretrained=True)
            # Remove FC Layer
            model.fc = nn.Identity()

            input_resize = 299

        if model_name == 'inception_resnet_v2':
            from torch_inception_resnet_v2.model import InceptionResNetV2
            model = InceptionResNetV2(1000) #upper to PCA
            # Remove FC Layer
            model.dropout = nn.Identity()
            model.fc = nn.Identity()
            model.softmax = nn.Identity()

            input_resize = 299
            output_features = 1888


        if model_name == 'densenet161':
            # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel 
            # RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be 
            # loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            from torchvision.models import densenet161
            model = densenet161(pretrained=True)
            #Just use the output of feature extractor and ignore the classifier
            model.classifier = nn.Identity()

            input_resize = 224
            output_features = 2208
        
        if model_name == 'efficient_net_b4':

            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained('efficientnet-b4')

            model._dropout = nn.Identity()
            model._fc = nn.Identity()
            model._swish = nn.Identity() #Swish activation function 
            #model.set_swish(memory_efficient=False)

            input_resize = 224
            output_features = 1792
        
        return Model(device=self.device, model_name=model_name, models_dir=self.models_dir, model=model, is_pretrained=is_pretrained, input_resize=input_resize, output_features=output_features)
        
    def is_model_saved(self, model_name):
        return os.path.isfile(Model.get_model_file_path(self.models_dir, model_name))
    
    def load_pretrained_model(self, model_name):
        # TODO: Implement load model from checkpoint
        pass
