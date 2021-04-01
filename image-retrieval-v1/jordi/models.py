import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from config import ModelTrainConfig


class Model:
    #TRAIN_FILE_NAME = 'trained_model.pt'
    TRAIN_SCRATCH_FILE_NAME = 'trained_scratch.pt'
    TRAIN_TRANSFER_LEARNIG_FILE_NAME = 'trained_transfer_learning.pt'

    def __init__(self, device, model_name, models_dir, model, optimizer=None, criterion=None, is_pretrained=True, input_resize=224, output_features=0, num_classes=0):
        self.device = device
        self.model_name = model_name
        self.models_dir = models_dir
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.is_pretrained = is_pretrained
        self.input_resize = input_resize
        self.output_features = output_features
        self.num_classes = num_classes
    
    def get_name(self):
        return self.model_name
    
    def get_model(self):
        return self.model

    def get_num_classes(self):
        return self.num_classes

    def get_criterion(self):
        return self.criterion

    def get_optimizer(self):
        return self.optimizer

    def get_input_resize(self):
        return self.input_resize
    
    def get_output_features(self):
        return self.output_features
    
    def get_device(self):
        return self.device

    def to_device(self):
        return self.model.to(self.device)
    
    def get_checkpoint(self):
        optimizer_state_dict = None

        if(self.optimizer != None):
            optimizer_state_dict = self.optimizer.state_dict()

        return {
            "model_name": self.model_name,
            "model_state_dict": self.model.cpu().state_dict(),
            "optimizer_state_dict": optimizer_state_dict,
            "is_pretrained": self.is_pretrained,
            "input_resize": self.input_resize
        }
    
    def save_model(self):
        model_dir = self.get_model_dir(self.models_dir, self.model_name)

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        model_file_path = self.get_model_file_path(self.models_dir, self.model_name)
        model_checkpoint = self.get_checkpoint()

        torch.save(model_checkpoint, model_file_path)
    
    def load_from_checkpoint(self, checkpoint=None):
        if(checkpoint == None):
            checkpoint = torch.load(self.get_model_file_path(self.models_dir, self.model_name))
        # model_checkpoint = {
        #     "model_name": self.model_name,
        #     "model_state_dict": self.model.cpu().state_dict(),
        #     "optimizer_state_dict": optimizer_state_dict,
        #     "is_pretrained": self.is_pretrained,
        #     "input_resize": self.input_resize
        # }
        model = self.get_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        self.to_device()
        model.eval()

        self.input_resize = checkpoint['input_resize']
        self.is_pretrained = checkpoint['is_pretrained']

        if(self.optimizer != None and checkpoint['optimizer_state_dict'] != None):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_input_transform(self):
        image_resized_size = self.get_input_resize() + 32
        return transforms.Compose([
                transforms.Resize(image_resized_size),
                transforms.CenterCrop(image_resized_size - 32),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    @staticmethod
    def get_model_dir(models_dir, model_name):
        return os.path.join(models_dir, model_name)

    @staticmethod
    def get_model_file_path(models_dir, model_name):
        if ModelTrainConfig.TRAIN_TYPE == "transferlearning":
            return os.path.join(Model.get_model_dir(models_dir, model_name), Model.TRAIN_TRANSFER_LEARNIG_FILE_NAME)
        elif ModelTrainConfig.TRAIN_TYPE == "scratch":
            return os.path.join(Model.get_model_dir(models_dir, model_name), Model.TRAIN_SCRATCH_FILE_NAME)
        else:
            raise Exception('Train type "{0}" unknow.'.format(ModelTrainConfig.TRAIN_TYPE))


class ModelManager:
    def __init__(self, device, models_dir):
        self.models =   {  
                            #'vgg16', # Documentation says input must be 224x224
                            'resnet50',
                            #'inception_v3', # [batch_size, 3, 299, 299]
                            #'inception_resnet_v2', #needs : [batch_size, 3, 299, 299]
                            #'densenet161',
                            #'efficient_net_b4'
                        }
                    
        self.device = device
        self.models_dir = models_dir

    # Returns the available models to use
    def get_model_names(self):
        return self.models

    def get_transferlearning_model(self, model_name):
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

    def get_scratch_model(self, model_name):
        device = self.device
        is_pretrained = False

        if model_name == 'vgg16':
            from torchvision.models import vgg16
            model = vgg16(pretrained=True)

            #TODO:
            #Define last layer for classifier
            #num_features = 
            #feature_classifier = nn.Linear(num_features, ModelTrainConfig.NUM_CLASSES)
            #features_model = model.features
            #for layer in features_model [:24]:  # Freeze layers 0 to 23
            #    for param in layer.parameters():
            #        param.requires_grad = False
            #for layer in feature_extractor[24:]:  # Train layers 24 to 30
            #    for param in layer.parameters():
            #        param.requires_grad = True
            #model = nn.Sequential(
                            #features_model,
                            #nn.Flatten(),
                            #feature_classifier
                        #)
        if model_name == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(pretrained=True)

            #Freeze all
            for layer in model.children():
                for param in layer.parameters():
                    param.requires_grad = False

            #unfreeze layer4
            for param in model.layer4.parameters():
                param.requires_grad = True
            
            #Define last layer for classifier
            num_features = model.fc.in_features
            feature_classifier = nn.Linear(num_features, ModelTrainConfig.NUM_CLASSES, bias=True)
            
            model.fc = feature_classifier

            #print(model)

            criterion = nn.CrossEntropyLoss()
            lr = ModelTrainConfig.get_learning_rate(model_name=model_name)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)


        # if model_name == 'inception_v3':
        #     from torchvision.models import inception_v3
        #     model = inception_v3(pretrained=True)

        # if model_name == 'inception_resnet_v2':
        #     from torch_inception_resnet_v2.model import InceptionResNetV2
        #     model = InceptionResNetV2(1000) #upper to PCA


        # if model_name == 'densenet161':
        #     from torchvision.models import densenet161
        #     model = densenet161(pretrained=True)
        
        # if model_name == 'efficient_net_b4':

        #     from efficientnet_pytorch import EfficientNet
        #     model = EfficientNet.from_pretrained('efficientnet-b4')

        return Model(device=self.device, model_name=model_name, models_dir=self.models_dir, model=model, is_pretrained=is_pretrained, optimizer=optimizer, criterion=criterion, num_classes=ModelTrainConfig.NUM_CLASSES)

    def is_model_saved(self, model_name):
        return os.path.isfile(Model.get_model_file_path(self.models_dir, model_name))

    def get_model_dir(self, model_name):
        return Model.get_model_dir(self.models_dir, model_name)
    
    def load_from_checkpoint(self, model_name, checkpoint=None):
        if not self.is_model_saved(model_name):
            raise Exception('Model "{0}" checkpoint cannot be found.'.format(model_name))

        model = self.get_transferlearning_model(model_name)
        model.load_from_checkpoint(checkpoint)

        return model