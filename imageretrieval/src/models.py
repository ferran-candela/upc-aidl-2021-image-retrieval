import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from config import ModelTrainConfig

class ModelType:
    CLASSIFIER = 'classifier'
    FEATURE_EXTRACTOR = 'feature_extractor'

class Model:
    CHECKPOINT_EXTENSION = '.pt'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __init__(self, device, model_name, model_type, models_dir, model, optimizer=None, criterion=None, is_pretrained=True, input_resize=224, output_features=0, num_classes=0):
        self.device = device
        self.model_name = model_name
        self.model_type = model_type
        self.models_dir = models_dir
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.is_pretrained = is_pretrained
        self.input_resize = int(input_resize)
        self.output_features = int(output_features)
        self.num_classes = int(num_classes)
    
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
    
    def create_checkpoint(self, epoch=-1, min_loss=-1, max_acc=-1):
        optimizer_state_dict = None

        checkpoint = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "model_state_dict": self.model.state_dict(),
            "is_pretrained": self.is_pretrained,
            "input_resize": self.input_resize
        }

        if(self.model_type is ModelType.CLASSIFIER):
            optimizer_state_dict = None

            if(self.optimizer != None):
                optimizer_state_dict = self.optimizer.state_dict()

            checkpoint['optimizer_state_dict'] = optimizer_state_dict

            if(epoch != -1):
                checkpoint['epoch'] = epoch
                checkpoint['min_loss'] = min_loss
                checkpoint['max_acc'] = max_acc

        return checkpoint
    
    def save_model(self, epoch=-1):
        model_dir = self.get_model_dir(self.models_dir, self.model_name, epoch=epoch)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_file_path = self.get_model_file_path(self.models_dir, self.model_name, self.model_type, epoch)
        model_checkpoint = self.create_checkpoint(epoch)

        torch.save(model_checkpoint, model_file_path)
    
    def load_from_checkpoint(self, checkpoint=None, epoch=-1):
        if(checkpoint == None):
            checkpoint = torch.load(self.get_model_file_path(self.models_dir, self.model_name, self.model_type, epoch), map_location=self.device)
        # model_checkpoint = {
        #     "model_name": self.model_name,
        #     "model_type": self.model_type,
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

        if(checkpoint['model_type'] == ModelType.CLASSIFIER and checkpoint['optimizer_state_dict'] != None):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_input_transform(self):
        image_resized_size = self.get_input_resize()
        return transforms.Compose([
                transforms.Resize(image_resized_size),
                transforms.CenterCrop(image_resized_size),
                transforms.ToTensor(),
                self.normalize
            ])

    def get_train_transform(self):
        image_resized_size = self.get_input_resize()
        return transforms.Compose([
                transforms.Resize(image_resized_size),
                transforms.RandomCrop(image_resized_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize
            ])

    def get_current_model_dir(self, epoch=-1):
        return Model.get_model_dir(self.models_dir, self.model_name, epoch)
    
    @staticmethod
    def get_model_dir(models_dir, model_name, epoch=-1):
        model_dir = os.path.join(models_dir, model_name)

        if(epoch != -1):
            model_dir = os.path.join(model_dir, 'v' + str(epoch))

        return model_dir

    @staticmethod
    def get_model_file_path(models_dir, model_name, model_type, epoch=-1):
        model_dir = Model.get_model_dir(models_dir, model_name, epoch)
        return os.path.join(model_dir, model_type + Model.CHECKPOINT_EXTENSION)

class ModelManager:
    def __init__(self, device, models_dir):
        self.models =   [  
                            # 'vgg16', # Documentation says input must be 224x224
                            'resnet50',
                            # 'inception_v3', # [batch_size, 3, 299, 299]
                            # 'inception_resnet_v2', #needs : [batch_size, 3, 299, 299]
                            # 'densenet161',
                            # 'efficient_net_b4',
                            'resnet50_custom'
                            ]
                    
        self.device = device
        self.models_dir = models_dir

    # Returns the available models to use
    def get_model_names(self):
        return self.models

    def get_feature_extractor(self, model_name, checkpoint=None, load_from=ModelType.FEATURE_EXTRACTOR):

        if checkpoint is None and not self.is_model_saved(model_name, load_from):
            raise Exception('Model "{0}" checkpoint cannot be found.'.format(model_name))

        model = self.get_classifier(model_name)

        if(load_from == ModelType.CLASSIFIER):
            model.load_from_checkpoint(checkpoint)

        model = self.transform_to_feature_extractor(model)

        if(load_from == ModelType.FEATURE_EXTRACTOR):
            model.load_from_checkpoint(checkpoint)

        return model

    def transform_to_feature_extractor(self, model):
        model_name = model.get_name()
        model.model_type = ModelType.FEATURE_EXTRACTOR

        if model_name == 'vgg16':
            # Just use the output of feature extractor and a globalAveragePooling
            model.model = nn.Sequential(
                model.model.features,
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten()
            )

            model.output_features = 512

        if model_name == 'resnet50':
            # Remove FC layer
            model.model.fc = nn.Identity()
            # Remove RELU in order to not lose negative features information
            model.model.layer4[2].relu = nn.Identity()

            model.output_features = 2048            

        if model_name == 'inception_v3':
            # Remove FC Layer
            model.model.fc = nn.Identity()
            model.output_features = 512

        if model_name == 'inception_resnet_v2':
            # Remove FC Layer
            model.model.dropout = nn.Identity()
            model.model.fc = nn.Identity()
            model.model.softmax = nn.Identity()

            model.output_features = 1888

        if model_name == 'densenet161':
            #Just use the output of feature extractor and ignore the classifier
            model.model.classifier = nn.Identity()

            model.output_features = 2208
        
        if model_name == 'efficient_net_b4':
            model.model._dropout = nn.Identity()
            model.model._fc = nn.Identity()
            model.model._swish = nn.Identity() #Swish activation function 
            #model.model.set_swish(memory_efficient=False)

            model.output_features = 1792

        if model_name == 'resnet50_custom':
            # Remove FC layer
            model.model.fc = nn.Identity()

            model.output_features = 2048
    
        return model


    def get_classifier(self, model_name, load_from_checkpoint=False, checkpoint=None, epoch=-1):
        device = self.device
        input_resize = 299
        is_pretrained = True
        criterion = None
        optimizer = None

        if model_name == 'vgg16':
            from torchvision.models import vgg16
            # Input must be 224x224
            model = vgg16(pretrained=True)
            input_resize = 224

        if model_name == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(pretrained=True)
            # Remove RELU in order to not lose negative features information
            model.layer4[2].relu = nn.Identity()

            input_resize = 299
            
        if model_name == 'inception_v3':
            from torchvision.models import inception_v3
            model = inception_v3(pretrained=True)
            input_resize = 299

        if model_name == 'inception_resnet_v2':
            from torch_inception_resnet_v2.model import InceptionResNetV2
            model = InceptionResNetV2(1000) #upper to PCA

            input_resize = 299

        if model_name == 'densenet161':
            # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel 
            # RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be 
            # loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
            from torchvision.models import densenet161
            model = densenet161(pretrained=True)

            input_resize = 224
        
        if model_name == 'efficient_net_b4':

            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained('efficientnet-b4')

            input_resize = 224

        if model_name == 'resnet50_custom':
            from torchvision.models import resnet50
            model = resnet50(pretrained=True)

            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, ModelTrainConfig.NUM_CLASSES)

            criterion = nn.CrossEntropyLoss()
            lr = ModelTrainConfig.get_learning_rate(model_name=model_name)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

            is_pretrained = False
            input_resize = 299

        classifier = Model(device=self.device, model_name=model_name, model_type=ModelType.CLASSIFIER, \
                models_dir=self.models_dir, model=model, is_pretrained=is_pretrained, optimizer=optimizer, \
                criterion=criterion, num_classes=ModelTrainConfig.NUM_CLASSES, input_resize=input_resize)

        

        if(load_from_checkpoint):
            classifier.load_from_checkpoint(checkpoint, epoch)
            
        return classifier

    def is_model_saved(self, model_name, model_type, epoch=-1):
        return os.path.isfile(Model.get_model_file_path(self.models_dir, model_name, model_type, epoch))

    def get_model_dir(self, model_name):
        return Model.get_model_dir(self.models_dir, model_name)
