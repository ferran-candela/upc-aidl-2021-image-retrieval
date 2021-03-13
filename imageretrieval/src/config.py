import os
import torch

class DebugConfig:
    DEBUG = False    
    if os.environ.get("DEBUG") == 'True':
        DEBUG = True

class DeviceConfig:
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class FoldersConfig:
    DATASET_BASE_DIR = os.environ.get("DATASET_BASE_DIR")
    DATASET_LABELS_DIR = os.environ.get("DATASET_LABELS_DIR")
    WORK_DIR = os.environ.get("WORK_DIR")
    LOG_DIR = os.environ.get("LOG_DIR")

class ModelBatchSizeConfig:
    @staticmethod
    def get_batch_size(model_name):
        batch_size = 8
        
        if model_name == 'vgg16':
            batch_size = 24

        if model_name == 'resnet50':
            batch_size = 12
            
        if model_name == 'inception_v3':
            # TODO: To be defined
            pass

        if model_name == 'inception_resnet_v2':
            batch_size = 12

        if model_name == 'densenet161':
            batch_size = 6

        if model_name == 'efficient_net_b4':
            batch_size = 8

        return batch_size

class ModelTrainConfig:
    TRAIN_SIZE = os.environ.get("TRAIN_SIZE") # "all" / "divide"=train(60%), Eval and test (20%) / number=fixed size
    TEST_VALIDATE_SIZE = os.environ.get("TEST_VALIDATE_SIZE") # used only for train_size = fixed size