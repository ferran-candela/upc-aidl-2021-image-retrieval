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
            batch_size = 10
            
        if model_name == 'inception_v3':
            # TODO: To be defined
            pass

        if model_name == 'inception_resnet_v2':
            batch_size = 12

        if model_name == 'densenet161':
            batch_size = 6

        if model_name == 'efficient_net_b4':
            batch_size = 6

        return batch_size
    
    @staticmethod
    def get_feature_extraction_batch_size(model_name):
        batch_size = 8
        
        if model_name == 'vgg16':
            batch_size = 24

        if model_name == 'resnet50':
            batch_size = 10
            
        if model_name == 'inception_v3':
            # TODO: To be defined
            pass

        if model_name == 'inception_resnet_v2':
            batch_size = 12

        if model_name == 'densenet161':
            batch_size = 6

        if model_name == 'efficient_net_b4':
            batch_size = 6

        return batch_size

class FeaturesConfig:
    @staticmethod
    def get_PCA_size(model_name):
        PCA = 128
        
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

        return PCA


class ModelTrainConfig:
    TRAIN_TYPE = os.environ.get("TRAIN_TYPE") # transferlearning / scratch
    TRAIN_SIZE = os.environ.get("TRAIN_SIZE") # "all" / "divide"=train(60%), Eval and test (20%) / number=fixed size
    TEST_VALIDATE_SIZE = os.environ.get("TEST_VALIDATE_SIZE") # used only for train_size = fixed size
    NUM_CLASSES = 54 # used only for train
    PATIENCE = 1000  #Number of epochs to wait if no improvement and then stop the training.

    @staticmethod
    def get_learning_rate(model_name):
        lr = 0.001
        
        if model_name == 'vgg16':
            # TODO: To be defined
            pass

        if model_name == 'resnet50':
            lr = 0.0001
            
        if model_name == 'inception_v3':
            # TODO: To be defined
            pass

        if model_name == 'inception_resnet_v2':
            # TODO: To be defined
            pass

        if model_name == 'densenet161':
            # TODO: To be defined
            pass

        if model_name == 'efficient_net_b4':
            # TODO: To be defined
            pass

        return lr

    @staticmethod
    def get_num_epochs(model_name):
        nepochs = 20
        
        if model_name == 'vgg16':
            # TODO: To be defined
            pass

        if model_name == 'resnet50':
            nepochs = 10
            pass
            
        if model_name == 'inception_v3':
            # TODO: To be defined
            pass

        if model_name == 'inception_resnet_v2':
            # TODO: To be defined
            pass

        if model_name == 'densenet161':
            # TODO: To be defined
            pass

        if model_name == 'efficient_net_b4':
            # TODO: To be defined
            pass

        return nepochs



class RetrievalEvalConfig:
    # OPTIONS:
    ## 'FirstN'
    ## 'Random'
    ## TODO: 'List'
    GT_SELECTION_MODE = os.environ.get("GT_SELECTION_MODE")
    MAP_N_QUERIES = int(os.environ.get("MAP_N_QUERIES"))
    TOP_K_IMAGE = int(os.environ.get("TOP_K_IMAGE"))