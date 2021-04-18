import os
import torch

class DebugConfig:
    DEBUG = False    
    if os.environ.get("DEBUG") == 'True':
        DEBUG = True

class DeviceConfig:
    DEVICE = torch.device("cpu")

    if(os.environ.get("DEVICE") == 'cuda' and torch.cuda.is_available()):
        DEVICE = torch.device("cuda")

class FoldersConfig:
    DATASET_BASE_DIR = os.environ.get("DATASET_BASE_DIR")
    DATASET_LABELS_DIR = os.environ.get("DATASET_LABELS_DIR")
    WORK_DIR = os.environ.get("WORK_DIR")
    LOG_DIR = os.environ.get("LOG_DIR")
    RESOURCES_DIR = os.environ.get("RESOURCES_DIR")

class ModelBatchSizeConfig:
    @staticmethod
    def get_batch_size(model_name):
        batch_size = 8
        
        if model_name == 'vgg16':
            batch_size = 16

        if model_name == 'resnet50':
            batch_size = 8
            
        if model_name == 'inception_v3':
            batch_size = 8

        if model_name == 'inception_resnet_v2':
            batch_size = 8

        if model_name == 'densenet161':
            batch_size = 6

        if model_name == 'efficient_net_b4':
            batch_size = 6

        if model_name == 'resnet50_custom':
            batch_size = 10
        
        if model_name == 'vgg16_custom':
            batch_size = 20

        return batch_size
    
    @staticmethod
    def get_feature_extraction_batch_size(model_name):
        batch_size = 8
        
        if model_name == 'vgg16':
            batch_size = 20

        if model_name == 'resnet50':
            batch_size = 10
            
        if model_name == 'inception_v3':
            batch_size = 12

        if model_name == 'inception_resnet_v2':
            batch_size = 12

        if model_name == 'densenet161':
            batch_size = 6

        if model_name == 'efficient_net_b4':
            batch_size = 6

        if model_name == 'resnet50_custom':
            batch_size = 10

        return batch_size

class FeaturesConfig:
    @staticmethod
    def get_PCA_size(model_name):
        PCA = 128
        
        if model_name == 'vgg16':
            PCA = 10

        if model_name == 'resnet50':
            PCA = 6
            
        if model_name == 'inception_v3':
            PCA = 6

        if model_name == 'inception_resnet_v2':
            PCA = 8

        if model_name == 'densenet161':
            PCA = 9

        if model_name == 'efficient_net_b4':
            pass

        if model_name == 'vgg16_custom':
            PCA = 44

        if model_name == 'resnet50_custom':
            pass

        return PCA


class ModelTrainConfig:
    DATASET_USEDNAME = os.environ.get("DATASET_USEDNAME")  # deepfashion / fashionproduct
    TRAIN_SIZE = os.environ.get("TRAIN_SIZE") # "all" / "divide"=train(60%), Eval and test (20%) / number=fixed size
    TEST_VALIDATE_SIZE = os.environ.get("TEST_VALIDATE_SIZE") # used only for train_size = fixed size
    # Using both filters
    # Filter 1: Remove unknown clouths, remove all but master category == Apparel || category == Footwear
    # Filter 2: At least 100 images for each category
    # Results: 
    # Tshirts         7065
    # Shirts          3215
    # Casual Shoes    2845
    # Sports Shoes    2036
    # Tops            1762
    # Heels           1323
    # Flip Flops       914
    # Sandals          897
    # Formal Shoes     637
    # Jeans            608
    # Shorts           547
    # Trousers         530
    # Flats            500
    # Dresses          464
    # Track Pants      304
    # Sweatshirts      285
    # Sweaters         277
    # Jackets          258
    # Nightdress       189
    # Leggings         177
    # Night suits      141
    # Skirts           128
    #NUM_CLASSES = 22
    NUM_CLASSES = 54 # used only for train
    PATIENCE = 1000  #Number of epochs to wait if no improvement and then stop the training.
    TOP_K_AQE = 15

    @staticmethod
    def get_learning_rate(model_name):
        lr = 0.0001
        
        if model_name == 'vgg16_custom':
            lr = 0.0001
            pass

        if model_name == 'resnet50_custom':
            lr = 0.0001
            
        if model_name == 'inception_v3_custom':
            # TODO: To be defined
            pass

        if model_name == 'inception_resnet_v2_custom':
            # TODO: To be defined
            pass

        if model_name == 'densenet161_custom':
            # TODO: To be defined
            pass

        if model_name == 'efficient_net_b4_custom':
            # TODO: To be defined
            pass

        return lr

    @staticmethod
    def get_num_epochs(model_name):
        nepochs = 20
        
        if model_name == 'vgg16_custom':
            # TODO: To be defined
            pass

        if model_name == 'resnet50_custom':
            nepochs = 20
            
        if model_name == 'inception_v3_custom':
            # TODO: To be defined
            pass

        if model_name == 'inception_resnet_v2_custom':
            # TODO: To be defined
            pass

        if model_name == 'densenet161_custom':
            # TODO: To be defined
            pass

        if model_name == 'efficient_net_b4_custom':
            # TODO: To be defined
            pass

        return nepochs



class RetrievalEvalConfig:
    # OPTIONS:
    ## 'FirstN'
    ## 'Random'
    ## 'List'
    GT_SELECTION_MODE = os.environ.get("GT_SELECTION_MODE")
    MAP_N_QUERIES = int(os.environ.get("MAP_N_QUERIES")) if (os.environ.get("MAP_N_QUERIES") is not None) else 0
    TOP_K_IMAGE = int(os.environ.get("TOP_K_IMAGE")) if (os.environ.get("TOP_K_IMAGE") is not None) else 0
    PCA_ACCURACY_TYPE = os.environ.get("PCA_ACCURACY_TYPE")
    QUERIES_PER_LABEL = int(os.environ.get("QUERIES_PER_LABEL")) if (os.environ.get("QUERIES_PER_LABEL") is not None) else 0