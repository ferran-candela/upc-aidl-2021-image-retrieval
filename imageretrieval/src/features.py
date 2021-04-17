import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from imageretrieval.src.config import DebugConfig, DeviceConfig, FoldersConfig, ModelBatchSizeConfig, ModelTrainConfig, FeaturesConfig
from imageretrieval.src.dataset import DatasetManager, FashionProductDataset
from imageretrieval.src.models import ModelManager, ModelType
from imageretrieval.src.utils import ProcessTime, LogFile

device = DeviceConfig.DEVICE
DEBUG = DebugConfig.DEBUG

class FeaturesManager:
    RAW_FEATURES_FILE_NAME = 'raw_features.pt'
    NORM_FEATURES_FILE_NAME = 'normalized_features.pt'
    AQE_FEATURES_FILE_NAME = 'aqe_features.pt'

    def __init__(self, device, model_manager):
        self.device = device
        self.model_manager = model_manager

    def get_raw_features_file_path(self, model_name):
        model_dir = self.model_manager.get_model_dir(model_name)
        return os.path.join(model_dir, self.RAW_FEATURES_FILE_NAME)
    
    def get_normalized_features_file_path(self, model_name):
        model_dir = self.model_manager.get_model_dir(model_name)
        return os.path.join(model_dir, self.NORM_FEATURES_FILE_NAME)

    def get_aqe_features_file_path(self, model_name):
        model_dir = self.model_manager.get_model_dir(model_name)
        return os.path.join(model_dir, self.AQE_FEATURES_FILE_NAME)

    def get_raw_features_checkpoint(self, model, df, features):
        checkpoint = model.create_checkpoint()
        checkpoint['data'] = df
        checkpoint['raw_features'] = features
        # checkpoint = {
        #     "model_name": # same that in model checkpoint,
        #     "model_state_dict": # same that in model checkpoint,
        #     "optimizer_state_dict": # same that in model checkpoint,
        #     "is_pretrained": # same that in model checkpoint,
        #     "input_resize": # same that in model checkpoint,
        #     "data": # data frame,
        #     "raw_features": # raw features,
        # }
        return checkpoint

    def get_normalized_features_checkpoint(self, model, df, features, pca, PCA_dim):
        checkpoint = model.create_checkpoint()
        checkpoint['data'] = df
        checkpoint['normalized_features'] = features
        checkpoint['pca'] = pca
        checkpoint['PCA_dim'] = PCA_dim
        # checkpoint = {
        #     "model_name": # same that in model checkpoint,
        #     "model_state_dict": # same that in model checkpoint,
        #     "optimizer_state_dict": # same that in model checkpoint,
        #     "is_pretrained": # same that in model checkpoint,
        #     "input_resize": # same that in model checkpoint,
        #     "data": # data frame,
        #     "normalized_features": # normalized features,
        #     "pca": # PCA configuration,
        #     "PCA_dim
        return checkpoint

    def get_aqe_features_checkpoint(self, model, df, features, pca, PCA_dim):
        checkpoint = model.create_checkpoint()
        checkpoint['data'] = df
        checkpoint['pca'] = pca
        checkpoint['PCA_dim'] = PCA_dim
        checkpoint['aqe_features'] = features
        # checkpoint = {
        #     "model_name": # same that in model checkpoint,
        #     "model_state_dict": # same that in model checkpoint,
        #     "optimizer_state_dict": # same that in model checkpoint,
        #     "is_pretrained": # same that in model checkpoint,
        #     "input_resize": # same that in model checkpoint,
        #     "data": # data frame,
        #     "pca": # PCA configuration,
        #     "PCA_dim
        #     "aqe_features": # aqe features,
        # }
        return checkpoint
    def save_raw_features_checkpoint(self, model, df, features):
        features_file_path = self.get_raw_features_file_path(model.get_name())
        raw_features_checkpoint = self.get_raw_features_checkpoint(model, df, features)        
        torch.save(raw_features_checkpoint, features_file_path)

    def save_normalized_features_checkpoint(self, model, df, features, pca, PCA_dim):
        features_file_path = self.get_normalized_features_file_path(model.get_name())
        normalized_features_checkpoint = self.get_normalized_features_checkpoint(model, df, features, pca, PCA_dim)        
        torch.save(normalized_features_checkpoint, features_file_path)

    def save_aqe_features_checkpoint(self, model, df, features, pca, PCA_dim):
        features_file_path = self.get_aqe_features_file_path(model.get_name())
        aqe_features_checkpoint = self.get_aqe_features_checkpoint(model, df, features, pca, PCA_dim)        
        torch.save(aqe_features_checkpoint, features_file_path)

    def is_raw_feature_saved(self, model_name):
        return os.path.isfile(self.get_raw_features_file_path(model_name))
    
    def is_normalized_feature_saved(self, model_name):
        return os.path.isfile(self.get_normalized_features_file_path(model_name))
        
    def is_aqe_feature_saved(self, model_name):
        return os.path.isfile(self.get_aqe_features_file_path(model_name))

    def load_from_raw_features_checkpoint(self, model_name):
        checkpoint = torch.load(self.get_raw_features_file_path(model_name), map_location=self.device)
        # checkpoint = {
        #     "model_name": # same that in model checkpoint,
        #     "model_state_dict": # same that in model checkpoint,
        #     "optimizer_state_dict": # same that in model checkpoint,
        #     "is_pretrained": # same that in model checkpoint,
        #     "input_resize": # same that in model checkpoint,
        #     "data": # data frame,
        #     "aqe_features": # AQE features,
        # }
        return {
            "model": self.model_manager.get_feature_extractor(model_name, checkpoint, load_from=ModelType.FEATURE_EXTRACTOR),
            "data": checkpoint['data'],
            "raw_features": torch.tensor(checkpoint['raw_features'])
        }

    def load_from_norm_features_checkpoint(self, model_name):
        checkpoint = torch.load(self.get_normalized_features_file_path(model_name), map_location=self.device)
        # checkpoint = {
        #     "model_name": # same that in model checkpoint,
        #     "model_state_dict": # same that in model checkpoint,
        #     "optimizer_state_dict": # same that in model checkpoint,
        #     "is_pretrained": # same that in model checkpoint,
        #     "input_resize": # same that in model checkpoint,
        #     "data": # data frame,
        #     "normalized_features": # normalized features,
        #     "pca": # PCA configuration,
        #     "PCA_dim": # PCA dimesion
        # }
        return {
            "model": self.model_manager.get_feature_extractor(model_name, checkpoint, load_from=ModelType.FEATURE_EXTRACTOR),
            "data": checkpoint['data'],
            "normalized_features": torch.tensor(checkpoint['normalized_features']),
            "pca": checkpoint['pca'],
            "PCA_dim": checkpoint['PCA_dim']
        }
    
    def load_from_aqe_features_checkpoint(self, model_name):
        checkpoint = torch.load(self.get_aqe_features_file_path(model_name), map_location=self.device)
        # checkpoint = {
        #     "model_name": # same that in model checkpoint,
        #     "model_state_dict": # same that in model checkpoint,
        #     "optimizer_state_dict": # same that in model checkpoint,
        #     "is_pretrained": # same that in model checkpoint,
        #     "input_resize": # same that in model checkpoint,
        #     "data": # data frame,
        #     "aqe_features": # AQE features,
        # }
        return {
            "model": self.model_manager.get_feature_extractor(model_name, checkpoint, load_from=ModelType.FEATURE_EXTRACTOR),
            "data": checkpoint['data'],
            "aqe_features": checkpoint['aqe_features'],
            "pca": checkpoint['pca'],
            "PCA_dim": checkpoint['PCA_dim']
        }
def prepare_data(dataset_base_dir, labels_file, process_dir, train_size, validate_test_size, clean_process_dir):
    dataset_manager = DatasetManager()      

    train_df, test_df, validate_df = dataset_manager.split_dataset(dataset_name=ModelTrainConfig.DATASET_USEDNAME,
                                                    dataset_base_dir=dataset_base_dir,
                                                    original_labels_file=labels_file,
                                                    process_dir=process_dir,
                                                    clean_process_dir=clean_process_dir,
                                                    train_size=train_size,
                                                    fixed_validate_test_size=validate_test_size
                                                    )
        
    train_df.reset_index(drop=True, inplace=True)
    if DEBUG:print(train_df.head(10))
    if not test_df is None:
        test_df.reset_index(drop=True, inplace=True)
        validate_df.reset_index(drop=True, inplace=True)

    return train_df, test_df, validate_df

def get_pending_features_model(features_manager, model_manager):
    pending_features = []
    models_list = model_manager.get_model_names()
    for model_name in models_list:
        if not features_manager.is_raw_feature_saved(model_name):
            load_from = ModelType.FEATURE_EXTRACTOR

            if model_manager.is_model_saved(model_name, ModelType.FEATURE_EXTRACTOR):
                load_from = ModelType.FEATURE_EXTRACTOR
            elif model_manager.is_model_saved(model_name, ModelType.CLASSIFIER):
                load_from = ModelType.CLASSIFIER
            else:
                raise Exception('Unable to find raw model checkpoint for "' + model_name + '"')
            
            model = model_manager.get_feature_extractor(model_name, load_from=load_from)
            pending_features.append(model)
        else:
            # Test model can be loaded from checkpoint
            loaded_features = features_manager.load_from_norm_features_checkpoint(model_name)
    return pending_features

def extract_features(model, dataloader):
    model_name = model.get_name()
    model.to_device()
    raw_model = model.get_model()
    raw_model.eval()
    
    n_batches = len(dataloader)
    i = 1
    features = []
    with torch.no_grad():
        for image_batch, image_id in dataloader:
            image_batch = image_batch.to(model.get_device())

            batch_features = raw_model(image_batch)

            # features to numpy
            batch_features = torch.squeeze(batch_features).cpu().numpy()

            # collect features
            features.append(batch_features)
            print(f'\rExtract Features {model_name}: Processed {i} of {n_batches} batches', end='', flush=True)

            i += 1

    # stack the features into a N x D matrix            
    features = np.vstack(features)
    return features

def fit_pca(features, PCAdimension):
    #The n_components of PCA must be lower than min(n_samples, n_features)
    return PCA(PCAdimension, whiten=True).fit(features)

def postprocess_features(features, pca):
    #Postprocessing
    # A standard postprocessing pipeline used in retrieval applications is to do L2-normalization,
    # PCA whitening, and L2-normalization again. 
    # Effectively this decorrelates the features and makes them unit vectors.
    features = normalize(features, norm='l2')
    features = pca.transform(features)
    features = normalize(features, norm='l2')

    return features

def Average_Query_Expansion(features, q_indx, top_k=5):
    #new_features = features.copy()
    new_features = features.clone()
    for index in q_indx:
        query = features[index]
        scores = features @ query
        ranking = (-scores).argsort()
        indices = ranking[:top_k+1]
        new_features[index] = features[indices].mean(axis=0)
    return new_features

def extract_models_features():
    # The path of original dataset
    dataset_base_dir = FoldersConfig.DATASET_BASE_DIR
    labels_file = FoldersConfig.DATASET_LABELS_DIR

    # Work directory
    work_dir = FoldersConfig.WORK_DIR

    train_df, test_df, validate_df = prepare_data(dataset_base_dir=dataset_base_dir,
                                                    labels_file=labels_file,
                                                    process_dir=work_dir,
                                                    clean_process_dir=False,
                                                    train_size=ModelTrainConfig.TRAIN_SIZE,
                                                    validate_test_size=ModelTrainConfig.TEST_VALIDATE_SIZE)

    model_manager = ModelManager(device, work_dir)
    features_manager = FeaturesManager(device, model_manager)

    ########### EXTRACT DATASET FEATURES IF NOT EXTRACTED PREVIOUSLY #####################
    pending_features = get_pending_features_model(features_manager, model_manager)

    if len(pending_features) > 0 :

        #create logfile for save statistics - extract features
        fields = ['ModelName', 'DataSetSize','TransformsResize', 'PCASize', 'RawFeatures', 'NormalizedFeatures', 'CalcAQEFeatures', 'ProcessTime']
        logfile = LogFile(fields)

        #Create timer to calculate the process time
        proctimer = ProcessTime()   

        for model in pending_features:
            try:

                proctimer.start()

                model_name = model.get_name()
                if DEBUG:print(f'Extracting features for model {model_name} ....')

                # Save model as feature extractor
                model.save_model()

                # Define input transformations
                transform = model.get_input_transform()
                batch_size = ModelBatchSizeConfig.get_batch_size(model_name)
                train_dataset = FashionProductDataset(dataset_base_dir, train_df, transform=transform)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

                if DEBUG:print(model.get_model())
                model.to_device()
                
                recalc_aqe = False 
                features_size = 0
                PCA_size = 0
                if not features_manager.is_raw_feature_saved(model_name) :
                    recalc_aqe = True  #if generate new normalitzed features, allways generate AQE

                    # Extract features
                    features = extract_features(model, train_loader)
                    features_size = features[0].shape[0]
                    features_manager.save_raw_features_checkpoint(model, train_df, features)

                    # Post process: normalize features
                    PCA_size = FeaturesConfig.get_PCA_size(model_name)
                    pca = fit_pca(features, PCA_size)
                    features = postprocess_features(features, pca)
                    postproc_features_size = features[0].shape[0]
                    features_manager.save_normalized_features_checkpoint(model, train_df, features, pca, PCA_size)

                calcAQE = False
                if not features_manager.is_aqe_feature_saved(model_name) or recalc_aqe:
                    calcAQE = True
                    loaded_model_features = features_manager.load_from_norm_features_checkpoint(model.get_name())
                    features = loaded_model_features['normalized_features']
                    postproc_features_size = features[0].shape[0]

                    # average of the top-K ranked results (including the query itself)
                    num_queries = features.shape[0] #all dataset
                    q_indx = range(0,num_queries - 1)
                    features = Average_Query_Expansion(features,q_indx,ModelTrainConfig.TOP_K_AQE)
                    features_manager.save_aqe_features_checkpoint(model, train_df, features, pca, PCA_size)

                #LOG
                processtime = proctimer.stop()
                values = {  'ModelName': model_name, 
                            'DataSetSize': train_df.shape[0], 
                            'TransformsResize': model.get_input_resize(),
                            'PCASize': PCA_size,
                            'RawFeatures': features_size,
                            'NormalizedFeatures': postproc_features_size,
                            'CalcAQEFeatures': calcAQE,
                            'ProcessTime': processtime
                        }
                logfile.writeLogFile(values)
            except Exception as e:
                print('\n', e)
                #LOG
                processtime = proctimer.stop()
                values = {  'ModelName': model_name, 
                            'DataSetSize': train_df.shape[0], 
                            'TransformsResize': model.get_input_resize(),
                            'PCASize': PCA_size,
                            'RawFeatures': features_size,
                            'NormalizedFeatures': 'ERROR',
                            'CalcAQEFeatures': 'ERROR',
                            'ProcessTime': processtime
                        }
                logfile.writeLogFile(values)

        #Print and save logfile    
        logfile.printLogFile()
        logfile.saveLogFile_to_csv("feature_extraction")

if __name__ == "__main__":
    if(device.type == 'cuda' and torch.cuda.is_available()):
        torch.cuda.empty_cache()

    extract_models_features()