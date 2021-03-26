import os
import torch
import numpy as np

from models import ModelManager
from config import DebugConfig, FoldersConfig, DeviceConfig
from features import FeaturesManager
from features import postprocess_features
from dataset import FashionProductDataset

device = DeviceConfig.DEVICE
DEBUG = DebugConfig.DEBUG

class RetrievalEngine:
    def __init__(self, device, work_dir):
        super().__init__()
        self.device = device
        # Work directory
        self.work_dir = work_dir
        self.model_manager = ModelManager(device, work_dir)
        self.features_manager = FeaturesManager(device, self.model_manager)
        self.model_features = {}

    # Returns the available models to use
    def get_model_names(self):
        return self.model_manager.get_model_names()

    # Load the precomputed features from dataset
    def load_models_and_precomputed_features(self):
        for model_name in self.get_model_names():
            try:
                loaded_model_features = self.features_manager.load_from_norm_features_checkpoint(model_name)
                self.model_features[model_name] = loaded_model_features
            except Exception as e:
                print('Failed to load model: '+ str(e))
    
    def query(self, model_name, query_image_path, top_k):
        print('Query for model ' + model_name)
        model_features = self.model_features[model_name]
        # Preprocess query image
        # Tensor [1, 3, input_resize, input_resize]
        query = FashionProductDataset.preprocess_image(query_image_path, model_features['model'].get_input_transform())
        query = torch.unsqueeze(query, 0)

        # Compute features
        query_features = self.compute_features(model_name, query)
        # Perform similarity
        scores = self.cosine_similarity(model_name, query_features)
        # Return top K ids
        # rank by score, descending, and skip the top match (because it will be the query)
        ranking = (-scores).argsort()[1:top_k + 1]
        return self.convert_ranking_to_image_ids(model_name, ranking)
        
    def convert_ranking_to_image_ids(self, model_name, ranking):
        model_features = self.model_features[model_name]
        ranking = ranking.numpy()
        data = model_features['data']
        id_ranking = [None] * len(ranking)
        for i in range(0, len(ranking)):    
            id_ranking[i] = data.iloc[ranking[i]]['id']
        
        return id_ranking

    def compute_features(self, model_name, query):
        model_features = self.model_features[model_name]
        model = model_features['model'].get_model()
        model.cpu()
        model.eval()
        with torch.no_grad():
            raw_features = model(query)
            numpy_features = raw_features.cpu().numpy()
            
        features = postprocess_features(numpy_features, model_features['pca'])
        return torch.tensor(features).squeeze()

    def cosine_similarity(self, model_name, query_features):
        features = self.model_features[model_name]['normalized_features']
        return features @ query_features.T

if __name__ == "__main__":

    top_k = 5
    model_name = 'vgg16'
    engine = RetrievalEngine(device, FoldersConfig.WORK_DIR)
    engine.load_models_and_precomputed_features()

    imageid = 26267
    base_dir = FoldersConfig.DATASET_BASE_DIR
    images_path = os.path.join(base_dir, FashionProductDataset.IMAGE_DIR_NAME)
    query_path = os.path.join(images_path, f"{imageid}{FashionProductDataset.IMAGE_FORMAT}")

    top_k_ranking = engine.query(model_name, query_path, top_k)
    print(top_k_ranking)