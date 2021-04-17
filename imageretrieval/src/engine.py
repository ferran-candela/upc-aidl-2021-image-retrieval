import os
import torch
import numpy as np

from imageretrieval.src.models import ModelManager
from imageretrieval.src.config import DebugConfig, FoldersConfig, DeviceConfig
from imageretrieval.src.features import FeaturesManager
from imageretrieval.src.features import postprocess_features
from imageretrieval.src.dataset import FashionProductDataset, DeepFashionDataset

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
            print('Loading model: '+ model_name)
            try:
                loaded_model_features = self.features_manager.load_from_norm_features_checkpoint(model_name)
                self.model_features[model_name] = loaded_model_features
            except Exception as e:
                print('\nFailed to load model: '+ str(e))
            print('Models loaded.')
    

    def get_query_features(self, model_name, query_image_path):
        model_features = self.model_features[model_name]
        # Preprocess query image
        # Tensor [1, 3, input_resize, input_resize]
        query = FashionProductDataset.preprocess_image(query_image_path, model_features['model'].get_input_transform())
        query = torch.unsqueeze(query, 0)

        # Compute features
        return self.compute_features(model_name, query)


    def query(self, model_name, query_image_path, top_k):
        print('Query for model ' + model_name)
        model_features = self.model_features[model_name]
        # Preprocess query image
        # Tensor [1, 3, input_resize, input_resize]
        print('Preprocessing...')
        query = FashionProductDataset.preprocess_image(query_image_path, model_features['model'].get_input_transform())
        query = torch.unsqueeze(query, 0)

        # Compute features
        print('Computing features...')
        query_features = self.compute_features(model_name, query)
        # Perform similarity
        print('Computing similarity...')
        scores = self.cosine_similarity(model_name, query_features)
        # Return top K ids
        ranking = (-scores).argsort()[:top_k]
        print('Convert result to image ids...')
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
    

    def get_image_path(self, img_id):
        base_dir = FoldersConfig.DATASET_BASE_DIR
        images_path = os.path.join(base_dir, FashionProductDataset.IMAGE_DIR_NAME)
        return os.path.join(images_path, f"{img_id}{FashionProductDataset.IMAGE_FORMAT}")


    def get_image_deep_fashion_path(self, img_id):
        base_dir = FoldersConfig.DATASET_BASE_DIR
        images_path = os.path.join(base_dir, DeepFashionDataset.IMAGE_DIR_NAME)
        return os.path.join(images_path, f"{img_id}")
        

    def print_query_results(self, query_image_id, ranking, description):
        import matplotlib.pyplot as plt
        import matplotlib.cbook as cbook

        # show the query image
        print(f'\rImage Query id: ', str(query_image_id),' - ',description)
        plt.figure(figsize=(2.8,2.8))
        with cbook.get_sample_data(self.get_image_path(query_image_id)) as image_file:
            query_image = plt.imread(image_file)
        plt.imshow(query_image)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.show()

        print(f'\rTop N Similarity: ', description)
        fig, ax = plt.subplots(nrows=int(len(ranking)/5), ncols=5, figsize=(18, 6))
        ax = ax.ravel()

        if DEBUG: print(ranking)
        for i in range(len(ranking)):
            # show the images
            with cbook.get_sample_data(self.get_image_path(ranking[i])) as image_file:
                query_image = plt.imread(image_file)
            ax[i].imshow(query_image)
            ax[i].grid(False) 
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(str(i) + " - id:" + str(ranking[i]) )
        plt.show()


if __name__ == "__main__":

    top_k = 15
    engine = RetrievalEngine(device, FoldersConfig.WORK_DIR)
    engine.load_models_and_precomputed_features()

    # img_id = 26267
    img_id = 8080
    query_path = engine.get_image_path(img_id)
    # query_path = '/tmp/43573.jpg'

    model_names = engine.get_model_names()

    for model_name in model_names:
        top_k_ranking = engine.query(model_name, query_path, top_k)
        print(model_name, ': ', top_k_ranking)
        engine.print_query_results(query_image_id=img_id, ranking=top_k_ranking, description=model_name + " - COSINE SIMILARITY")