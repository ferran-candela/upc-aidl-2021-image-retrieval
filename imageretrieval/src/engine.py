import os
from .models import ModelManager

class RetrievalEngine:
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.modelmanager = ModelManager(self.device)

    # Returns the available models to use
    def get_model_names(self):
        return self.modelmanager.get_model_names()

    # Load the precomputed features from dataset
    def load_models_and_precomputed_features(self):
        for model_name in pretained_list_models:
            model_dir = os.path.join(config["work_dir"], model_name)
            features_file = os.path.join(model_dir , 'features.pickle')
            features = pickle.load(open(features_file , 'rb'))
    
    