import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

class PretainedModels:
    def __init__(self,device):
        super().__init__()
        self.models =   {  
                            'vgg16', # Documentation says input must be 224x224
                            # 'resnet50',
                            # 'inception_v3', [batch_size, 3, 299, 299]
                            # 'inception_resnet_v2' #needs : [batch_size, 3, 299, 299]
                            #'densenet161',
                            #'efficient_net_b4'
                        }
                    
        self.device = device

    def get_pretained_models_names(self):
        return self.models

    def load_pretrained_model(self,model_name):
        if model_name == 'vgg16':
            from torchvision.models import vgg16
            # Input must be 224x224
            pretrained_model = vgg16(pretrained=True)
            # Just use the output of feature extractor and ignore the classifier
            pretrained_model.classifier = nn.Identity()

        if model_name == 'resnet50':
            from torchvision.models import resnet50
            pretrained_model = resnet50(pretrained=True)
            # Remove FC layer
            pretrained_model.fc = nn.Identity()
            # Remove RELU
            pretrained_model.layer4[2].relu = nn.Identity()
            
        if model_name == 'inception_v3':
            from torchvision.models import inception_v3
            pretrained_model = inception_v3(pretrained=True)
            # Remove FC Layer
            pretrained_model.fc = nn.Identity()

        if model_name == 'inception_resnet_v2':
            from torch_inception_resnet_v2.model import InceptionResNetV2
            pretrained_model = InceptionResNetV2(1000) #upper to PCA
            # Remove FC Layer
            pretrained_model.dropout = nn.Identity()
            pretrained_model.fc = nn.Identity()
            pretrained_model.softmax = nn.Identity()
        
        if model_name == 'densenet161':
            from torchvision.models import densenet161
            pretrained_model = densenet161(pretrained=True)
            #Just use the output of feature extractor and ignore the classifier
            pretrained_model.classifier = nn.Identity()

        if model_name == 'efficient_net_b4':

            from efficientnet_pytorch import EfficientNet
            pretrained_model = EfficientNet.from_pretrained('efficientnet-b4')

            pretrained_model._dropout = nn.Identity()
            pretrained_model._fc = nn.Identity()
            pretrained_model._swish = nn.Identity() #Swish activation function 
            #pretrained_model.set_swish(memory_efficient=False)
        
        return pretrained_model
    
    def tuning_batch_norm_statistics(self,model,loader):
        # Here we're going to apply a rudimentaty domain adaptation trick. The batch norm statistics for this network match those of the ImageNet dataset. 
        # We can use a trick to get them to match our dataset. The idea is to put the network into train mode and do a pass over the dataset without doing any backpropagation. 
        # This will cause the network to update the batch norm statistics for the model without modifying the weights. This can sometimes improve results.        
        model.train()
        n_batches = len(loader)
        i = 1
        for image_batch, image_id in loader:
            # move batch to device and forward pass through network
            model(image_batch.to(self.device))
            print(f'\rTuning batch norm statistics {i}/{n_batches}', end='', flush=True)
            i += 1
        return model

    def extract_features(self,model,dataloader,transform):
        model.eval()
        #model.to(self.device)
        n_batches = len(dataloader)
        i = 1
        features = []
        with torch.no_grad():
            for image_batch, image_id in dataloader:
                image_batch = image_batch.to(self.device)

                batch_features = model(image_batch)

                # features to numpy
                batch_features = torch.squeeze(batch_features).cpu().numpy()

                # collect features
                features.append(batch_features)
                print(f'\rExtract Features: Processed {i} of {n_batches} batches', end='', flush=True)

                i += 1

        # stack the features into a N x D matrix            
        features = np.vstack(features)
        return features 

    def postprocessing_features(self,features,PCAdimension):
        #Postprocessing
        # A standard postprocessing pipeline used in retrieval applications is to do L2-normalization,
        # PCA whitening, and L2-normalization again. 
        # Effectively this decorrelates the features and makes them unit vectors.
        features = normalize(features, norm='l2')
        features = PCA(PCAdimension, whiten=True).fit_transform(features) #The n_components of PCA must be lower than min(n_samples, n_features)
        features= normalize(features, norm='l2')

        return features

    def Cosine_Similarity(self,features,imgidx,top_k):
        # This gives the same rankings as (negative) Euclidean distance 
        # when the features are L2 normalized (as ours are).
        # The cosine similarity can be efficiently computed for all images 
        # in the dataset using a matrix multiplication!
        query = features[imgidx]
        scores = features @ query 

        # rank by score, descending, and skip the top match (because it will be the query)
        ranking = (-scores).argsort()[1:top_k + 1]
        return ranking

    def Euclidean_Distance(self,features,imgidx,top_k):
        neighbors = NearestNeighbors(n_neighbors=top_k + 1, algorithm='brute',
                                    metric='euclidean').fit(features)
        distances, indices = neighbors.kneighbors([features[imgidx]])
        #the nearest index will be the image itself. Remove the first
        return distances[0][1:], indices[0][1:]


    def Average_Query_Expansion(self,features, q_indx, top_k=5):
        new_features = features.copy()
        for index in q_indx:
            query = features[index]
            scores = features @ query
            ranking = (-scores).argsort()
            indices = ranking[:top_k+1]
            new_features[index] = features[indices].mean(axis=0)
        return new_features

    def Count_Parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)        
