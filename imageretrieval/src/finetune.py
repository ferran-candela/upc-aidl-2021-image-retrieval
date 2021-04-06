import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from evaluation import make_ground_truth_matrix, create_ground_truth_queries, evaluate    
from models import ModelManager
from features import FeaturesManager,postprocess_features,fit_pca
from dataset import DatasetManager
from config import DebugConfig, FoldersConfig, DeviceConfig, RetrievalEvalConfig, ModelTrainConfig
from utils import ProcessTime, LogFile

device = DeviceConfig.DEVICE
DEBUG = DebugConfig.DEBUG

def PCA_VarianceDimension_Plot(model_name,features,PCAdimension,save_path):
    #https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    #The individual variance will tell us how important the newly added features are
    pca = fit_pca(features, PCAdimension)
    #PCA(PCAdimension,whiten=True)
    #pca.fit(features)

    fig = plt.figure(figsize=(10,8))
    plt.style.use('seaborn')
    plt.plot(range(1,PCAdimension + 1),pca.explained_variance_ratio_,'o--', markersize=4)
    plt.title (model_name + ': Variance for each PCA dimension - Num.Features: ' + str(pca.n_features_))
    plt.xlabel('PCA Dimensions')
    plt.ylabel('Variance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig.savefig(os.path.join(save_path, 'indiv_variance_plot_' + str(PCAdimension) + '.png'), bbox_inches='tight')

    #visualize how much of the original data is explained by the limited number of features by finding the cumulative variance
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,PCAdimension + 1),pca.explained_variance_ratio_.cumsum(),'o--', markersize=4)
    plt.title (model_name + ': Cumulative Variance with each PCA dimension - Num.Features: ' + str(pca.n_features_))
    plt.xlabel('PCA Dimensions')
    plt.ylabel('Variance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()  

    fig.savefig(os.path.join(save_path, 'cumul_variance_plot_' + str(PCAdimension) + '.png'), bbox_inches='tight')


def PCA_Tune(model_name,features,dataframe,num_queries,pca_dimensions,save_path):

    pca_accuracy = []
    pca_time = []

    for dimension in pca_dimensions:
        # Perform postprocess        
        pca = fit_pca(features, dimension)
        features_pp = postprocess_features(features,pca)
        # Calculate accuracy over the postprocesed features
        accuracy, time_taken = accuracy_mAP_calc(features_pp[:],dataframe,num_queries)
        # 
        pca_time.append(time_taken)
        pca_accuracy.append(accuracy)
        print("For PCA Dimensions = ", dimension, ",\tAccuracy = ",accuracy,"%",
            ",\tTime = ", pca_time[-1])

    fig = plt.figure(figsize=(10,8))
    plt.plot(pca_time, pca_accuracy,'o--', markersize=4)
    for label, x, y in zip(pca_dimensions, pca_time,pca_accuracy):
        plt.annotate(label, xy=(x, y), ha='right', va='bottom')
    plt.title (model_name + ': Test Time vs Accuracy for each PCA dimension')    
    plt.xlabel('Test Time')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()            
    fig.savefig(os.path.join(save_path, 'pca_tune_' + str(PCAdimension) + '.png'), bbox_inches='tight')


def accuracy_mAP_calc(features,dataframe,num_queries):
    #mAP accuracy

    proctimer = ProcessTime()
    proctimer.start()

    #compute the similarity matrix
    S = features @ features.T

    queries = create_ground_truth_queries( dataframe, "FirstN", num_queries, [])
    q_indx, y_true = make_ground_truth_matrix(dataframe, queries)

    #Compute mean Average Precision (mAP)
    df = evaluate(S, y_true, q_indx)
    processtime = proctimer.stop()    

    return round(df.ap.mean(),4),processtime

def finetune_pca():
    model_manager = ModelManager(device, FoldersConfig.WORK_DIR)
    features_manager = FeaturesManager(device, model_manager)

    model_names = model_manager.get_model_names()

    for model_name in model_names:
        if(features_manager.is_raw_feature_saved(model_name)):
            print('\n\n## FineTune PCA model ', model_name)
            # LOAD FEATURES
            print('\nLoading features from checkpoint...')
            loaded_model_features = features_manager.load_from_raw_features_checkpoint(model_name)
            features = loaded_model_features['raw_features']

            print('\noriginal dimension features: ', str(features[0].shape[0]))

            model_path = model_manager.get_model_dir(model_name=model_name)
            PCApath = os.path.join(model_path, 'PCAtune')
            pca_dimensions = [80,100,120,140,160,180,200]
            for dimension in pca_dimensions:                
                PCA_VarianceDimension_Plot(model_name,features,dimension,PCApath)
            
            data = loaded_model_features['data']

            pca_dimensions = [15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]
            num_queries = RetrievalEvalConfig.MAP_N_QUERIES
            
            for dimension in pca_dimensions:
                PCA_Tune(model_name=model_name,features=features,dataframe=data,num_queries=num_queries,pca_dimensions=pca_dimensions,save_path=PCApath)
                


if __name__ == "__main__":
    finetune_pca()