import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from imageretrieval.src.evaluation import make_ground_truth_matrix, create_ground_truth_queries, evaluate, cosine_similarity, evaluation_hits    
from imageretrieval.src.models import ModelManager
from imageretrieval.src.features import FeaturesManager,postprocess_features,fit_pca
from imageretrieval.src.dataset import DatasetManager
from imageretrieval.src.config import DebugConfig, FoldersConfig, DeviceConfig, RetrievalEvalConfig, ModelTrainConfig
from imageretrieval.src.utils import ProcessTime, LogFile

device = DeviceConfig.DEVICE
DEBUG = DebugConfig.DEBUG

def PCA_VarianceDimension_Plot(model_name,features,PCAdimension,save_path):
    #https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    #The individual variance will tell us how important the newly added features are
    print('\nPlot individual variance - PCA: ', str(PCAdimension))
    pca = fit_pca(features, PCAdimension)
    fig = plt.figure(figsize=(10,8))
    plt.style.use('seaborn')
    plt.plot(range(1,PCAdimension + 1),pca.explained_variance_ratio_,'o--', markersize=4)
    plt.title (model_name + ': Variance for each PCA dimension - Num.Features: ' + str(pca.n_features_))
    plt.xlabel('PCA Dimensions')
    plt.ylabel('Variance')
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()
    #plt.show()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig.savefig(os.path.join(save_path, 'indiv_variance_plot_' + str(PCAdimension) + '.png'), bbox_inches='tight')
    plt.close(fig)

    print('\nPlot cumulative variance - PCA: ', str(PCAdimension))
    #visualize how much of the original data is explained by the limited number of features by finding the cumulative variance
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,PCAdimension + 1),pca.explained_variance_ratio_.cumsum(),'o--', markersize=4)
    plt.title (model_name + ': Cumulative Variance with each PCA dimension - Num.Features: ' + str(pca.n_features_))
    plt.xlabel('PCA Dimensions')
    plt.ylabel('Variance')
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()
    #plt.show()  

    fig.savefig(os.path.join(save_path, 'cumul_variance_plot_' + str(PCAdimension) + '.png'), bbox_inches='tight')

    plt.close(fig)


def PCA_Tune(model_name,features,dataframe,querylist,pca_dimensions,save_path,accuracy_type="pHits"):

    patiente = 2

    pca_accuracy = []
    pca_time = []
    no_improve = 0
    best_accuracy = 0
    best_pca = 0
    for dimension in pca_dimensions:
        # Perform postprocess        
        print('\nFitting pca.....: ', str(dimension))        
        pca = fit_pca(features, dimension)
        print('\nPostprocessing features.... pca: ', str(dimension))        
        features_pp = postprocess_features(features,pca)
        # Calculate accuracy over the postprocesed features

        if accuracy_type=='mAP':
            print('\nCalculating accuracy (mAP) for query.... - PCA: ', str(dimension))        
            accuracy, time_taken = accuracy_mAP_calc(features_pp[:],dataframe,querylist)
        else:
            print('\nCalculating accuracy (precision Hits) for query.... - PCA: ', str(dimension))        
            accuracy, time_taken = accuracy_precisionHits_calc(features_pp[:],dataframe,querylist)

        # 
        pca_time.append(time_taken)
        pca_accuracy.append(accuracy)
        print("For PCA Dimension = ", dimension, ",\tAccuracy = ",accuracy,"%",
            ",\tTime = ", pca_time[-1])

        is_better = accuracy > best_accuracy
        if is_better:
            no_improve = 0
            best_accuracy = accuracy
            best_pca = dimension
        else:
            no_improve += 1
        if no_improve == patiente: #patience: Number iterations to wait if no improvement
            break
        
    fig = plt.figure(figsize=(10,8))
    plt.plot(pca_time, pca_accuracy,'o--', markersize=4)
    for label, x, y in zip(pca_dimensions, pca_time,pca_accuracy):
        plt.annotate(label, xy=(x, y), ha='right', va='bottom')
    plt.title (model_name + ': Test Time vs Accuracy for each PCA dimension')    
    plt.xlabel('Test Time')
    plt.ylabel('Accuracy')
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()
    #plt.show()            
    fig.savefig(os.path.join(save_path, 'pca_tune_' + str(best_pca) + '.png'), bbox_inches='tight')
    plt.close(fig)

    return best_pca, best_accuracy

def accuracy_mAP_calc(features,dataframe,querylist):
    #mAP accuracy

    proctimer = ProcessTime()
    proctimer.start()

    #compute the similarity matrix
    S = features @ features.T

    queries = create_ground_truth_queries( dataframe, dataframe, "List",0, querylist)
    q_indx, y_true = make_ground_truth_matrix(dataframe, queries)

    #Compute mean Average Precision (mAP)
    df = evaluate(S, y_true, q_indx)
    processtime = proctimer.stop()    

    return round(df.ap.mean(),4),processtime

def accuracy_precisionHits_calc(features,dataframe,querylist):
    #mAP accuracy

    proctimer = ProcessTime()
    proctimer.start()

    #compute the similarity matrix
    S = features @ features.T

    queries = create_ground_truth_queries( dataframe, dataframe, "List",0, querylist)
    q_indx, y_true = make_ground_truth_matrix(dataframe, queries)

    accuracy = []
    for index in q_indx:
        ranking = cosine_similarity(features, index, RetrievalEvalConfig.TOP_K_IMAGE)
        precision = evaluation_hits(dataframe, dataframe, index, ranking)
        accuracy.append(precision)
    precision = np.mean(accuracy)
    processtime = proctimer.stop()  

    return round(precision,4),processtime

                
def generate_pca_interval(dimension_ini, dimension_end, interval):
    step = int( (dimension_end - dimension_ini) / interval )
    if (step == 0): step = 1
    pca_dimensions = []
    pca = dimension_ini
    for i in range(1, interval + 1 ): 
        pca = pca + step
        if ( i==interval or pca >= dimension_end):
            pca = dimension_end
            pca_dimensions.append(pca)
            break
        pca_dimensions.append(pca)
    return pca_dimensions, step

def calculate_best_pca(model_name,features,data,querylist,tunefilespath):
    features_dimension = features[0].shape[0]
    print('\nOriginal dimension features: ', str(features_dimension))

    interval = 10
    dimension_ini = 0
    dimension_end = features_dimension
    
    best_pca = 0
    best_accuracy = 0
    while True:
        #generate pca interval. Every time more closed
        pca_dimensions, step = generate_pca_interval(dimension_ini=dimension_ini, dimension_end=dimension_end, interval=interval)

        #Save plots Variance
        for dimension in pca_dimensions:                
            PCA_VarianceDimension_Plot(model_name,features,dimension,tunefilespath)

        #calculate best pca from pca interval
        pca,accuracy = PCA_Tune(model_name=model_name,features=features,dataframe=data,querylist=querylist,pca_dimensions=pca_dimensions, save_path=tunefilespath,accuracy_type=RetrievalEvalConfig.PCA_ACCURACY_TYPE)

        if (best_pca == pca) or (pca == dimension_end):
            return pca,accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_pca = pca
        else:
            return best_pca,accuracy

        #new interval more closed
        dimension_ini = best_pca - step
        dimension_end = best_pca + step
        if (dimension_ini < 0): dimension_ini = 0
        if (dimension_end > features_dimension): dimension_end = features_dimension

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

            data = loaded_model_features['data']

            #for evaluate, calculate always the same queries
            #Create a random list
            print('\nCreating query...')
            num_queries = RetrievalEvalConfig.MAP_N_QUERIES
            qrylist = data.sample(num_queries).index
            
            model_path = model_manager.get_model_dir(model_name=model_name)
            tunepath = os.path.join(model_path, 'PCAtune')

            best_pca,accuracy = calculate_best_pca(model_name=model_name,features=features,data=data,querylist=qrylist,tunefilespath=tunepath)
            print('\n', model_name , ' Best PCA: ', str(best_pca), ' Accuracy: ', str(accuracy))
            #save result into file
            filepath = os.path.join(tunepath, 'bestpca.txt')
            f = open(filepath, "w")
            f.write("PCA=" + str(best_pca) + " ACCURACY " + RetrievalEvalConfig.PCA_ACCURACY_TYPE + " : " + str(accuracy))
            f.close()

if __name__ == "__main__":
    finetune_pca()