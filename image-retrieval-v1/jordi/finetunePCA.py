from sklearn.decomposition import PCA
from evaluation import make_ground_truth_matrix, create_ground_truth_queries, evaluate    
import matplotlib.pyplot as plt
from utils import ProcessTime

def PCA_VarianceDimension(model_name,features,PCAdimension):
    #https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    #The individual variance will tell us how important the newly added features are
    pca = PCA(PCAdimension)
    pca.fit(features)
    plt.style.use('seaborn')
    plt.plot(range(1,PCAdimension + 1),pca.explained_variance_ratio_,'o--', markersize=4)
    plt.title (model_name + ': Variance for each PCA dimension - Num.Features: ' + str(pca.n_features_))
    plt.xlabel('PCA Dimensions')
    plt.ylabel('Variance')
    plt.grid(True)
    plt.show()

    #visualize how much of the original data is explained by the limited number of features by finding the cumulative variance
    plt.plot(range(1,PCAdimension + 1),pca.explained_variance_ratio_.cumsum(),'o--', markersize=4)
    plt.title (model_name + ': Cumulative Variance with each PCA dimension - Num.Features: ' + str(pca.n_features_))
    plt.xlabel('PCA Dimensions')
    plt.ylabel('Variance')
    plt.grid(True)
    plt.show()    


def PCA_Tune(PretainedModelClass,model_name,features,dataframe,num_queries):
    pca_dimensions = [10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,32,34,36,38,40,42,44]

    pca_accuracy = []
    pca_time = []

    for dimensions in pca_dimensions:
        # Perform PCA
        feature_list_compressed = PretainedModelClass.postprocessing_features(features,dimensions)
        # Calculate accuracy over the compressed features
        accuracy, time_taken = accuracy_mAP_calc(feature_list_compressed[:],dataframe,num_queries)
        # 
        pca_time.append(time_taken)
        pca_accuracy.append(accuracy)
        print("For PCA Dimensions = ", dimensions, ",\tAccuracy = ",accuracy,"%",
            ",\tTime = ", pca_time[-1])

    plt.plot(pca_time, pca_accuracy,'o--', markersize=4)
    for label, x, y in zip(pca_dimensions, pca_time,pca_accuracy):
        plt.annotate(label, xy=(x, y), ha='right', va='bottom')
    plt.title (model_name + ': Test Time vs Accuracy for each PCA dimension')    
    plt.xlabel('Test Time')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()            


def accuracy_mAP_calc(features,dataframe,num_queries):
    #mAP accuracy

    proctimer = ProcessTime()
    proctimer.start()

    #compute the similarity matrix
    S = features @ features.T

    queries = create_ground_truth_queries( dataframe, "Fixed", num_queries, [])
    q_indx, y_true = make_ground_truth_matrix(dataframe, queries)

    #Compute mean Average Precision (mAP)
    df = evaluate(S, y_true, q_indx)
    processtime = proctimer.stop()    

    return round(df.ap.mean(),4),processtime


