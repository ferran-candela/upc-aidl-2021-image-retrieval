import os
import torch
import numpy as np

from dataset import MyDataset
from data_preparation import Prepare_Data
from pretained_models import PretainedModels
from utils import ProcessTime, LogFile, ImageSize
from evaluation import make_ground_truth_matrix, create_ground_truth_queries, evaluate, evaluation_hits    
from sklearn.neighbors import NearestNeighbors

from torchvision import transforms
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import time

#if not torch.cuda.is_available():
#    raise Exception("You should enable GPU")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def Print_Similarity(dataset_img, imgidx, ranking, description,debug=False):
    # show the query image
    print(f'\rImage Query id: ', str(dataset_img[imgidx][1]),' - ',description)
    plt.figure(figsize=(2.8,2.8))
    plt.imshow(dataset_img[imgidx][0])
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    print(f'\rTop 10 Similarity: ', description)
    fig, ax = plt.subplots(nrows=int(ranking.size/5), ncols=5, figsize=(18, 6))
    ax = ax.ravel()

    if debug: print(ranking)
    for i in range(ranking.size):
        img = dataset_img[ranking[i]][0]
        # show the images
        ax[i].imshow(img)
        ax[i].grid(False) 
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(str(i) + " - id:" + str(dataset_img[ranking[i]][1]) )
    plt.show()


def main(config):
    # The path of original dataset
    dataset_base_dir = config["dataset_base_dir"]
    dataset_image_dir = os.path.join(dataset_base_dir, 'images')
    labels_file = config["dataset_labels_file"]
    if config["debug"]=='True':
        debug = True
    else:
        debug = False

    # Work directory
    work_dir = config["work_dir"]

    #Log directory
    if not os.path.exists(config["log_dir"]):
        os.mkdir(config["log_dir"])

    if config["train_size"] == 'all':
        train_size = -1
    elif config["train_size"] == 'divide':
        train_size = 0
    else:
        train_size = config["train_size"]
    validate_test_size = config["test_validate_size"]                

    train_df, test_df, validate_df = Prepare_Data(img_dir=dataset_image_dir,
                                                    original_labels_file=labels_file,
                                                    process_dir=work_dir,
                                                    clean_process_dir=False,
                                                    fixed_train_size=train_size,
                                                    fixed_validate_test_size=validate_test_size,
                                                    debug=debug
                                                    )
        
    train_df.reset_index(drop=True, inplace=True)
    if debug:print(train_df.head(10))
    if not test_df is None:
        test_df.reset_index(drop=True, inplace=True)
        validate_df.reset_index(drop=True, inplace=True)

    pretained_models = PretainedModels(device)    
    pretained_list_models = pretained_models.get_pretained_models_names()

    #Create folder for each model
    for model_name in pretained_list_models:
        model_dir = os.path.join(work_dir, model_name)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)


    ########### EXTRACT FEATURES IF NOT EXTRACTED PREVIUS #####################
    #extract image features for any model only if not previously extracted
    pending_models_extract = []
    for model_name in pretained_list_models:
        model_dir = os.path.join(work_dir, model_name)
        features_file = os.path.join(model_dir, 'features.pickle')
        if not os.path.isfile(features_file):
            pending_models_extract.append(model_name)

    if len(pending_models_extract) > 0 :
        transform = transforms.Compose([
            transforms.Resize((config["transforms_resize"])),
            transforms.CenterCrop(config["transforms_resize"]-32),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        train_dataset = MyDataset(dataset_image_dir,train_df,transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)

        #create logfile for save statistics - extract features
        fields = ['ModelName', 'DataSetSize','TransformsResize','ParametersCount', 'FeaturesSize', 'PostProcFeaturesSize', 'ProcessTime']
        logfile = LogFile(fields)        

        #Create timer to calculate the process time
        proctimer = ProcessTime()

        for model_name in pending_models_extract:
            #PATH to save features
            model_dir = os.path.join(work_dir, model_name)

            # Load pretrained model
            pretained_model = pretained_models.load_pretrained_model(model_name)
            if debug:print(pretained_model)
            pretained_model.to(device)

            proctimer.start()

            #put the network into train mode and do a pass over the dataset without doing any backpropagation
            pretained_model = pretained_models.tuning_batch_norm_statistics(pretained_model,train_loader) 
            #extract features
            features = pretained_models.extract_features(pretained_model,train_loader,transform) 
            features_size = features[0].shape[0]
            #save complete features
            features_file = os.path.join(model_dir, 'features_full.pickle')
            pickle.dump(features , open(features_file, 'wb'))

            #normalize features            
            features = pretained_models.postprocessing_features(features,config["PCAdimension"]) 
            postproc_features_size = features[0].shape[0]

            #save features
            features_file = os.path.join(model_dir, 'features.pickle')
            pickle.dump(features , open(features_file, 'wb'))

            #LOG
            processtime = proctimer.stop()
            values = {  'ModelName':model_name, 
                        'DataSetSize':train_df.shape[0], 
                        'TransformsResize':config["transforms_resize"],
                        'ParametersCount':pretained_models.Count_Parameters(pretained_model), 
                        'FeaturesSize': features_size,
                        'PostProcFeaturesSize': postproc_features_size,
                        'ProcessTime':processtime
                    } 
            logfile.writeLogFile(values)

        #Print and save logfile    
        logfile.printLogFile()
        logfile.saveLogFile_to_csv("extractfeatures",config)


    ##### IMAGE RETRIEVAL TEST #############

    #create logfile for image retrieval
    fields = ['ModelName', 'DataSetSize','FeaturesSize', 'ProcessTime', 'mAPqueries', 'mAP', 'Precision']
    logfile = LogFile(fields)        
    #Create timer to calculate the process time
    proctimer = ProcessTime()

    img_ds = MyDataset(dataset_image_dir,train_df)

    Test_size = 5  #number images to test
    testid = np.random.randint(train_df.shape[0], size=Test_size)
    it = 0
    for index in testid:
        # change this to the index of the image to use as a query!
        imgidx = index
        for model_name in pretained_list_models:
            print(f'\rTest image ... {it}/{Test_size }', end='', flush=True)
            proctimer.start()

            model_dir = os.path.join(config["work_dir"], model_name)
            features_file = os.path.join(model_dir , 'features.pickle')
            features = pickle.load(open(features_file , 'rb'))

            # COSINE SIMILARITY
            ranking = pretained_models.Cosine_Similarity(features,imgidx,config["top_n_image"])
            Print_Similarity(img_ds,imgidx,ranking,model_name + " - COSINE SIMILARITY",debug)

            #LOG
            processtime = proctimer.stop()

            if  config["evaluate"] == "True":
                #Run evaluation 

                #compute the similarity matrix
                S = features @ features.T

                num_queries = config["mAP_n_queries"]

                queries = create_ground_truth_queries( train_df, "Random", num_queries,[])
                q_indx, y_true = make_ground_truth_matrix(train_df, queries)

                #Compute mean Average Precision (mAP)
                df = evaluate(S, y_true, q_indx)
                if debug:print(df)
                print(f'\rmAP: {df.ap.mean():0.04f}')

                #Compute evaluation Hits
                precision = evaluation_hits(train_df,imgidx,ranking)
                print(f'\rPrecision hits: {precision:0.04f}')

                values = {  'ModelName':model_name, 
                            'DataSetSize':train_df.shape[0], 
                            'FeaturesSize': features[0].shape[0],
                            'ProcessTime': processtime,
                            'mAPqueries': num_queries,
                            'mAP': f'mAP: {df.ap.mean():0.04f}',
                            'Precision:' : precision
                        } 
                logfile.writeLogFile(values)

            it += 1

    #Print and save logfile    
    logfile.printLogFile()
    logfile.saveLogFile_to_csv("imageretrieval",config)
    ##### END IMAGE RETRIEVAL TEST #############


    
if __name__ == "__main__":
    config = {
        "dataset_base_dir" : "/home/manager/upcschool-ai/data/FashionProduct/",
        "dataset_labels_file" : "/home/manager/upcschool-ai/data/FashionProduct/styles.csv",
        "work_dir" : "/home/manager/upcschool-ai/data/FashionProduct/processed_datalab/",
        "transforms_resize" : 332,
        "PCAdimension" : 512,
        "train_size" : "all",  # "all" / "divide"=train(60%), Eval and test (20%) / number=fixed size
        "test_validate_size": 1, #used only for train_size = fixed zize
        "batch_size" : 8,
        "top_n_image" : 10,  #multiple of 5
        "mAP_n_queries": 1,
        "log_dir" : "/home/manager/upcschool-ai/data/FashionProduct/processed_datalab/log/",
        "debug" : "False",
        "evaluate" : "True" # True/False  Process Evaluation
    }

    main(config)
