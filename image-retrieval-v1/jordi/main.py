import os
import torch

from dataset import MyDataset
from data_preparation import Prepare_Data
from pretained_models import PretainedModels
from utils import ProcessTime, LogFile, ImageSize
from evaluation import make_ground_truth_matrix, create_ground_truth_entries, evaluate    

from torchvision import transforms
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import time

#if not torch.cuda.is_available():
#    raise Exception("You should enable GPU")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def Print_Similarity(dataset_img, imgidx, ranking, description):
    # show the query image
    print('Image Query : ', description)
    plt.figure(figsize=(2.8,2.8))
    plt.imshow(dataset_img[imgidx][0])
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    print('Top 10 Similarity: ', description)
    fig, ax = plt.subplots(nrows=int(ranking.size/5), ncols=5, figsize=(18, 6))
    ax = ax.ravel()
    for i in range(ranking.size):
        img = dataset_img[ranking[i]][0]

        # show the image (remove ticks and grid)
        ax[i].imshow(img)
        ax[i].grid(False) 
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(i)
    plt.show()


def main(config):
    # The path of original dataset
    dataset_base_dir = config["dataset_base_dir"]
    dataset_image_dir = os.path.join(dataset_base_dir, 'images')
    labels_file = config["dataset_labels_file"]

    # Work directory
    work_dir = config["work_dir"]

    #Log directory
    if not os.path.exists(config["log_dir"]):
        os.mkdir(config["log_dir"])

    train_df, test_df, validate_df = Prepare_Data(img_dir=dataset_image_dir,
                                                    original_labels_file=labels_file,
                                                    process_dir=work_dir,
                                                    clean_process_dir=False,
                                                    fixed_train_size=64,
                                                    fixed_validate_test_size=128)
        
    train_df.reset_index(drop=True, inplace=True)
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

        #create logfile for save statistics results
        fields = ['ModelName', 'DataSetSize','TransformsResize','ParametersCount', 'ProcessTime']
        logfile = LogFile(fields)        

        #Create timer to calculate the process time
        proctimer = ProcessTime()

        for model_name in pending_models_extract:
            # Load pretrained model
            pretained_model = pretained_models.load_pretrained_model(model_name)
            pretained_model.to(device)

            proctimer.start()

            #put the network into train mode and do a pass over the dataset without doing any backpropagation
            pretained_model = pretained_models.tuning_batch_norm_statistics(pretained_model,train_loader) 
            #extract features
            features = pretained_models.extract_features(pretained_model,train_loader,transform) 
            #normalize features
            features = pretained_models.postprocessing_features(features) 

            #save features
            model_dir = os.path.join(work_dir, model_name)
            features_file = os.path.join(model_dir, 'features.pickle')
            pickle.dump(features , open(features_file, 'wb'))

            #LOG
            processtime = proctimer.stop()
            values = {  'ModelName':model_name, 
                        'DataSetSize':train_df.shape[0], 
                        'TransformsResize':config["transforms_resize"],
                        'ParametersCount':pretained_models.Count_Parameters(pretained_model), 
                        'ProcessTime':processtime
                    } 
            logfile.writeLogFile(values)

        #Print and save logfile    
        logfile.printLogFile()
        logfile.saveLogFile_to_csv()

    # Show Similarity Result
    img_ds = MyDataset(dataset_image_dir,train_df)
    # change this to the index of the image to use as a query!
    imgidx = 0
    for model_name in pretained_list_models:
        model_dir = os.path.join(config["work_dir"], model_name)
        features_file = os.path.join(model_dir , 'features.pickle')
        features = pickle.load(open(features_file , 'rb'))

        # COSINE SIMILARITY
        ranking = pretained_models.Cosine_Similarity(features,imgidx,config["top_n_image"])
        Print_Similarity(img_ds,imgidx,ranking,model_name + " - COSINE SIMILARITY")

        #EUCLIDEAN DISTANCE
        # This gives the same rankings as (negative) Euclidean distance 
        # when the features are L2 normalized (as ours are)
                
        #distances,ranking = pretained_models.Euclidean_Distance(features,imgidx,config["top_n_image"])
        #Print_Similarity(img_ds,imgidx,ranking,model_name + " - EUCLIDEAN DISTANCE")

    #Run evaluation 

    #compute the similarity matrix
    S = train_features @ train_features.T
    print(S.shape)

    num_evaluation = 60

    queries = create_ground_truth_entries(labels_file, train_df, num_evaluation)
    q_indx, y_true = make_ground_truth_matrix(train_df, queries)

    #Compute mean Average Precision (mAP)
    df = evaluate(S, y_true, q_indx)
    print(f'mAP: {df.ap.mean():0.04f}')
    
if __name__ == "__main__":
    config = {
        "dataset_base_dir" : "/home/manager/upcschool-ai/data/FashionProduct/",
        "dataset_labels_file" : "/home/manager/upcschool-ai/data/FashionProduct/styles.csv",
        "work_dir" : "/home/manager/upcschool-ai/data/FashionProduct/processed_datalab/",
        "transforms_resize" : 332,
        "batch_size" : 128,
        "top_n_image" : 5,  #multiple of 5
        "log_dir" : "/home/manager/upcschool-ai/data/FashionProduct/processed_datalab/log/"
    }

    main(config)
