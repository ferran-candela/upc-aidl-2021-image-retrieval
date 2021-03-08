import os
import torch
import pickle
from finetunePCA import PCA_VarianceDimension,PCA_Tune
from pretained_models import PretainedModels
from data_preparation import Prepare_Data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    config = {
        "dataset_base_dir" : "/home/manager/upcschool-ai/data/FashionProduct/",
        "dataset_labels_file" : "/home/manager/upcschool-ai/data/FashionProduct/styles.csv",
        "work_dir" : "/home/manager/upcschool-ai/data/FashionProduct/processed_datalab/",
        "transforms_resize" : 332,
        "PCAdimension" : 13,
        "train_size" : "all",  # "all" / "divide"=train(60%), Eval and test (20%) / number=fixed size
        "test_validate_size": 1, #used only for train_size = fixed zize
        "batch_size" : 8,
        "top_n_image" : 10,  #multiple of 5
        "mAP_n_queries": 60,
        "log_dir" : "/home/manager/upcschool-ai/data/FashionProduct/processed_datalab/log/",
        "debug" : "False" # True/False
    }

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

    pretained_models = PretainedModels(device)    
    pretained_list_models = pretained_models.get_pretained_models_names()

    train_df, test_df, validate_df = Prepare_Data(img_dir=dataset_image_dir,
                                                    original_labels_file=labels_file,
                                                    process_dir=work_dir,
                                                    clean_process_dir=False,
                                                    fixed_train_size=train_size,
                                                    fixed_validate_test_size=validate_test_size,
                                                    debug=debug
                                                    )
    for model_name in pretained_list_models:
        model_dir = os.path.join(work_dir, model_name)
        features_file = os.path.join(model_dir, 'features_full.pickle')
        if not os.path.isfile(features_file):
            continue
        full_features = pickle.load(open(features_file , 'rb'))
        PCA_VarianceDimension(model_name,full_features,config["PCAdimension"])
        PCA_Tune(pretained_models,model_name,full_features,train_df,config["mAP_n_queries"])
