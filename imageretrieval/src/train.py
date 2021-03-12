import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import DatasetManager
from dataset import FashionProductDataset
from models import ModelManager
from utils import ProcessTime, LogFile, ImageSize

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
debug = False

def tune_batch_norm_statistics(model, loader):
    # The batch norm statistics for this network match those of the ImageNet dataset. 
    # We can use a trick to get them to match our dataset. The idea is to put the network into train mode and do a pass over the dataset without doing any backpropagation. 
    # This will cause the network to update the batch norm statistics for the model without modifying the weights. This can sometimes improve results.        
    model_raw = model.get_model()
    model_raw.train()
    n_batches = len(loader)
    i = 1
    for image_batch, image_id in loader:
        # move batch to device and forward pass through network
        model_raw(image_batch.to(model.get_device()))
        print(f'\rTuning batch norm statistics {i}/{n_batches}', end='', flush=True)
        i += 1

def train_model(model, loader):
    if(model.is_pretrained):
        tune_batch_norm_statistics(model, loader)
    else:
        # TODO: Here is where we have to implement the custom training
        pass

def prepare_data(dataset_image_dir, labels_file, process_dir, train_size, validate_test_size, clean_process_dir):
    dataset_manager = DatasetManager()      

    train_df, test_df, validate_df = dataset_manager.split_dataset(img_dir=dataset_image_dir,
                                                    original_labels_file=labels_file,
                                                    process_dir=process_dir,
                                                    clean_process_dir=clean_process_dir,
                                                    train_size=train_size,
                                                    fixed_validate_test_size=validate_test_size,
                                                    debug=debug
                                                    )
        
    train_df.reset_index(drop=True, inplace=True)
    if debug:print(train_df.head(10))
    if not test_df is None:
        test_df.reset_index(drop=True, inplace=True)
        validate_df.reset_index(drop=True, inplace=True)

    return train_df, test_df, validate_df

def train(config):
    if config["debug"]=='True':
        debug = True
    else:
        debug = False

    # The path of original dataset
    dataset_base_dir = config["dataset_base_dir"]
    labels_file = config["dataset_labels_file"]

    # Work directory
    work_dir = config["work_dir"]

    train_df, test_df, validate_df = prepare_data(dataset_image_dir=dataset_base_dir,
                                                    labels_file=labels_file,
                                                    process_dir=work_dir,
                                                    clean_process_dir=False,
                                                    train_size=config["train_size"],
                                                    validate_test_size=config["test_validate_size"])

    model_manager = ModelManager(device, work_dir)
    models_list = model_manager.get_model_names()
  
    ########### TRAIN MODEL IF NOT TRAINED PREVIOUSLY #####################
    pending_models_train = []
    for model_name in models_list:        
        if not model_manager.is_model_saved(model_name):
            pending_models_train.append(model_name)

    if len(pending_models_train) > 0 :
        #create logfile for save statistics - extract features
        fields = ['ModelName', 'DataSetSize','TransformsResize', 'ParametersCount', 'ProcessTime']
        logfile = LogFile(fields)

        #Create timer to calculate the process time
        proctimer = ProcessTime()

        for model_name in pending_models_train:
            # Load raw model
            model = model_manager.get_raw_model(model_name)

            # Define input transformations
            transform = transforms.Compose([
                transforms.Resize(model.get_input_resize() + 32),
                transforms.CenterCrop(model.get_input_resize() - 32),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            train_dataset = FashionProductDataset(dataset_base_dir, train_df, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True)

            if debug:print(model.get_model())
            model.to_device()

            proctimer.start()

            # Train
            train_model(model, train_loader) 
            
            # Save raw model
            model.save_model()

            #LOG
            processtime = proctimer.stop()
            values = {  'ModelName': model_name, 
                        'DataSetSize': train_df.shape[0], 
                        'TransformsResize': model.get_input_resize(),
                        'ParametersCount': model.count_parameters(), 
                        'ProcessTime': processtime
                    } 
            logfile.writeLogFile(values)

        #Print and save logfile    
        logfile.printLogFile()
        logfile.saveLogFile_to_csv("train_raw_log", config)


if __name__ == "__main__":

    config = {
        "dataset_base_dir" : "/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full/fashion-dataset",
        "dataset_labels_file" : "/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full/fashion-dataset/styles.csv",
        "work_dir" : "/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full_Subset",
        "transforms_resize" : 332,
        "PCAdimension" : 512,
        "train_size" : 500,  # "all" / "divide"=train(60%), Eval and test (20%) / number=fixed size
        "test_validate_size": 500, #used only for train_size = fixed zize
        "batch_size" : 8,
        "top_n_image" : 10,  #multiple of 5
        "mAP_n_queries": 1,
        "log_dir" : "/home/fcandela/src/upc/upc-jmc-project/datasets/Fashion_Product_Full_Subset/log/",
        "debug" : "True",
        "evaluate" : "True" # True/False  Process Evaluation
    }
    
    train(config)
