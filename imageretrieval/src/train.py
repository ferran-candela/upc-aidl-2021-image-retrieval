import torch

from .dataset import DatasetManager

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def tuning_batch_norm_statistics(model, loader):
    # The batch norm statistics for this network match those of the ImageNet dataset. 
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

def train(config):
    # The path of original dataset
    dataset_base_dir = config["dataset_base_dir"]
    
    labels_file = config["dataset_labels_file"]

    if config["debug"]=='True':
        debug = True
    else:
        debug = False

    # Work directory
    work_dir = config["work_dir"]

    dataset_manager = DatasetManager()      

    train_df, test_df, validate_df = dataset_manager.Prepare_Data(img_dir=dataset_image_dir,
                                                    original_labels_file=labels_file,
                                                    process_dir=work_dir,
                                                    clean_process_dir=False,
                                                    fixed_train_size=config["train_size"],
                                                    fixed_validate_test_size=config["test_validate_size"],
                                                    debug=debug
                                                    )
        
    train_df.reset_index(drop=True, inplace=True)
    if debug:print(train_df.head(10))
    if not test_df is None:
        test_df.reset_index(drop=True, inplace=True)
        validate_df.reset_index(drop=True, inplace=True)

    pretrained_models = PretainedModels(device)    
    pretrained_list_models = pretrained_models.get_pretained_models_names()

    ########### TRAIN MODEL IF NOT TRAINED PREVIOUSLY #####################
    pending_models_extract = []
    for model_name in pretrained_list_models:
        model_dir = os.path.join(work_dir, model_name)
        features_file = os.path.join(model_dir, 'features.pickle')
        if not os.path.isfile(features_file):
            pending_models_extract.append(model_name)

    if len(pending_models_extract) > 0 :
        transform = transforms.Compose([
            transforms.Resize(config["transforms_resize"]),
            transforms.CenterCrop(config["transforms_resize"]-32),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = MyDataset(dataset_image_dir, train_df, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True)

        #create logfile for save statistics - extract features
        fields = ['ModelName', 'DataSetSize','TransformsResize','ParametersCount', 'FeaturesSize', 'PostProcFeaturesSize', 'ProcessTime']
        logfile = LogFile(fields)        

        #Create timer to calculate the process time
        proctimer = ProcessTime()

        for model_name in pending_models_extract:
            #PATH to save features
            model_dir = os.path.join(work_dir, model_name)

            # Load pretrained model
            pretrained_model = pretrained_models.load_pretrained_model(model_name)
            if debug:print(pretrained_model)
            pretrained_model.to(device)

            proctimer.start()

            #put the network into train mode and do a pass over the dataset without doing any backpropagation
            pretrained_model = pretrained_models.tuning_batch_norm_statistics(pretrained_model,train_loader) 
            #extract features
            features = pretrained_models.extract_features(pretrained_model,train_loader,transform) 
            features_size = features[0].shape[0]
            #save complete features
            features_file = os.path.join(model_dir, 'features_full.pickle')
            pickle.dump(features , open(features_file, 'wb'))

            #normalize features            
            features = pretrained_models.postprocessing_features(features) 
            postproc_features_size = features[0].shape[0]

            #save features
            features_file = os.path.join(model_dir, 'features.pickle')
            pickle.dump(features , open(features_file, 'wb'))

            #LOG
            processtime = proctimer.stop()
            values = {  'ModelName':model_name, 
                        'DataSetSize':train_df.shape[0], 
                        'TransformsResize':config["transforms_resize"],
                        'ParametersCount':pretrained_models.Count_Parameters(pretrained_model), 
                        'FeaturesSize': features_size,
                        'PostProcFeaturesSize': postproc_features_size,
                        'ProcessTime':processtime
                    } 
            logfile.writeLogFile(values)

        #Print and save logfile    
        logfile.printLogFile()
        logfile.saveLogFile_to_csv("extractfeatures", config)


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
    
    train(config)
