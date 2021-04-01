import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from dataset import DatasetManager, FashionProductDataset
from models import ModelManager
from features import FeaturesManager
from utils import ProcessTime, LogFile
from config import DebugConfig, DeviceConfig, FoldersConfig, ModelBatchSizeConfig, ModelTrainConfig

device = DeviceConfig.DEVICE
DEBUG = DebugConfig.DEBUG

def tune_batch_norm_statistics(model, loader):
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

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

def train_epoch(train_loader, model, optimizer, criterion):
    # switch to train mode
    model.train()

    loader_len = len(train_loader)
    log_interval = 10
    batch_loss = 0.0
    batch_corrects = 0
    total_train_images = 0
    for batch_idx, (images_batch, labels_batch) in enumerate(train_loader):
        # move images to gpu
        images_batch = images_batch.to(device)
        labels_batch = labels_batch.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # compute output
        outputs = model(images_batch)

        # loss            
        loss = criterion(outputs, labels_batch)

        loss.backward()
        optimizer.step()

        # statistics
        batch_loss += loss.item() * images_batch.size(0)
        batch_corrects += accuracy(outputs,labels_batch)
        total_train_images += images_batch.size(0)

        if (batch_idx) % log_interval == 0:
            print ('Training batch Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}' 
                .format(batch_idx, loader_len, batch_loss/total_train_images, batch_corrects/total_train_images))
    epoch_loss = batch_loss / total_train_images
    epoch_acc =  batch_corrects / total_train_images

    return epoch_loss,epoch_acc

def val_epoch(validate_loader, model, criterion):
    # switch to eval mode
    model.eval()         

    total_val_images = 0
    loader_len = len(validate_loader)
    batch_loss = 0.0
    batch_corrects = 0
    log_interval = 10

    with torch.no_grad():
        for batch_idx, (images_batch, labels_batch) in enumerate(validate_loader):

            # move images to gpu
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)

            # compute output
            outputs = model(images_batch)

            # loss
            loss = criterion(outputs, labels_batch)

            # statistics
            batch_loss += loss.item() * images_batch.size(0)
            batch_corrects += accuracy(outputs,labels_batch)
            total_val_images += images_batch.size(0)

            if (batch_idx) % log_interval == 0:
                print ('Validating batch Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}' 
                    .format(batch_idx, loader_len, batch_loss/total_val_images, batch_corrects/total_val_images))
        epoch_loss = batch_loss / total_val_images
        epoch_acc = batch_corrects / total_val_images

        return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader=None):
    #Create timer to calculate the process time
    proctimer = ProcessTime()
    proctimer.start()

    avg_train_loss = []
    avg_train_acc = []
    avg_val_loss = []
    avg_val_acc = []

    if(ModelTrainConfig.TRAIN_TYPE=="transferlearning"):
        #create logfile for save statistics - extract features
        fields = ['ModelName', 'DataSetSize','TransformsResize', 'ParametersCount', 'OutputFeatures', 'ProcessTime']
        logfile = LogFile(fields)

        tune_batch_norm_statistics(model, train_loader)

        # Save raw model
        model.save_model()

        #LOG
        processtime = proctimer.stop()
        values = {  'ModelName': model.get_name, 
                    'DataSetSize': len(train_loader), 
                    'TransformsResize': model.get_input_resize(),
                    'ParametersCount': model.count_parameters(),
                    'OutputFeatures': model.get_output_features(),
                    'ProcessTime': processtime
                } 
        logfile.writeLogFile(values)

        #Print and save logfile    
        logfile.printLogFile()
        logfile.saveLogFile_to_csv(model.get_name() + '_transfer_learning_model_log')

        
    elif(ModelTrainConfig.TRAIN_TYPE=="scratch"):
        #create logfile for training statistics
        fields = ['ModelName','Criterion','Optimizer','lr','Epoch', 'Step','Loss','Accuracy', 'Time']
        logfile = LogFile(fields) 

        loss_min = np.Inf
        best_acc = 0.
        num_epochs = ModelTrainConfig.get_num_epochs(model_name=model.get_name())
        for epoch in range(1, num_epochs + 1):
            #epoch_train_loss,epoch_train_acc = train_epoch(train_loader=train_loader,model=model.get_model(),optimizer=model.get_optimizer(),criterion=model.get_criterion())
            #avg_train_loss.append(epoch_train_loss)
            #avg_train_acc.append(epoch_train_acc)

            epoch_val_loss,epoch_val_acc = val_epoch(validate_loader=val_loader,model=model.get_model(),criterion=model.get_criterion())
            avg_val_loss.append(epoch_val_loss)
            avg_val_acc.append(epoch_val_acc)

            # early stopping
            network_learned = epoch_val_loss < loss_min     #based on loss
            #network_learned = epoch_val_acc > best_acc     #based on accuracy
            if network_learned:
                epochs_no_improve = 0
                print('Validation loss decreased ({:.2f} --> {:.2f}).  Saving model ...'
                    .format(loss_min,epoch_val_loss))
                #print('Better Accuracy ({:.2f} --> {:.2f}).  Saving model ...'
                #    .format(best_acc,epoch_val_acc))

                loss_min = epoch_val_loss
                #best_acc = epoch_val_acc

                # Save model
                model.save_model()
            else:
                epochs_no_improve += 1

            print ('Train Epoch [{}/{}], Epoch Loss: {:.4f}, Epoch Accuracy: {:.4f}' 
                .format(epoch, num_epochs, epoch_val_loss, epoch_val_acc))


            processtime = proctimer.current_time()
            values = {  'ModelName': model.get_name, 
                    'Criterion': 'CrossEntropyLoss', #model.get_criterion().__name__
                    'Optimizer': 'SGD',#model.get_optimizer().__name__,
                    'lr': ModelTrainConfig.get_learning_rate,
                    'Epoch': epoch, 
                    'Loss': epoch_val_loss, 
                    'Accuracy': epoch_val_acc,
                    'Time': processtime
                    } 
            logfile.writeLogFile(values)
            logfile.saveLogFile_to_csv(model.get_name() + '_scratch_model_log')

            if epochs_no_improve == ModelTrainConfig.PATIENCE: #patience: Number of epochs to wait if no improvement and then stop the training.
                print('Early stopping!' )
                break
        proctimer.stop()
    else:
        raise Exception('Train type "{0}" unknow.'.format(ModelTrainConfig.TRAIN_TYPE))

    return avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc

def prepare_data(dataset_base_dir, labels_file, process_dir, train_size, validate_test_size, clean_process_dir):
    dataset_manager = DatasetManager()      

    train_df, test_df, validate_df = dataset_manager.split_dataset(dataset_base_dir=dataset_base_dir,
                                                    original_labels_file=labels_file,
                                                    process_dir=process_dir,
                                                    clean_process_dir=clean_process_dir,
                                                    train_size=train_size,
                                                    fixed_validate_test_size=validate_test_size
                                                    )
        
    train_df.reset_index(drop=True, inplace=True)
    if DEBUG:print(train_df.head(10))
    if not test_df is None:
        test_df.reset_index(drop=True, inplace=True)
        validate_df.reset_index(drop=True, inplace=True)

    return train_df, test_df, validate_df

def test_model(model, test_loader):
    #create logfile for testing statistics
    fields = ['ModelName','Criterion','Step','Loss','Accuracy', 'Time']
    logfile = LogFile(fields) 

    # switch to eval mode
    model.eval()         

    total_test_images = 0
    loader_len = len(test_loader)
    batch_loss = 0.0
    batch_corrects = 0
    log_interval = 10

    #Create timer to calculate the process time
    proctimer = ProcessTime()
    proctimer.start()

    with torch.no_grad():
        for batch_idx, (images_batch, labels_batch) in enumerate(test_loader):

            # move images to gpu
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)

            # compute output
            outputs = model(images_batch)

            # loss
            loss = criterion(outputs, labels_batch)

            # statistics
            batch_loss += loss.item() * images_batch.size(0)
            batch_corrects += accuracy(outputs,labels_batch)
            total_test_images += images_batch.size(0)

            if (batch_idx) % log_interval == 0:
                print ('Testing batch Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}' 
                    .format(batch_idx, loader_len, batch_loss/total_test_images, batch_corrects/total_test_images))

            processtime = proctimer.current_time()
            values = {  'ModelName': model.get_name, 
                    'Criterion': 'CrossEntropyLoss', #model.get_criterion().__name__
                    'Step': batch_idx, 
                    'Loss': batch_loss/total_test_images, 
                    'Accuracy': batch_corrects/total_test_images,
                    'Time': processtime
                    } 
            logfile.writeLogFile(values)
            logfile.saveLogFile_to_csv(model.get_name() + '_scratch_model_log')


        test_loss = batch_loss / total_test_images
        test_acc = batch_corrects / total_test_images

        return test_loss, test_acc


def train():

    # The path of original dataset
    dataset_base_dir = FoldersConfig.DATASET_BASE_DIR
    labels_file = FoldersConfig.DATASET_LABELS_DIR

    # Work directory
    work_dir = FoldersConfig.WORK_DIR

    train_df, test_df, validate_df = prepare_data(dataset_base_dir=dataset_base_dir,
                                                    labels_file=labels_file,
                                                    process_dir=work_dir,
                                                    clean_process_dir=False,
                                                    train_size=ModelTrainConfig.TRAIN_SIZE,
                                                    validate_test_size=ModelTrainConfig.TEST_VALIDATE_SIZE)

    model_manager = ModelManager(device, work_dir)
    models_list = model_manager.get_model_names()

    ########### TRAIN MODEL IF NOT TRAINED PREVIOUSLY #####################
    pending_models_train = []
    for model_name in models_list:        
        if not model_manager.is_model_saved(model_name):
            # Get raw model
            if(ModelTrainConfig.TRAIN_TYPE=="transferlearning"):
                pending_models_train.append(model_manager.get_transferlearning_model(model_name))
            elif(ModelTrainConfig.TRAIN_TYPE=="scratch"):
                pending_models_train.append(model_manager.get_scratch_model(model_name))
            else:
                raise Exception('Train type "{0}" unknow.'.format(ModelTrainConfig.TRAIN_TYPE))
        else:
            # Test model can be loaded from checkpoint
            loaded_model = model_manager.load_from_checkpoint(model_name)

    if len(pending_models_train) > 0 :

        for model in pending_models_train:
            model_name = model.get_name()
            if DEBUG:print(f'Training model {model_name} ....')

            # Define input transformations
            transform = model.get_input_transform()
            batch_size = ModelBatchSizeConfig.get_batch_size(model_name)
            train_dataset = FashionProductDataset(dataset_base_dir, train_df, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

            val_loader = None
            if(ModelTrainConfig.TRAIN_TYPE=="scratch"):
                val_dataset = FashionProductDataset(dataset_base_dir, validate_df, transform=transform)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
                test_dataset = FashionProductDataset(dataset_base_dir, test_df, transform=transform)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

            if DEBUG:print(model.get_model())

            # Train
            avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc = train_model(model, train_loader, val_loader) 
            
            if(ModelTrainConfig.TRAIN_TYPE=="scratch"):
                # visualize the loss and save graph
                fig = plt.figure(figsize=(10,8))
                plt.plot(range(1,len(avg_train_loss)+1),avg_train_loss, label='Training Loss')
                plt.plot(range(1,len(avg_val_loss)+1),avg_val_loss,label='Validation Loss')
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.ylim(0, 0.8)
                plt.xlim(0, len(avg_train_loss)+1)
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()
                #fig.savefig(os.path.join(model.get_model_dir,'loss_plot.png'), bbox_inches='tight')

                # TEST
                test_loss,test_acc = test_model(model, test_loader) 
                print ('Test (Overall), Loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss,test_acc))

if __name__ == "__main__":
    train()
