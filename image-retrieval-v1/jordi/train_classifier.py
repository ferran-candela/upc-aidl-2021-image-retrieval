import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
import matplotlib.pyplot as plt
from utils import ProcessTime, LogFile
from train_dataset import MyTrainDataset
from data_preparation import Prepare_Data
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

def train_model(train_loader, model, criterion, optimizer, num_epochs, device):

    # switch to train mode
    model.train()

    train_loss = []
    train_acc = []

    loader_len = len(train_loader)
    log_interval = 50

    for epoch in range(1, num_epochs+1):
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
            batch_loss += loss.item() #* images.size(0)
            batch_corrects += accuracy(outputs,labels_batch)
            total_train_images += images_batch.size(0)

            if (batch_idx) % log_interval == 0:
                print ('Batch Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Corrects: {:.4f}' 
                    .format(epoch, num_epochs, batch_idx, loader_len, batch_loss, batch_corrects))

        
        epoch_loss = batch_loss / loader_len
        epoch_acc = batch_corrects / total_train_images
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        print ('Train Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}' 
                .format(epoch, num_epochs, batch_idx, loader_len, np.mean(train_loss), np.mean(train_acc)))

def validate_model(validate_loader, model, criterion, num_epochs, save_path, device):        
    # switch to eval mode
    model.eval()            
    val_loss = []
    val_acc = []
    loss_min = np.Inf
    best_acc = 0.
    total_val_images = 0
    loader_len = len(validate_loader)
    log_interval = 50
    for epoch in range(1, num_epochs+1):
        with torch.no_grad():
            for batch_idx, (images_batch, labels_batch) in enumerate(validate_loader):

                # move images to gpu
                images_batch = images_batch.to(device)
                labels_batch = labels_batch.to(device)

                # compute output
                output = model(images_batch)

                # loss
                loss = criterion(output, labels_batch)

                # statistics
                batch_loss += loss.item() #* images.size(0)
                batch_corrects += accuracy(outputs,labels_batch)
                total_val_images += images_batch.size(0)

                if (batch_idx) % print_every == 0:
                    print ('Batch Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Corrects: {:.4f}' 
                        .format(epoch, num_epochs, batch_idx, loader_len, batch_loss, batch_corrects))

            epoch_loss = batch_loss / loader_len
            epoch_acc = batch_corrects / total_val_images
            val_loss.append(epoch_loss)
            val_acc.append(epoch_acc)

            network_learned = epoch_loss < loss_min
            #network_learned = epoch_acc > best_acc

            print ('Validate Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}' 
                .format(epoch, num_epochs, batch_idx, loader_len, np.mean(val_loss), np.mean(val_acc)))

            if network_learned:
                loss_min = epoch_loss
                #best_acc = epoch_acc
                print('Validation loss decreased ({:.2f} --> {:.2f}).  Saving model ...'
                    .format(loss_min,epoch_loss))
                #print('Better Accuracy ({:.2f} --> {:.2f}).  Saving model ...'
                #    .format(best_acc,epoch_acc))
                torch.save(model.state_dict(), save_path)

if __name__ == "__main__":

    config = {
        "dataset_base_dir" : "C:\\UPC\\data\\FashionProduct\\",
        "dataset_labels_file" : "C:\\UPC\data\\FashionProduct\\styles.csv",
        "work_dir" : "C:\\UPC\\data\\FashionProduct\\processed_datalab",
        "transforms_resize" : 332,
        "PCAdimension" : 10,
        "train_size" : "all",  # "all" / "divide"=train(60%), Eval and test (20%) / number=fixed size
        "test_validate_size": 1, #used only for train_size = fixed zize
        "batch_size" : 8,
        "log_dir" : "C:\\UPC\\data\\FashionProduct\\processed_datalab\\log\\",
        "retrieval_test" : "True", # True/False  Retrieval Test
        "retrieval_test_size" : 5,
        "top_k_image" : 15,  #multiple of 5
        "evaluate" : "False", # True/False  Process Evaluation
        "mAP_n_queries": 300,
        "debug" : "False"
    }


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

    #Show classes graphic
    # plt.figure(figsize=(7,20))
    # train_df.articleType.value_counts().sort_values().plot(kind='barh')
    # #How many classes have minimum 100 images
    # N_Pictures = 100
    # N_Classes = np.sum(train_df.articleType.value_counts().to_numpy() > N_Pictures)
    # temp = train_df.articleType.value_counts().sort_values(ascending=False)[:N_Classes]
    # temp[-5:]

    transform = transforms.Compose([
            transforms.Resize(config["transforms_resize"]),
            transforms.CenterCrop(config["transforms_resize"]-32),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = MyTrainDataset(dataset_image_dir, train_df, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    from torchvision.models import resnet50

    model = resnet50(pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #num classes
    class_names=pd.unique(train_df['articleTypeEncoded'])
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    #model.fc = model.fc.cuda() if torch.cuda.is_available() else model.fc

    model_name='resnet50'
    model_dir = os.path.join(work_dir, model_name)

    train_model(train_loader, model, criterion, optimizer, 1, device)

    #Validation
    #val_dataset = 
    #val_loader = DataLoader(val_dataset, batch_size=config["batchsize"], shuffle=False)
    #validate_model(val_loader,model, criterion, optimizer, 1,model_dir,device)