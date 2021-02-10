import torch
from torchvision import transforms
import numpy as np

def extract_features(dataloader, pretrained_model, n_batches, device):
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.CenterCrop(128-32),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    n_batches = len(dataloader)
    i = 1    
    features = []
    with torch.no_grad():
        for image_batch, label_batch in dataloader:
            image_batch = image_batch.to(device)

            batch_features = pretrained_model(image_batch)

            # features to numpy
            batch_features = torch.squeeze(batch_features).cpu().numpy()

            # collect features
            features.append(batch_features)
            print(f'\rProcessed {i} of {n_batches} batches', end='', flush=True)

            i += 1

    # stack the features into a N x D matrix            
    features = np.vstack(features)
    return features