import torch
from torchvision.models import vgg16

if not torch.cuda.is_available():
    raise Exception("You should enable GPU")
device = torch.device("cuda")

if __name__ == "__main__":

    pretrained_model = vgg16(pretrained=True)
    pretrained_model.eval()
    pretrained_model.to(device)