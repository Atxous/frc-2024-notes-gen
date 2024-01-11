from torch.utils.data import Dataset, DataLoader
import torch
import os
import torchvision

class StandardDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = os.listdir(root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.files[idx])
        image = torchvision.io.read_image(img_name)
        if self.transform:
            image = self.transform(image)
        return image
    
class Scale0to1(object):
    def __call__(self, image):
        return image / 255



def build_dataset(root, batch_size, img_size, device = "cpu", shuffle = True, pin_memory = True):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size, antialias = True),
        Scale0to1()
    ])
    return DataLoader(StandardDataset(root, transform = transforms), batch_size = batch_size, shuffle = shuffle, pin_memory = pin_memory, generator = torch.Generator(device = device))