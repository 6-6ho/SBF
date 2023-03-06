from torchvision import transforms, datasets

class CustomImageDataset:
    
    def __init__(self, build_name: str):    
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
        
        data_root = './resized_data/'
        self.custom_dataset = datasets.ImageFolder(root=data_root + build_name, transform=transform)
        
    def __len__(self):
        return len(self.custom_dataset)
    
    def __getitem__(self, idx):
        return self.custom_dataset[idx]