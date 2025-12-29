import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms


class PacmanDataset(Dataset):
    def __init__(self, hdf5_path='pacman_gameplay_data.h5', transform=None, normalize=False):
        """
        Args:
            hdf5_path (str): Path to HDF5 file
            transform (callable, optional): Optional transform to be applied on frames
            normalize (bool): Whether to apply stable diffusion normalization
        """
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.normalize = normalize
        
        with h5py.File(hdf5_path, 'r') as f:
            self.current_frames_len = len(f['frames'])
            self.next_frames_len = len(f['next_frames'])
            self.length = self.current_frames_len + self.next_frames_len
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as h5_file:
            if idx < self.current_frames_len:
                image = h5_file['frames'][idx]
              #  image = image[:160, :160, :]
            else:
                image = h5_file['next_frames'][idx - self.current_frames_len]
               # image = image[:160, :160, :]
            image = image.astype(np.float32) / 255.0
    
            if self.transform:
                image = self.transform(image)
            image = torch.from_numpy(image).permute(2, 0, 1)
            if self.normalize:
                image = 2.0 * image - 1.0
               
            return torch.clamp(image, 0, 1)


def create_pacman_dataloaders(hdf5_path, batch_size=16, train_split=0.8, normalize=False):
    """
    Create train and validation dataloaders from HDF5 file with proper normalization
    
    Args:
        hdf5_path (str): Path to HDF5 file
        batch_size (int): Batch size for dataloaders
        train_split (float): Fraction of data to use for training
        normalize (bool): Whether to apply stable diffusion normalization
    """
  
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])
    
    dataset = PacmanDataset(hdf5_path, normalize=normalize)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader


def display_normalization_comparison(hdf5_path, num_samples=5):
    """
    Display original, cropped, and normalized images side by side
    """
  
    original_dataset = PacmanDataset(hdf5_path, normalize=False)
    normalized_dataset = PacmanDataset(hdf5_path, normalize=True)
    
    plt.figure(figsize=(15, 3*num_samples))
    
    for i in range(num_samples):
        idx = np.random.randint(0, len(original_dataset))
        
        
        original_img = original_dataset[idx]
        normalized_img = normalized_dataset[idx]
        
       
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.title(f"Original [{original_img.min():.1f}, {original_img.max():.1f}]")
        plt.imshow(original_img.permute(1, 2, 0))
        plt.axis('off')
        
      
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.title(f"Normalized [{normalized_img.min():.1f}, {normalized_img.max():.1f}]")
        plt.imshow(((normalized_img + 1.0) / 2.0).permute(1, 2, 0).clamp(0, 1))
        plt.axis('off')
        
      
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.title("Normalized (raw values)")
        plt.imshow(normalized_img.permute(1, 2, 0))
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    hdf5_path = r''
    
    # Display normalization comparison
    print("Displaying normalization comparison:")
    display_normalization_comparison(hdf5_path)
    
    # Create dataloaders
    train_loader, val_loader = create_pacman_dataloaders(hdf5_path, batch_size=32)
    print(f"Total batches: {len(train_loader) + len(val_loader)}")
    
    # Display a batch
    for batch_idx, images in enumerate(train_loader):
        if batch_idx == 0:
            print(f"Batch shape: {images.shape}")
            
            # Display a few images from the batch
            plt.figure(figsize=(15, 8))
            for i in range(min(8, images.shape[0])):
                plt.subplot(2, 4, i+1)
                # Convert back from [-1,1] to [0,1] for display
                img_display = (images[i] + 1) / 2
                plt.imshow(img_display.permute(1, 2, 0).cpu().numpy())
                plt.title(f"Sample {i}")
                plt.axis('off')
            plt.tight_layout()
            plt.show()
            break
    
    print("Dataset and dataloader creation complete!")
