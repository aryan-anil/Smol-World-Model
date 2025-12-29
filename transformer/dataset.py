import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms

class H5Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        
  
        with h5py.File(h5_path, 'r') as h5_file:
            self.length = len(h5_file['frames'])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as h5_file:
            frames = h5_file['frames'][idx]
            user_actions = h5_file['user_actions'][idx]
            ghost_actions = h5_file['ghost_actions'][idx]
            next_frames = h5_file['next_frames'][idx]
            reset_tokens = h5_file['resets'][idx]
            
           
            frames = torch.from_numpy(frames).float()
            next_frames = torch.from_numpy(next_frames).float()
            
            # Handle user actions (single action)
            if isinstance(user_actions, bytes):
                user_actions = int(user_actions.decode())
            else:
                user_actions = int(user_actions)
            user_actions = torch.tensor(user_actions).long()
            # Handle reset tokens (single token)
            reset_tokens = int(reset_tokens)
            
            # Handle ghost actions (array of 4 actions)
            ghost_actions_list = []
            for action in ghost_actions:
                if isinstance(action, bytes):
                    ghost_actions_list.append(int(action.decode()))
                else:
                    ghost_actions_list.append(int(action))
            ghost_actions = torch.tensor(ghost_actions_list).long()
            
            # Apply transforms if specified
            if self.transform:
                frames = self.transform(frames)
                next_frames = self.transform(next_frames)
            
            # Normalize images to [0, 1] if needed
            if frames.max() > 1.0:
                frames = frames / 255.0
                next_frames = next_frames / 255.0
                
            # Permute dimensions if needed (assuming HDF5 stores as [H, W, C])
            if frames.shape[-1] == 3:  # If channels are last dimension
                frames = frames.permute(2, 0, 1)  # [C, H, W]
                next_frames = next_frames.permute(2, 0, 1)  # [C, H, W]
            
            return frames, user_actions, ghost_actions, next_frames, reset_tokens
