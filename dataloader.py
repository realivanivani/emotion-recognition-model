
import torch
from torch.utils.data import Dataset

class AffectNetDataset(Dataset):
    def __init__(self, metadata, image_dir, transform=None):
        self.metadata = metadata
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, item['image_path'])
        image = load_image(img_path)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        # Get labels
        expression = item['expression']
        valence = item['valence']
        arousal = item['arousal']
        
        # Convert to tensors
        expression = torch.tensor(expression, dtype=torch.long)
        valence = torch.tensor(valence, dtype=torch.float32)
        arousal = torch.tensor(arousal, dtype=torch.float32)
        
        return {
            'image': image,
            'expression': expression,
            'valence': valence,
            'arousal': arousal
        }
