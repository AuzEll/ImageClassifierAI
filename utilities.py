import torch
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np

# Loads checkpoint from specified filepath
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hidden_layers = checkpoint['hidden_layers']
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

# Processes a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
  
    pil_image = Image.open(image)
    
    pil_image.resize((256, 256))
    
    #The below code was borrowed from a solution from a forum post because the cropping code I used in
    #Part 1 gave me a mismatch error which made no sense to me, but this code seems to work just fine.
    center_width = pil_image.width // 2
    center_height = pil_image.height // 2
    left = center_width - 112
    upper = center_height - 112
    right = center_width + 112
    lower = center_height + 112
    im_crop = pil_image.crop((left, upper, right, lower))
    
    np_image = np.array(im_crop)
    np_image = np_image / 255
    
    mean = np.array([ 0.485, 0.456, 0.406 ])
    std = np.array([ 0.229, 0.224, 0.225 ])
    np_image = (np_image - mean) / std
    
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image
