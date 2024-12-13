import torch
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import argparse
import json

from utilities import *

# Main program function
def main():
    # Get input arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('path_to_image', type = str, action="store")
    parser.add_argument('checkpoint', type = str, action="store")
    parser.add_argument("--top_k", type = int, default = 5, help = "return top K most likely classes")
    parser.add_argument("--category_names", type = str, default = "cat_to_name.json", help = "Use a mapping of categories to real names")
    parser.add_argument("--gpu", type = str, default = "True", help = "Use GPU for training (True or False)")
    
    path_to_image = parser.parse_args().path_to_image
    checkpoint = parser.parse_args().checkpoint
    top_k = parser.parse_args().top_k
    category_names = parser.parse_args().category_names
    gpu = parser.parse_args().gpu
    
    # Check torch version and GPU status
    print("torch version: " + torch.__version__)
        
    if gpu == "True" or gpu == "true":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU enabled: True\n")
        else:
            print("GPU is not available. Proceeding with CPU")
            device = torch.device("cpu")
    elif gpu == "False" or gpu == "false":
        device = torch.device("cpu")
        print("GPU enabled: False\n")
    else:
        raise ValueError("A value other than 'True' or 'False' was parsed in --gpu")
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    print("Loading model...") # Loading the model from checkpoint
    model = load_checkpoint(checkpoint)
    print(model)
    print("Model loaded!\n")
    
    print(f"Predicting Top {top_k} image classes...") # Predicting the class from the image file
    model.to(device)
    
    image = process_image(path_to_image)
    image = torch.from_numpy(image)
    image = image.float().unsqueeze(0)
    image = image.to(device)
    model.eval()

    with torch.no_grad():
        output = model.forward(image)
        output = torch.exp(output)
        
        ps = output.topk(top_k, dim=1)
        probs = ps[0].tolist()[0]
        idx_to_class = dict(zip(model.class_to_idx.values(), model.class_to_idx.keys()))
        top_indices = np.array(ps[1][0].cpu().numpy())
        classes = [idx_to_class[x] for x in top_indices]

    # Printing class names and probabilities
    class_names = []
    for idx in range(len(classes)):
        class_names.append(cat_to_name[str(classes[idx])])
        
    for i in range(top_k):
        print(f"{i+1}. {class_names[i].title()} (Probability: {probs[i]*100:.2f}%)")
    
    print("\nComplete! Classes successfully predicted!")

# Call to main function to run the program
if __name__ == "__main__":
    main()