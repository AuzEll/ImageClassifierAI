import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.autograd import Variable
import argparse
import json

# Main program function
def main():
    # Get input arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_directory', type = str, action="store")
    parser.add_argument("--save_dir", type = str, default = "", help = "directory to save checkpoints")
    parser.add_argument("--arch", type = str, default = "vgg16", help = "choose architecture")
    parser.add_argument("--learning_rate", type = float, default = 0.001, help = "set hyperparameter: learning rate")
    parser.add_argument("--hidden_unit", type = int, default = 2048, help = "set hyperparameter: hidden unit")
    parser.add_argument("--epochs", type = int, default = 1, help = "set hyperparameter: epochs")
    parser.add_argument("--gpu", type = str, default = "True", help = "Use GPU for training (True or False)")
    
    data_directory = parser.parse_args().data_directory
    save_dir = parser.parse_args().save_dir
    arch = parser.parse_args().arch
    learning_rate = parser.parse_args().learning_rate
    hidden_unit = parser.parse_args().hidden_unit
    epochs = parser.parse_args().epochs
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
        
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'
    
    if save_dir != "" and save_dir[-1] != "/":
        save_dir += "/"
    
    # Defining the transforms for the training, validation and testing sets
    #This transform is slightly different than in Part 1 cause here it was giving me errors with the dataloader for some reason.
    training_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])])

    validation_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])])

    testing_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])])

    # Loading the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=training_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=validation_transform)
    test_dataset  = datasets.ImageFolder(test_dir, transform=testing_transform)
    
    # Defining the dataloaders
    batch_size = 32
    #batch sizes for the trainloaders were 64 previously but it would not work when training the model on CPU.
    #Changing these back to 64 for training on GPU may produce a more accurate model.
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
    testloader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    print("Building network...") # Building the network 
    model_constructor = getattr(models, arch, None)
    if model_constructor is not None:
        model = model_constructor(pretrained=True)
    else:
        raise ValueError(f"Model architecture '{arch}' not recognized.")
    
    for param in model.parameters():
        param.require_grad = False

    # Determining input size based on model architecture
    if "vgg" in arch:
        input_size = 25088
    if "densenet" in arch:
        input_size = 1024
    if "resnet" in arch:
        input_size = 2048
    classifier = nn.Sequential(nn.Linear(input_size, hidden_unit),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_unit, len(train_dataset.classes)),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)
    #DISCLAIMER: Running this program with CPU instead of GPU takes significantly longer. I tested it and it works. It's just
    # very long. About 25 minutes per loss & accuracy print. Hence why I have GPU enabled as the default for the hyperparameter.
    print(model)
    print("Complete!\n")
    
    print("Training network...") # Training the network
    steps = 0
    print_every = 5

    for epoch in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            steps += 1
            if device == torch.device("cuda"):
                images = Variable(images.float().cuda())
                labels = Variable(labels.long().cuda())
            else:
                images = Variable(images.float())
                labels = Variable(labels.long())
            optimizer.zero_grad()

            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {test_loss/len(validloader):.3f}.. "
                      f"Accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    print("Complete!\n")
    
    print("Testing accuracy...") # Testing the network accuracy
    test_loss = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy/len(testloader)*100:.3f}%")
    print("Complete\n")
    
    print("Saving checkpoint...") # Saving the model to a checkpoint
    model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {'model': model_constructor(pretrained=True),
                  'input_size': input_size,
                  'output_size': 102,
                  'hidden_layers': [hidden_unit],
                  'classifier': model.classifier,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, save_dir + 'checkpoint.pth')
    print("Complete! Model succesfully saved!\n")

# Call to main function to run the program
if __name__ == "__main__":
    main()