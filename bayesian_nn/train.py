# Trainin script for bayesien neural network on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import SimpleBayesianCNN

if __name__ == "__main__":
    # Use a subset of MNIST for quick training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Use a smaller subset of the training set for quick experimentation
    indices = torch.arange(0, 10000)
    train_subset = torch.utils.data.Subset(trainset, indices)
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleBayesianCNN(dropout_p=0.2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Simple training loop
    for epoch in range(3): # small number of epochs for demonstration
        model.train()
        total_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(trainloader)}")

    torch.save(model.state_dict(), "bayesian_nn_model.pth")