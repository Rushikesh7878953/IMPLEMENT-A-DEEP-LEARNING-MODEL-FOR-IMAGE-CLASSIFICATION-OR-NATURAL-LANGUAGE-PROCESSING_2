# pip install torch torchvision matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Loading The MNIST Dataset
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

# Download And load Training And Test Data
trainset=torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)

testset=torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)
testloader=torch.utils.data.DataLoader(testset, batch_size=64,shuffle=False)

# Step 2: Define A Simple Feed-Forward Neural Network (MLP)
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP,self).__init__()
        self.fc1=nn.Linear(28*28,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self,x):
        x=x.view(-1,28*28)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x

# Step 3: Instantiate The Model,Loss Function,And Optimizer
model=SimpleMLP()
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)

# Step 4: Train The Model
num_epochs=5
for epoch in range(num_epochs):
    running_loss=0.0
    for i,(inputs,labels) in enumerate(trainloader, 0):
        optimizer.zero_grad()

# Forward Pass
        outputs=model(inputs)
        loss=criterion(outputs,labels)

# Backward Pass and Optimize
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if i % 100==99:
            print(f"[Epoch{epoch+1},Mini-batch{i+1}]loss:{running_loss/100:.3f}")
            running_loss=0.0

print("Finished Training")

# Step 5: Test The Model
correct=0
total=0
with torch.no_grad():
    for data in testloader:
        images,labels=data
        outputs=model(images)
        _, predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100*correct/total}%')

# Step 6: Visualize Some Test Images
dataiter=iter(testloader)
images,labels=next(dataiter)

# Show Images
def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)),cmap="gray")
    plt.show()

# Print The Labels And Display The Images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{labels[j].item()}'for j in range(4)))