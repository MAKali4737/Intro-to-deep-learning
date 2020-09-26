import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.datasets as dset
from torchvision import transforms
print("progress 1")
root = './data'
if not os.path.exists(root):
    os.mkdir(root)
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)
print("progress 2")
batch_size = 100
train_loader = torch.utils.data.DataLoader(
                                           dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(
                                          dataset=test_set,
                                          batch_size=batch_size,
                                          shuffle=True)
print ('total number of trainning batches: {}'.format(len(train_loader)))
print ('total number of testing batches: {}'.format(len(test_loader)))
print("progress 3")
batch_number = 0
image_number = 82
batch_index = 0
for i in train_loader:
    plt.imshow(i[batch_number][10][batch_index])
    break

print("progress 4")
input_dim = 784 #hight * width
hidden_dim = 200
output_dim = 10


print("progress 5")
class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# ANN
model = ANNModel(input_dim, hidden_dim, output_dim)
error = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print("progress 6")
model = ANNModel(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
count = 0
num_epochs = 20
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        train = Variable(images.view(-1, input_dim))
        labels = Variable(labels)
        
        
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        
        # Calculate softmax and cross entropy loss
        loss = error(outputs, labels)
        
        # Calculating gradients
        loss.backward()
        
        optimizer.step()
        count += 1
        
        if count % 50 == 0:
            
            correct = 0
            total = 0
            
            for images, labels in test_loader:
                
                test = Variable(images.view(-1, input_dim))
                
                
                outputs = model(test)
                
                
                predicted = torch.max(outputs.data, 1)[1]
                
                
                total += len(labels)
                
                
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count % 100 == 0:
                
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, round(float(loss.data.item()), 4), accuracy))



#loss ka graph
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iterations")
plt.ylabel("Loss")
plt.show()

#accuracy ka graph
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iterations")
plt.ylabel("Accuracy")
plt.show()


