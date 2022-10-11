import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logistic
from tqdm import tqdm
from skimage import io
from matplotlib import pyplot as plt
import os
import numpy as np

#net arcs
class Net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        #two layer cnn
        # self.conv1 = nn.Conv2d(in_channel, 2, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(2, 3, 5)
        # self.fc1 = nn.Linear(3 * 4 * 4, out_channel)

        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 2)
        #one layer cnn
        self.conv1 = nn.Conv2d(in_channel, 1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1 * 12 * 12, out_channel)
        

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = self.fc1(x)
        #x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x

#print losses plot
def plot_loss(losses):         
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('100 mini-batches')
    plt.title('losses for Deep nrural network')
    plt.show()

#evaluate the validation set, output num of correct labeling
def evaluate(val_loader, net):
    net.eval() 
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            # calculate outputs 
            outputs = net(images)
            # choose the highest prediction
            _, predicted = torch.max(outputs.data, 1)
    return labels.size(0), (predicted == labels).sum().item()

#output the train_loader and val_loader
def image_process(filename):
    all_image = io.imread_collection(filename)

    #train_08000_train_47999 len 40000
    train_image = np.array(all_image[8000:48000])/255 #normalize the data
    train_image = torch.FloatTensor(train_image[:, np.newaxis, ...]) 
    train_label = torch.LongTensor(logistic.txt_process('train.txt'))

    #val 48000- 57999 len 10000
    val_image = np.array(all_image[48000:58000])/255 #normalize the data
    val_image = torch.FloatTensor(val_image[:, np.newaxis, ...])
    val_label = torch.LongTensor(logistic.txt_process('val.txt'))

    #create train data loader
    train_dataset = torch.utils.data.TensorDataset(train_image, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

    #create validation data loader
    val_dataset = torch.utils.data.TensorDataset(val_image, val_label)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=2)
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # # show images
    # imshow(utils.make_grid(images))
    # print(labels[i] for i in range(4))
    # return

    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # fig, ax = plt.subplots()
    # plt.imshow( images)
    # plt.show() 
    # print(labels)
    print("finish data processing\nstart training")
    return  train_loader, val_loader



def main():

    #image processing    
    train_loader, val_loader = image_process('image_data/*.png')
    
    best_acc = 0
    PATH = './garen_net.pth'
    losses = []

    net = Net(1,2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #Adam
    for epoch in range(1):  #   number of epoches
        net.train() #train mode
        running_loss = 0.0
        for i, data in tqdm(enumerate(train_loader, 0)):
            # get data -> [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward, backward, optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print loss
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches

                #print loss
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')

                #add loss to list for plot
                losses.append(running_loss / 100)

                #zero the loss
                running_loss = 0.0
                
                #testing
                total, correct = evaluate(val_loader, net)

                #print accuracy
                print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

                #save the best model
                if 100 * correct / total > best_acc:
                    best_acc = 100 * correct / total
                    torch.save(net.state_dict(), PATH)
                
    #print losses          
    plot_loss(losses)

    print('-------Finished------')

if __name__ == "__main__":
    main()