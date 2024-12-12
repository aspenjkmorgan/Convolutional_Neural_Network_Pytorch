import torch.optim as optim
import torch
from torch import nn

class dl_assignment4_helper_functions:
   def __init__(self):
      pass
   
   def trainMyModel(self, net, lr, trainloader, n_epochs=2):
        optimizer = optim.Adam(net.parameters(), lr)
        criterion = nn.CrossEntropyLoss()

        # Attempt to put your neural network onto the GPU
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net.to(device)

        for epoch in range(n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                # inputs is one batch of 128 images, each [1, 28, 28]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Don't forget, your function must print out the training loss on each
                # 100th mini-batch
                running_loss += loss.item()
                if i % 100 == 99:    # print every 100 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        print('Finished Training')
        return net
   
   def testMyModel(self, trainedNet, testloader):
        correct = 0
        total = 0

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data

                # calculate outputs by running images through the network
                outputs = trainedNet(images)

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = round((100 * correct / total), 2)
        return acc