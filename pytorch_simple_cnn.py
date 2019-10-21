import torch
import torchvision
import torchvision.transforms as transforms
import models.models as model
import torch.nn as nn
import torch.optim as optim
import os

DATA_DIR = os.path.join(os.path.dirname(__file__),'data/flowers')

transform = transforms.Compose(
    [
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

trainset = torchvision.datasets.ImageFolder(root='data/flowers',transform=transform)
dataset_loader = torch.utils.data.DataLoader(trainset,
                                             batch_size=1, shuffle=True,
                                             num_workers=4)


net = model.SimpleCNN2()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataset_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
