import torch
import torch.optim as optim
import torch.nn as nn
from model import BCE
from main import data_load
import numpy as np


def train(net, trainloader, optimizer, criterion, epoch):
    net.train()
    criterion.train()
    train_idx = 0

    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        target = target.type(torch.cuda.LongTensor)
        target = target.squeeze()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_idx += len(data)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, train_idx, len(trainloader.dataset),
                100. * train_idx / len(trainloader.dataset), loss.item()))


def evaluate(net, testloader, criterion):
    net.eval()
    criterion.eval()
    test_loss = 0
    correct = 0

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    total_c = 0
    correct_c = 0

    allFiles, _ = map(list, zip(*testloader.dataset.samples))
    files_result = {}

    with torch.no_grad():
        for i, (data, target) in enumerate(testloader):
            data, target = data.to(device), target.to(device)
            output = net(data)
            target = target.type(torch.cuda.LongTensor)
            target = target.squeeze()

            test_loss += criterion(output, target).item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

            for label, prediction in zip(target, pred):
                label = int(label)
                if label == prediction:
                    correct_pred[classes[label]] += 1
                    correct_c += 1
                total_pred[classes[label]] += 1
                total_c += 1

            for j in range(data.size()[0]):
                files_result['/'.join(allFiles[i*64+j].split('/')[-5:])] = np.argmax(output[j].tolist())

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    test_loss /= len(testloader.dataset)
    test_accuracy = 100 * correct / len(testloader.dataset)
    return test_loss, test_accuracy, files_result


classes = ('neg', 'pos')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = BCE().to(device)
criterion = nn.CrossEntropyLoss().to(device)

dataroot_train = 'Enter the train data path'
dataroot_test = 'Enter the test data path'

testloader = data_load(dataroot_test)

PATH = 'Enter the trained weight path'
net.load_state_dict(torch.load(PATH))

_, test_accuracy, files_result = evaluate(net, testloader, criterion)

print('Accuracy: {:.2f}%'.format(test_accuracy))

for (k,v) in files_result.items():
    print(f'File path: {k},  Result: {v}')
