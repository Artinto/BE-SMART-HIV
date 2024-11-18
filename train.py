import torch
import torch.optim as optim
import torch.nn as nn
from model import BCE
from main import data_load


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

    with torch.no_grad():
        for data, target in testloader:
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

    cls_acc_list = []
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        cls_acc_list.append(accuracy)
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    test_loss /= len(testloader.dataset)
    test_accuracy = sum(cls_acc_list) / len(cls_acc_list)
    return test_loss, test_accuracy


classes = ('neg', 'pos')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = BCE().to(device)

criterion = nn.BCELoss().to(device)
optimizer = optim.AdamW(net.parameters(), lr=0.0001)
EPOCHS = 1000

dataroot_train = 'Enter the train data path'
dataroot_valid = 'Enter the valid data path'

trainloader = data_load(dataroot=dataroot_train, batch_size=64, shuffle=True)
validloader = data_load(dataroot=dataroot_valid, batch_size=64, shuffle=False)

best_accuracy = 0
epoch = 0

for epoch in range(1, EPOCHS + 1):
    train(net, trainloader, optimizer, criterion, epoch)
    valid_loss, valid_accuracy = evaluate(net, validloader, criterion)

    if best_accuracy <= valid_accuracy:
        best_accuracy = valid_accuracy
        # path of weights
        PATH = 'best_weight.pth'
        torch.save(net.state_dict(), PATH)

    print('[{}] Valid Loss: {:.4f}, Best Accuracy: {:.2f}%, Accuracy: {:.2f}%'.format(
        epoch, valid_loss, best_accuracy, valid_accuracy))
