# coding=gbk
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from model import ResNet18

class model_train:
    def __init__(self, EPOCH = 135, pre_epoch = 0, BATCH_SIZE = 128, LR = 0.001):
        # init some parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument('--net', default='./model/best_model.pth', help="path to net (to continue training)")
        parser.add_argument('--acc', default=0, help="last best accuracy")
        self.args = parser.parse_args()
        self.EPOCH = EPOCH
        self.pre_epoch = pre_epoch
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.model = ResNet18().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.LR, momentum=0.9, weight_decay=5e-4)

    def reload_model(self):
        # reload the trained model before pruning
        self.model = ResNet18().to(self.device)
        self.model.load_state_dict(torch.load('./model/best_model.pth'))

    def loaddataset(self):

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    def train(self):
        best_acc = self.args.acc
        for epoch in range(self.pre_epoch, self.EPOCH):
            print('\nEpoch: %d' % (epoch + 1))
            self.model.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, data in enumerate(self.trainloader, 0):
                length = len(self.trainloader)
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # forward + backward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

            print("Waiting Test!")
            with torch.no_grad():
                correct = 0
                total = 0
                for data in self.testloader:
                    self.model.eval()
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                print('test_acc: %.3f%%' % (100 * correct / total))
                acc = 100. * correct / total
                if acc > best_acc:
                    best_acc = acc
                    print('Saving model...')
                    torch.save(self.model.state_dict(), self.args.net)

if __name__ == "__main__":
    print("Start Training, Resnet-18!")
    model = model_train()
    model.loaddataset()
    model.train()
    print("Training Finished, TotalEPOCH=%d" % model.EPOCH)