import torch
import random
import time
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader

NUM_EPOCHS = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def create_pairs(X):  # X is a list containing 40 lists with 10 imgs for each
    pairs = []
    class_num = len(X)
    for i in range(class_num):
        # append all positive pairs in one class
        for x1 in X[i]:
            for x2 in X[i]:
                pairs.append([x1, x2, 0])
                random_class = (
                    i + random.randint(1, class_num - 1)) % class_num
                x3 = random.choice(X[random_class])
                pairs.append([x1, x3, 1])
    return pairs


class ImageDataset(Dataset):
    def __init__(self, X):
        self.pairs = create_pairs(X)

    def __getitem__(self, index):
        x1 = torch.Tensor(self.pairs[index][0]).unsqueeze(0)
        x2 = torch.Tensor(self.pairs[index][1]).unsqueeze(0)
        y = torch.Tensor([self.pairs[index][2]])
        return x1, x2, y

    def __len__(self):
        return len(self.pairs)


class SiameseNetwork(nn.Module):
    def __init__(self,shape=(64,64)):
        super(SiameseNetwork, self).__init__()
        self.shape = shape
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * self.shape[0] * self.shape[1], 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 20),
        )

        self.optimizer = torch.optim.Adam(self.parameters())
        self.lr_adaptor = torch.optim.lr_scheduler.StepLR(self.optimizer, 5)
        self.criterion = nn.MSELoss()

    def forward_once(self, x):
        if isinstance(x, np.ndarray) and x.shape[0] == self.shape[0] * self.shape[1]:
            x = torch.Tensor(x.reshape(1, self.shape[0], self.shape[1])).unsqueeze(0)
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    def distance(self, input1, input2):
        output1, output2 = self.forward(input1, input2)
        d = torch.nn.functional.pairwise_distance(output1, output2)
        return d

    def fit(self, dataloader):
        stop_line = 0.5
        combo = 0
        since = time.time()
        for epoch in range(NUM_EPOCHS):
            batch = 0
            self.lr_adaptor.step()
            for batch_x1, batch_x2, batch_y in dataloader:
                bx1 = batch_x1.to(device)
                bx2 = batch_x2.to(device)
                by = batch_y.to(device)

                self.optimizer.zero_grad()
                by_pred = self.distance(bx1, bx2).squeeze()
                by = by.squeeze() * 10
                loss = self.criterion(by_pred, by)

                if loss < stop_line:
                    combo += 1
                    if combo == 5:
                        return
                else:
                    combo = 0

                loss.backward()
                self.optimizer.step()
                batch += 1
                print('Epoch:{:02d}/{}, Batch:{:02d}, Loss:{:.4f}'.format(
                    epoch + 1, NUM_EPOCHS, batch, loss))
        print('Training completed in {:.2f}s'.format(time.time() - since))




def siamese_network(X,shape):
    if not X:
        X = [[torch.rand(64, 64) for i in range(500)] for j in range(2)]
    dataset = ImageDataset(X)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=4)
    print('data loaded...')
    mlnet = SiameseNetwork(shape).to(device)
    print('model loaded...')
    print('start training...')
    mlnet.fit(dataloader)
    return mlnet.distance

def nd(x1,x2):
    l1 = x1.shape[0]
    x1 = np.concatenate((x1,np.zeros((l1,1))),axis=1).reshape((l1,32,24))
    l2 = x2.shape[0]
    x2 = np.concatenate((x2,np.zeros((l2,1))),axis=1).reshape((l2,32,24))
    res = []
    for x in x1:
        x = np.array([x] * l2)

        input1 = torch.Tensor(x).unsqueeze(1).cuda()
        input2 = torch.Tensor(x2).unsqueeze(1).cuda()
        _d = d(input1,input2).data.cpu().numpy()
        res.append(_d)

    # print (l1,l,np.array(res).shape)
    return np.array(res)


if __name__ == '__main__':
    main()