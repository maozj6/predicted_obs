import datetime

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from Seq2Seq import Seq2Seq
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import save_image

import io
import imageio
from ipywidgets import widgets, HBox
from IPython.core.display import display

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 800)
        self.fc2 = nn.Linear(800, 120)
        self.fc3 = nn.Linear(120, 2)

    def forward(self, x):
        input_size = x.size(0)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(input_size,-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VAE Trainer')
    parser.add_argument('--evpath', type=str, default="saved_model/grey100Pre0_CNN_20.pth", metavar='N',
                        help='path of evaluator')
    parser.add_argument('--dataset', type=str, default='dataset/20grey_test_series.npz', metavar='N',
                        help='path of dataset')
    parser.add_argument('--rnnpath', type=str, default='saved_model/300RCNN350.pth', help='Directory where results are logged')
    parser.add_argument('--savepath', type=str, default='result/',
                        help='Best model is not reloaded if specified')
    # parser.add_argument('--nosamples', action='store_true',
    #                     help='Does not save samples during training if specified')

    # rnnOut=np.load("RNNout.npz")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TP=rnnOut['TP']
    # FN=rnnOut['FN']
    #
    # TN=rnnOut['TN']
    # FP=rnnOut['FP']
    #
    # acc=rnnOut['acc']
    # acc=acc/4650
    print(device)

    TP = np.zeros(shape=(10))
    FN = np.zeros(shape=(10))
    TN = np.zeros(shape=(10))
    FP = np.zeros(shape=(10))
    acc=np.zeros(shape=(10))

    # test0=np.load("/home/mao/23Spring/cars/racing_car_data/0224_data_single_model/test/20grey_test_series.npz")['series']
    # # train0=np.load("/home/mao/23Spring/cars/racing_car_data/0224_data_single_model/test/20grey_test_series.npz")['series']
    # tag=np.load("/home/mao/23Spring/cars/racing_car_data/0224_data_single_model/test/20grey_test_series.npz")['safe']
    test0=np.load(args.dataset)['series']
    tag=np.load(args.dataset)['safe']
    train_data = test0
    val_data = test0
    test_data = test0

    datas=[test0,tag]


    model = Seq2Seq(num_channels=1, num_kernels=64,
                    kernel_size=(3, 3), padding=(1, 1), activation="relu",
                    frame_size=(64, 64), num_layers=3).to(device)
    model.load_state_dict(torch.load(args.rnnpath))
    model=model.to(device)
    evaluator=Net()
    evaluator=torch.load(args.evpath)
    evaluator=evaluator.to(device)
    for i in range(len(test0)):

        print(str(i)+'/'+str(len(test0)))


        if i%1000==0:
            print(TP)
            print(FN)
            print(TN)
            print(FP)
            print(acc)
            print("-----")
        for j in range(10):
            if j==0:
                input=test0[i][j:10]

                input = torch.tensor(input).unsqueeze(0).unsqueeze(0)

                input = input / 255.0


                save_image(input.squeeze(0).view(10, 1, 64, 64),
                           (args.savepath + str(i) + 'ori.png'))
                input = input.to(device)
            output = model(input)




            outt = evaluator(output)
            _, predicted = torch.max(outt.data, 1)

            if tag[i][j+10] == 1:
                if predicted.data == 1:
                    TP[j] = TP[j] + 1
                    acc[j]=acc[j]+1
                else:
                    FN[j] = FN[j] + 1
            else:
                if predicted.data == 0:
                    TN[j] = TN[j] + 1
                    acc[j]=acc[j]+1

                else:
                    FP[j] = FP[j] + 1
            input = input[:, :, 1:10]
            input = torch.cat((input, output.reshape(1, 1, 1, 64, 64)), 2)  # 在 0 维(纵向)进行拼接

            if j==9:
                save_image(input.view(10, 1, 64, 64),
                       ( args.savepath+ str(i) + 'pre.png'))
                # j0=output


    np.savez_compressed("RNNout2.npz", TP=TP, FN=FN,TN=TN,FP=FP,acc=acc)
    print("end")


