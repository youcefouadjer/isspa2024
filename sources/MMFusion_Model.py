import numpy as np
import torch 
import torch.nn as nn 

'''
Input to RNN need to have shape: (batch, seq_length, input_features)

'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ConvRNN(nn.Module):

    def __init__(self, input_size, hidden_size, seq_length, num_layers, batch_size, num_classes):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes = num_classes

        # input shape for RNN: (128, 3, 64) 

        # (N, input_fatures, seq_length)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU())
        
        self.maxpool = nn.MaxPool1d(1, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU())
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.dropout = nn.Dropout(0.5)
        
        self.RNN = nn.RNN(input_size=128, hidden_size=256, num_layers=2, nonlinearity='tanh', batch_first=True, dropout=0.5)

        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        x = x.reshape(self.batch_size, self.input_size, self.seq_length)
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)

        x = x.reshape(self.batch_size, 16, 128) 
        # 128: Final output from the Conv3 layer

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, hn = self.RNN(x, h0)
        out_logit = out[:, -1, :]
        out_features = self.fc(out_logit)
        return out_logit
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TouchGestureModel(nn.Module):

    def __init__(self, input_size, hidden_size, seq_length, num_layers, batch_size, num_classes):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes = num_classes

        # input shape for RNN: (128, 3, 64) 

        # (N, input_fatures, seq_length)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU())
        
        self.maxpool = nn.MaxPool1d(1, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU())
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.dropout = nn.Dropout(0.5)
        
        self.RNN = nn.RNN(input_size=128, hidden_size=256, num_layers=2, nonlinearity='tanh', batch_first=True, dropout=0.5)

        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        x = x.reshape(self.batch_size, self.input_size, self.seq_length)
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)

        x = x.reshape(self.batch_size, 77, 128) 
        # 128: Final output from the Conv3 layer

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, hn = self.RNN(x, h0)
        out_logit = out[:, -1, :]
        out_features = self.fc(out_logit)
        return out_logit
    

class SpectrogramModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()
        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU())
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU())
        
        #self.maxpool2d = nn.MaxPool2d(kernel_size=1, stride=2)
        
        self.fc = nn.Linear(4608, 256)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #x = self.maxpool2d(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


class fusionFFN(nn.Module):
    def __init__(self, num_logits):

        super().__init__()
        self.layer1 = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU())
        self.layer2 = nn.Linear(512, num_logits)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class vanillaFFN(nn.Module):
    def __init__(self, num_logits):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.layer2 = nn.Linear(256, num_logits)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x

class pred_model(nn.Module):
    
    def __init__(self, sensorModelPred, touchModelPred, sensorFusionPred, touchFusionPred, fusion=None):
        super().__init__()

        self.sensorModelPred = sensorModelPred
        self.touchModelPred = touchModelPred
        self.sensorFusionPred = sensorFusionPred
        self.touchFusionPred = touchFusionPred
        self.fusion = fusion


        for p1 in self.sensorModelPred.parameters():
            p1.requires_grad = True

        for p2 in self.touchModelPred.parameters():
            p2.requires_grad = True

        for p3 in self.sensorFusionPred.parameters():
            p3.requires_grad = True

        for p4 in self.touchFusionPred.parameters():
            p4.requires_grad = True
        
    def forward(self, x_i, x_j):
        
        y_i = self.sensorModelPred(x_i)
        y_j = self.touchModelPred(x_j)

        if self.fusion == True:
            
            y_fi = self.sensorFusionPred(y_i, y_j)
            y_fj = self.touchFusionPred(y_i, y_j)

        else:
            y_fi = self.sensorFusionPred(y_i)
            y_fj = self.touchFusionPred(y_j)

        return y_i, y_j , y_fi, y_fj
    
    
