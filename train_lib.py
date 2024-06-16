
import numpy as np
import os 
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sources import utils_spec, loss, models, MMFusion_Model
import matplotlib.pyplot as plt

import time
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torchmetrics.classification import Accuracy, BinaryAccuracy
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from torchmetrics import ROC


def time_file_str():
  ISOTIMEFORMAT='%Y-%m-%d'
  string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string + '-{}'.format(random.randint(1, 10000))

prefix = time_file_str()


path = os.path.join(os.getcwd(), 'outputs')
if not os.path.exists(path):
    os.makedirs(path)
else:
    print("path exists")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
path = os.path.join(path, timestamp)
if not os.path.isdir(path):
    os.makedirs(path)

log = open(os.path.join(path, 'log_seed_{}.txt'.format(prefix)), 'w')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()


num_classes = 40
accuracy = Accuracy(task = "multiclass", num_classes=num_classes).to(device)
# accuracy = BinaryAccuracy().to(device)



class contrastiveDataset(Dataset):
    
    def __init__(self, X_i, X_j, Y, transform=None):

        self.sensorData = np.array(X_i)
        self.touchData = np.array(X_j)
        
        self.Y = np.array(Y)

        self.transform = transform

    def __getitem__(self, i):
        x_i = self.sensorData[i]
        x_j = self.touchData[i]
        y = self.Y[i]

        x_i = self.transform(x_i)
        x_j = self.transform(x_j)

        return (x_i, x_j, y)
    
    def __len__(self):
        return len(self.sensorData)
    


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    

def testSpec(model, input):
    x = input.to(device)
    y0 = model(x)
    print_log("Output shape of spectrogram model: \n", y0.shape, log)
    print_log(y0, log)
    return y0

def testTouch(model, in_planes):
    x = torch.randn(16, in_planes, 1250)
    x = x.to(device)
    y0 = model(x)
    print_log("output shape of touch model: \n", y0.shape, log)
    print_log(y0, log)
    return y0




def reshape_logs(Logs, y = None, numof_chunks = 8, get_labels = True):
    # numof_chunks : divide data on multu chunks (numof_chunks = 1 -> process the hole csv file)

    Logs_2 = Logs.reshape(Logs.shape[0] * numof_chunks, Logs.shape[1]//numof_chunks, -1)

    if get_labels:
        y_labels = []

        for y0 in y:
            y_labels.append(np.zeros(numof_chunks) + y0)

        y_labels = np.reshape(y_labels, -1)
        return Logs_2,y_labels
    
    else:
        return Logs_2


def compute_spectrogram(data, window_size, overlap, window_fn = tf.signal.hann_window):
    
    x_spec = []

    X = np.transpose(data, axes = [0, -1, 1])

    # Create a Hann window (commonly used for STFT)


    for i, x in enumerate(tqdm(X)):
        x = abs(tf.signal.stft(x, window_size, overlap, window_fn = window_fn))
        x_spec.append(x)
    

    x_spec = np.stack(x_spec)

    #put the number of channels in the last axe to have the same data format supported by TensorFlow (9 channels) 
    x_spec = np.transpose(x_spec, axes = [0, 3, 2, 1])

    return x_spec



def get_models(input_shape):
        sensorFusion = MMFusion_Model.fusionFFN(num_logits=256).to(device)
        touchFusion = MMFusion_Model.fusionFFN(num_logits=256).to(device)

        specModel = MMFusion_Model.SpectrogramModel(input_shape).to(device)
        touchModel = MMFusion_Model.TouchGestureModel(input_size=4, hidden_size=256, seq_length=1250, num_layers=2).to(device)

        fusion_model = MMFusion_Model.pred_model(sensorModelPred=specModel, touchModelPred=touchModel, sensorFusionPred=sensorFusion, touchFusionPred=touchFusion,  fusion=True).to(device)

        return sensorFusion, touchFusion, fusion_model



def test_fusion_pred(fusion_model):
    x1 = torch.randn(16, 3, 33, 1187).to(device)
    x2 = torch.randn(16,4, 1250).to(device)
    y_i, y_j, y_fi, y_fj = fusion_model(x1, x2)
    return y_i, y_j, y_fi, y_fj



def get_data(num_classes, mode, batch_size, window_size, overlap, split = True):
    
    numof_chunks = 8
    Logs, touch_ds, y = utils_spec.dataGenerator(numUsers=num_classes, mode=mode)
    Logs_2, y_labels = reshape_logs(Logs, y, numof_chunks=numof_chunks)

    touch_logs = touch_ds.reshape(touch_ds.shape[0]*numof_chunks, touch_ds.shape[1]//numof_chunks, -1)
    x_spec = compute_spectrogram(Logs_2, window_size, overlap)
    print_log(x_spec.shape, log)

    transform = transforms.Compose([transforms.ToTensor()])
    if split:
        x_train_sensor, x_test_sensor, y_train_sensor, y_test_sensor = train_test_split(x_spec, y_labels, test_size=0.2, shuffle=True, random_state=10)
        x_train_touch, x_test_touch, y_train_touch, y_test_touch = train_test_split(touch_logs, y_labels, test_size=0.2, shuffle=True, random_state=10)

        print_log("Sensor : Train Size {}".format(x_train_sensor.shape), log)
        print_log("Sensor : Test Size {}".format(x_test_sensor.shape), log)
        print_log("Touch : Train Size {}".format(x_train_touch.shape), log)
        print_log("Touch : Test Size {}".format(x_test_touch.shape), log)
        
        train_set = contrastiveDataset(X_i=x_train_sensor, X_j=x_train_touch, Y = y_train_sensor, transform=transform)
        test_set = contrastiveDataset(X_i=x_test_sensor, X_j = x_test_touch, Y = y_train_touch, transform=transform)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
        return train_loader, test_loader
    else:
        data = contrastiveDataset(X_i = x_spec, X_j = touch_logs, Y = y_labels, transform=transform)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True)

        return data_loader
    
    

    


'''
Self Supervised pre-training with fusion model
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def contrastive_train_loop(model, train_loader, optimizer, criterion, epoch):

    model.train()
    train_loss = 0.0
        
    for step, (x_i, x_j, y) in enumerate(train_loader):
        
        optimizer.zero_grad()
        x_i = x_i.squeeze().to(device).float()
        x_j = x_j.squeeze().to(device).float()
                
        z_i, z_j, z_fi, z_fj = model(x_i, x_j)
        
        loss1 = criterion(z_i, z_j)
        loss2 = criterion(z_fi, z_fj)

        loss = loss1 + loss2
        
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item() * x_i.size(0)


    total = len(train_loader.dataset)
    train_loss = train_loss / total

    print_log(
        "EPOCH: {}\ttrain_loss: {:.3f}".format(epoch, train_loss),
        log
        )  
         
    return train_loss


def contrastive_val_loop(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for step, (x_i, x_j, y) in enumerate(val_loader):
            
            x_i = x_i.squeeze().to(device).float()
            x_j = x_j.squeeze().to(device).float()
            
            z_i, z_j, z_fi, z_fj = model(x_i, x_j)

            loss1 = criterion(z_i, z_j)
            
            loss2 = criterion(z_fi, z_fj)

            loss = loss1 + loss2
            
            val_loss += loss.item() * x_i.size(0)

        total = len(val_loader.dataset)
        val_loss = val_loss / total
        
        print_log(
        "Validation: \tval_loss: {:.3f}".format(val_loss),
        log
        )  

        return val_loss
    


def compute_accuracy(output, target):

    # Convert output logits to predicted labels (assuming logits)
    _, predicted = torch.argmax(output, 1)
    
    # Compute the number of correct predictions
    correct = (predicted == target).sum().item()
    
    # Compute accuracy
    accuracy = correct / target.size(0)
    
    return accuracy


def finetuning_loop(model, train_loader, optimizer, criterion, epoch):
    
    model.train()
    train_loss = 0.0
    acc = 0.0

    for batch_idx, (x_i, x_j, y) in enumerate(train_loader):

        x_i , x_j = x_i.to(device=device, dtype=torch.float32), x_j.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype = torch.int64)

        optimizer.zero_grad()
        
        outputs,_ = model(x_i, x_j)
        
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        

        train_loss += loss.item() * outputs.size(0)

        
        # _, outputs = torch.max(outputs, 1)
        # outputs = outputs > 0.5

        # acc += compute_accuracy(outputs, y)
        acc += accuracy(outputs, y).item() * outputs.size(0)

    total = len(train_loader.dataset)
    train_acc = acc / total
    train_loss = train_loss / total

    print_log(
        "EPOCH: {}\ttrain_acc: {:.3f}\ttrain_loss: {:.3f}".format(epoch, train_acc, train_loss),
        log
        )  
    
    return train_acc, train_loss


def finetuning_val_loop(model, val_loader, criterion):

    model.eval()

    val_loss = 0.0
    acc = 0.0

    with torch.no_grad():
        for batch_idx, (x_i, x_j, y) in enumerate(val_loader):

            x_i , x_j = x_i.to(device=device, dtype=torch.float32), x_j.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype = torch.int64)
            
            outputs,_ = model(x_i, x_j)

            loss = criterion(outputs, y)

            val_loss += loss.item() * outputs.size(0)

            # outputs = outputs > 0.5
            acc += accuracy(outputs, y).item() * outputs.size(0)
            # acc += compute_accuracy(outputs, y)

    total = len(val_loader.dataset)      
    val_acc = acc / total
    val_loss = val_loss / total

    print_log(
    "Validation: \tval_acc: {:.3f}\tval_loss: {:.3f}".format(val_acc, val_loss),
    log
    )  

    return val_acc, val_loss
        


def contrastive_learning(model, train_loader, val_loader, batch_size, loss, epochs, checkpoint):

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.1)
    criterion = loss.Contrastive_Loss(batch_size=batch_size, temperature=0.5)

    best_val_loss = torch.inf

    model_history = {'train_loss':[], 'val_loss':[]}
    
    for epoch in range(epochs):

        train_loss = contrastive_train_loop(model, train_loader, optimizer, criterion, epoch)
        val_loss = contrastive_val_loop(model, val_loader, criterion)


        model_history['train_loss'].append(train_loss)
        model_history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            
            print_log("Saving the best checkpoint : loss imporved from {:.3f} to {:.3f}".format(best_val_loss, val_loss), log)
            torch.save(model.state_dict(), checkpoint)
            best_val_loss = val_loss

    # Loading the best checkpoint
    print_log("="*150, log)
    print_log("End of Contrastive learning process. Loading the best chechkpoint", log)
    model.load_state_dict(torch.load(checkpoint))
    print_log("Best chechkpoint successfully loaded from {}".format(checkpoint), log)
    
    return model, model_history



def finetune(model, train_loader, val_loader, epochs, checkpoint):
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.1)
    criterion = nn.CrossEntropyLoss().cuda()

    best_acc = 0

    model_history = {'train_acc': [], 'train_loss':[], 'val_acc': [], 'val_loss':[]}
    for epoch in range(epochs):

        train_acc, train_loss = finetuning_loop(model, train_loader, optimizer, criterion, epoch)
        val_acc, val_loss = finetuning_val_loop(model, val_loader, criterion)

        model_history['train_acc'].append(train_acc)
        model_history['train_loss'].append(train_loss)

        model_history['val_acc'].append(val_acc)
        model_history['val_loss'].append(val_loss)

        if best_acc < val_acc:
            
            print_log("Saving the best checkpoint : Accuracy imporved from {:.3f} to {:.3f}".format(best_acc, val_acc), log)
            torch.save(model.state_dict(), checkpoint)
            best_acc = val_acc

    # Loading the best checkpoint
    print_log("="*150, log)
    print_log("End of FineTuning process. Loading the best chechkpoint", log)
    model.load_state_dict(torch.load(checkpoint))
    print_log("Best chechkpoint successfully loaded from {}".format(checkpoint), log)

    return model, model_history


def evaluation(model, data_loader, save_plots_to):

    target = []
    embedding = []
    probas = []
    predicted = []

    model.eval()

    with torch.no_grad():
        for i, (x1, x2, y) in enumerate(data_loader):

            labels = y.clone().detach()
            x_i, x_j = x1.to(device, dtype=torch.int64), x2.to(device, dtype=torch.int64)
            labels = labels.to(device, dtype=torch.int64)

            x_i = x_i.float()
            x_j = x_j.float()

            scores, emb = model(x_i, x_j)
            p_target = (scores > 0.5).long()
            
            predicted.extend(p_target.detach().cpu().tolist())
            embedding.extend(emb.detach().cpu())
            target.extend(labels.detach().cpu().tolist())
            probas.extend(scores.detach().cpu().tolist())

    embedding = np.vstack(embedding)
    target = np.array(target)
    probas = np.array(probas)
    predicted = np.array(predicted)
    
    predicted = np.array(predicted).squeeze()

    roc = ROC(task='binary')

    probas = (probas - probas.mean())/probas.std()

    fpr_nn, tpr_nn, th_nn = roc(torch.tensor(probas.ravel()), torch.tensor(target))
    

    plt.figure(figsize = (6, 6))
    cm = confusion_matrix(target, predicted)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    cm_display.plot()
    plt.savefig(save_plots_to)


    return fpr_nn, tpr_nn, th_nn


def plot_values(model_dict, title = None, plot_accuracy = False, save_to = None):
    x_axis = range(0,len(model_dict["train_loss"]))
    plt.figure(figsize = (8, 4))
    if title is not None:
        plt.title(title)
    if plot_accuracy:
        plt.subplot(2, 1, 1)
        plt.plot(x_axis, model_dict['train_acc'], '-', label='Training')
        plt.plot(x_axis, model_dict['val_acc'], '-', label='Validation', color='r')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()

        plt.subplot(2, 1, 2)

    plt.plot(x_axis, model_dict['train_loss'], '-', label='Training')
    plt.plot(x_axis, model_dict['val_loss'], '-', label='Validation', color='r')
    plt.ylabel('Loss' if plot_accuracy else "Contrastive Loss")

    plt.legend()
    plt.grid()
    if save_to is not None:
        plt.savefig(os.path.join(save_to, title+".png"))
    else:
        plt.show()



def evaluate(ft_model, data_loader, save_to):    
    fpr_nn, tpr_nn, th_nn = evaluation(ft_model, data_loader, save_to)
    return fpr_nn, tpr_nn, th_nn



def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()
