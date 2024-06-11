# Implement the downstream model for finetuning

import torch
import torch.nn as nn

class DsModel(nn.Module):
    
    def __init__(self, predmodel, num_classes):
        
        super().__init__() 
        
        # predmodel = encoder (CNN) + projector (MLP)
        self.predmodel = predmodel
        self.num_classes = num_classes
        
        for p in self.predmodel.parameters():
            p.requires_grad = True
            
        for p in self.predmodel.projector.parameters():
            p.requires_grad = False
            
        self.last_layer = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Sigmoid())
        
        for p in self.last_layer.parameters():
            p.requires_grad = True

        
    def forward(self, x):
        out = self.predmodel.pretrained(x)
        out = self.last_layer(out)
        
        
        return out
    

class DSModel_Fusion(nn.Module):
    def __init__(self, pred_model, num_classes, fusion=None):
        super().__init__()

        self.predmodel = pred_model
        self.num_classes = num_classes
        self.fusion=fusion

        for p in self.predmodel.parameters():
            p.requires_grad = True

        for p1 in self.predmodel.sensorModelPred.parameters():
            p1.requires_grad = False

        for p2 in self.predmodel.touchModelPred.parameters():
            p2.requires_grad = False

        for p3 in self.predmodel.sensorFusionPred.parameters():
            p3.requires_grad = True

        for p4 in self.predmodel.touchFusionPred.parameters():
            p4.requires_grad = True

        self.fc_embed = nn.Linear(512, 256)

        self.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid())
        

    def forward(self, X_i, X_j):

        y_i = self.predmodel.sensorModelPred(X_i)
        y_j = self.predmodel.touchModelPred(X_j)

        if self.fusion == True:
        
            y_fi = self.predmodel.sensorFusionPred(y_i, y_j)
            y_fj = self.predmodel.touchFusionPred(y_i, y_j)

        else:
            y_fi = self.predmodel.sensorFusionPred(y_i)
            y_fj = self.predmodel.touchFusionPred(y_j)

        x = torch.cat((y_fi, y_fj), dim=1)
        embeddings = self.fc_embed(x)

        x_scores = self.fc(x)

        

        return x_scores, embeddings

