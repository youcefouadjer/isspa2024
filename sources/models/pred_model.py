import torch
import torch.nn as nn

class Identity(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

 

class Pred_Model(nn.Module): 
    
    def __init__(self, model, hidden_features, out_features):
        super().__init__()
        
        self.hidden_features = hidden_features
        self.out_features = out_features
        
        self.pretrained = model
        internal_embedding = 256
        self.pretrained.fc = Identity()
        
        
        for p in self.pretrained.parameters():
            p.requires_grad = True
            
        self.projector = nn.Sequential(
            nn.Linear(internal_embedding, out_features=self.hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_features, out_features=self.out_features)
        )
        self.logits = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        out = self.pretrained(x)
        
        x_p = self.projector(torch.squeeze(out))
        
       
        return x_p
