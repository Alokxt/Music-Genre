import torch 
import torch.nn as nn 
from sklearn.metrics import f1_score 
import numpy as np 
from models import get_cnn

import os
from dotenv import load_dotenv

load_dotenv()

import wandb
wandb.login()  

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            outputs = model(x)
            loss = criterion(outputs, y)
            

            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            correct += (preds == y).sum().item()
            total += y.size(0)
    f1_macro = f1_score(all_labels,all_preds,average="macro")
    f1_weighted = f1_score(all_labels,all_preds, average="weighted")
    return total_loss / total, correct / total , f1_macro ,f1_weighted 

model = get_cnn()
"""num_epochs = 30
best_val_acc = 0
run = wandb.init(
    project="23f2001025-t12026",       
    entity="23f2001025-indian-institue-of-technology-madras",  
    name="CNN-1",       
    config={
        "model": "CNN",
        "Optim":"Adam",
        "epochs": 30,
        "batch_size": 16,
        "lr": 1e-3,
        "n_mels": 128,
        "augmentation": "No",
    }
)
for epoch in range(num_epochs):

    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )

    val_loss, val_acc,f1_macro,f1_weighted = validate(
        model, val_loader, criterion, device
    )

    

    scheduler.step(val_acc)

   

    wandb.log({
        "epoch": epoch,
    
        
        "train_loss": train_loss,
        "val_loss": val_loss,
    
        
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
    
        
        "val_f1_macro": f1_macro,
        "val_f1_weighted": f1_weighted,
    
        
        "train_lr": optimizer.param_groups[0]['lr'],
    })

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    print("-" * 40)

   

torch.save(model.state_dict(), "cnn-1.pth")
artifact = wandb.Artifact(name="model-CNN-1", type="model")
artifact.add_file("cnn-1.pth")
run.log_artifact(artifact)
wandb.finish()"""

