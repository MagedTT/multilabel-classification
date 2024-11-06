import time
import numpy as np

import torch
import torchvision
import torch.nn.functional as F

def train_model(model, num_epochs, optimizer, train_loader, val_loader, device, loss_fn=None, threshold=None):
    if loss_fn is None:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    if threshold is None:
        thresold = 0.5

    train_loss = []
    total_train_loss = []
    train_acc = []
    total_train_acc = []

    val_loss = []
    total_val_loss = []
    val_acc = []
    total_val_acc = []
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images)

            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            probabilities = F.sigmoid(outputs)
            predicted_labels = (probabilities >= threshold).float()
            train_acc.append((predicted_labels == labels).sum().item() / labels.numel())
            
            train_loss.append(loss.detach().cpu().item())

            if not batch_idx % 200:
                print(f'EPOCH: {epoch + 1:03d}/{num_epochs:03d} | '
                      f'Batch: {batch_idx:03d}/{len(train_loader):03d} | '
                      f'Loss: {loss.item():.4f}')

        total_train_acc.append(np.mean(train_acc))
        total_train_loss.append(np.mean(train_loss))
        
        print(f'total_train_acc: {(total_train_acc[-1] * 100):.4f} | '
              f'total_train_loss: {total_train_loss[-1]:.4f}')
        # scheduler.step()
        
        model.eval()
        # with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images)

            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            probabilities = F.sigmoid(outputs)
            predicted_labels = (probabilities >= threshold).float()
            val_acc.append((predicted_labels == labels).sum().item() / labels.numel())
            
            val_loss.append(loss.detach().cpu().item())

        total_val_acc.append(np.mean(val_acc))
        total_val_loss.append(np.mean(val_loss))
        
        print(f'total_val_acc: {(total_val_acc[-1] * 100):.4f} | '
              f'total_val_loss: {total_val_loss[-1]:.4f}')
        print(f'Elapsed Time: {(time.time() - start_time) / 60} min')
        
    print(f'Total Training Time: {(time.time() - start_time) / 60} min')
    
    return [total_train_acc, total_train_loss, total_val_acc, total_val_loss]