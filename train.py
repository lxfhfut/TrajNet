import os
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from datetime import datetime
from time import gmtime, strftime
from model import VideoClassifier
from utils import TrainingVisualizer
from dataloader import TrajPointDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def test_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for padded_trajectories, traj_lengths, video_lengths, batch_labels, video_ids in dataloader:
            outputs = model((padded_trajectories, traj_lengths, video_lengths))

            # Convert outputs to predictions
            outputs = torch.atleast_1d(outputs.squeeze())
            preds = (outputs >= 0.5).float()
            batch_preds = preds.cpu().numpy()
            batch_labels = batch_labels.cpu().numpy()
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
            wrong_idx = np.where(np.abs(batch_preds - batch_labels) > 0.5)[0]
            for idx in wrong_idx:
                print(f"Video ID: {video_ids[idx]}")
                print(f"Predicted: {batch_preds[idx]:.0f} (confidence: {outputs[idx].item():.3f})")
                print(f"Ground Truth: {batch_labels[idx]}")
                print("-" * 30)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }

    return metrics


def evaluate_model(model, dataloader, criterion):
    """
    Evaluate the model on the given dataloader
    Returns loss and predictions for metric calculation
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for padded_trajectories, traj_lengths, video_lengths, batch_labels, video_ids in dataloader:
            outputs = model((padded_trajectories, traj_lengths, video_lengths))
            loss = criterion(outputs.squeeze(), batch_labels.float())
            total_loss += loss.item()
            
            # Convert outputs to predictions
            preds = (outputs.squeeze() >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }
    
    return metrics


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, save_dir=None):
    """
    Training loop for the trajectory feature extractor
    
    Args:
        model: The VariableLengthTrajectoryExtractor model
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
    """
    out_dir = "./ckpts/" if save_dir is None else save_dir
    criterion = nn.BCELoss()
    
    # Initialize optimizer
    config = {
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'momentum': 0.9,
            'weight_decay': 1e-3,
            'lr_decay_factor': 0.5,
            'lr_decay_epochs': list(range(0, num_epochs, 30)),  # epochs at which to decay learning rate
        }
    
    # Initialize optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, learning_rate,
        epochs=num_epochs, steps_per_epoch=len(train_loader))
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.05, mode='exp_range')
    visualizer = TrainingVisualizer(num_epochs)
    best_accuracy = 0
    best_model_state = None

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        all_train_preds = []
        all_train_labels = []
        
        for padded_trajectories, traj_lengths, video_lengths, batch_labels, video_ids in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model((padded_trajectories, traj_lengths, video_lengths))
            loss = criterion(outputs.squeeze(dim=-1), batch_labels.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Collect predictions for metrics
            preds = (outputs.squeeze() >= 0.5).float()
            all_train_preds.extend(np.atleast_1d(preds.cpu().numpy()))
            all_train_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate training metrics
        train_metrics = {
            'loss': train_loss / len(train_loader),
            'accuracy': accuracy_score(all_train_labels, all_train_preds),
            'precision': precision_score(all_train_labels, all_train_preds, zero_division=0),
            'recall': recall_score(all_train_labels, all_train_preds, zero_division=0),
            'f1': f1_score(all_train_labels, all_train_preds, zero_division=0)
        }
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, criterion)

        # scheduler.step(val_metrics['loss'])
        scheduler.step()
        
        # Update visualization
        visualizer.update(train_metrics['loss'], val_metrics['loss'],
                          train_metrics['accuracy'], val_metrics['accuracy'])
        
        if val_metrics['accuracy'] >= best_accuracy:
            print(f"Best accuracy: {val_metrics['accuracy']:.3f}")
            best_accuracy = val_metrics['accuracy']
            # best_model_state = model.state_dict().copy()
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'train_accuracy': train_metrics['accuracy']
            }
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    visualizer.save(os.path.join(out_dir, f'training_results_{time_stamp}.png'))
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        # model.load_state_dict(best_model_state)
        torch.save({
            'model_state_dict': model.state_dict(),
        }, os.path.join(out_dir, f'best_model_{time_stamp}.pt'))

    return model


if __name__ == "__main__":

    data_dir = "/Users/lxfhfut/Dropbox/Garvan/CBVCC/tif/cytotorch_0"
    anno_dir = "/Users/lxfhfut/Dropbox/Garvan/CBVCC/dataset/"

    train_dataset = TrajPointDataset(data_dir, split="training", augment=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_dataset.collate_fn)

    val_dataset = TrajPointDataset(data_dir, split="testing")
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=val_dataset.collate_fn)

    model = VideoClassifier(n_features=64, max_sequence_length=20, sample_attention="single")
    best_model = train_model(model, train_dataloader, val_dataloader, num_epochs=1000, learning_rate=0.005)

    # best_model = VideoClassifier(n_features=32, max_sequence_length=20, sample_attention="single")
    # best_model.load_state_dict(torch.load("ckpts/best_model_20241128_144647tick.pt")["model_state_dict"])
    results = test_model(best_model, train_dataloader)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    results = test_model(best_model, val_dataloader)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


