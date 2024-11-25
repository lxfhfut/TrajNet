import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
from time import gmtime, strftime
from model import VideoClassifier
from visualize import TrainingVisualizer
from dataloader import TrajPointDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """
    Training loop with improved state management and monitoring
    """
    criterion = nn.BCELoss()

    config = {
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'momentum': 0.9,
        'weight_decay': 1e-3,
        'lr_decay_factor': 0.5,
        'lr_decay_epochs': list(range(0, num_epochs, 30)),
    }

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )

    visualizer = TrainingVisualizer(num_epochs)
    best_val_accuracy = 0
    best_model_state = None

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        all_train_preds = []
        all_train_labels = []

        for batch in train_loader:
            padded_trajectories, traj_lengths, video_lengths, batch_labels, video_ids = batch
            optimizer.zero_grad()

            outputs = model((padded_trajectories, traj_lengths, video_lengths))
            loss = criterion(outputs.squeeze(), batch_labels.float())

            loss.backward()
            # Add gradient clipping to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = (outputs.squeeze() >= 0.5).float()
            all_train_preds.extend(preds.cpu().numpy())
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
        model.eval()
        val_metrics = evaluate_model(model, val_loader, criterion)

        # Learning rate adjustment
        if epoch in config['lr_decay_epochs']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= config['lr_decay_factor']

        # Update visualization
        visualizer.update(train_metrics['loss'], val_metrics['loss'],
                          train_metrics['accuracy'], val_metrics['accuracy'])

        # Model saving logic
        if val_metrics['accuracy'] >= best_val_accuracy:
            print(f"Epoch {epoch}: New best validation accuracy: {val_metrics['accuracy']:.4f} "
                  f"(previous: {best_val_accuracy:.4f})")
            best_val_accuracy = val_metrics['accuracy']
            # Create a deep copy of the model state
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'train_accuracy': train_metrics['accuracy']
            }

        # Print detailed epoch statistics
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print("-" * 60)

    # Save the visualization
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    visualizer.save(f'training_results_{time_stamp}.png')

    # Restore best model and save it
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        save_path = f'best_model_{time_stamp}.pt'
        torch.save(best_model_state, save_path)
        print(f"Best model saved to {save_path}")
        print(f"Best validation accuracy: {best_model_state['val_accuracy']:.4f} "
              f"achieved at epoch {best_model_state['epoch']}")

    return model


def test_model(model, dataloader):
    """
    Enhanced test function with more detailed error analysis
    """
    model.eval()
    all_preds = []
    all_labels = []
    wrong_predictions = []

    with torch.no_grad():
        for padded_trajectories, traj_lengths, video_lengths, batch_labels, video_ids in dataloader:
            outputs = model((padded_trajectories, traj_lengths, video_lengths))
            preds = (outputs.squeeze() >= 0.5).float()

            # Store predictions and labels
            batch_preds = preds.cpu().numpy()
            batch_labels = batch_labels.cpu().numpy()
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)

            # Track wrong predictions
            wrong_idx = np.where(np.abs(batch_preds - batch_labels) > 0.5)[0]
            for idx in wrong_idx:
                wrong_predictions.append({
                    'video_id': video_ids[idx],
                    'predicted': batch_preds[idx],
                    'ground_truth': batch_labels[idx],
                    'confidence': outputs.squeeze()[idx].item()
                })

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }

    # Print detailed analysis
    print("\nTest Results:")
    print("-" * 60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("\nWrong Predictions Analysis:")
    print("-" * 60)
    for pred in wrong_predictions:
        print(f"Video ID: {pred['video_id']}")
        print(f"Predicted: {pred['predicted']:.4f} (confidence: {pred['confidence']:.4f})")
        print(f"Ground Truth: {pred['ground_truth']}")
        print("-" * 30)

    return metrics, wrong_predictions


if __name__ == "__main__":

    data_dir = "/Users/lxfhfut/Dropbox/Garvan/CBVCC/tif/cytotorch_0"
    anno_dir = "/Users/lxfhfut/Dropbox/Garvan/CBVCC/dataset/"

    train_dataset = TrajPointDataset(data_dir, anno_dir, split="training", augment=True)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=train_dataset.collate_fn)

    val_dataset = TrajPointDataset(data_dir, anno_dir, split="testing")
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=val_dataset.collate_fn)

    model = VideoClassifier(n_features=32, max_sequence_length=20, sample_attention="single")
    best_model = train_model(model, train_dataloader, val_dataloader, num_epochs=1000, learning_rate=0.01)
    # time_stamp = strftime('%Y-%m-%d-%H-%M-%S', gmtime())
    # torch.save(best_model.state_dict(), f"/Users/lxfhfut/Dropbox/Garvan/CBVCC/ckpts/Complete_Model_{time_stamp}.pth")
    results = test_model(best_model, val_dataloader)
