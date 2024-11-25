import matplotlib.pyplot as plt


class TrainingVisualizer:
    def __init__(self, num_epochs):
        """
        Initialize visualization components
        
        Args:
            num_epochs: Total number of training epochs
        """
        # Prepare figure with subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 10))
        self.fig.suptitle('Training and Validation Metrics')
        
        # Training loss plot
        self.ax1.set_title('Training Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        
        # Accuracy plot
        self.ax2.set_title('Validation Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy (%)')
        
        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.train_accuracies = []
        
        # Line objects for dynamic plotting
        self.train_line, = self.ax1.plot([], [], 'b-')
        self.val_line, = self.ax2.plot([], [], 'r-')
        
        # Prepare plot
        plt.tight_layout()
        
    def update(self, train_loss, val_loss, train_acc, val_acc):
        """
        Update visualization with new metrics
        
        Args:
            train_loss: Current epoch's training loss
            val_accuracy: Current epoch's validation accuracy
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        self.train_accuracies.append(train_acc)
        
        # Update loss plot
        self.ax1.clear()
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.plot(self.train_losses, 'b--', label="Training Loss")
        self.ax1.plot(self.val_losses, 'g-', label="Validation Loss")
        self.ax1.legend()
        
        # Update accuracy plot
        self.ax2.clear()
        self.ax2.set_title('Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy (%)')
        self.ax2.plot(self.val_accuracies, 'r-', label="Validation Acc.")
        self.ax2.plot(self.train_accuracies, 'm--', label="Training Acc.")
        self.ax2.legend()
        
        # Redraw
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
    def save(self, filename='training_metrics.png'):
        """
        Save the final visualization
        
        Args:
            filename: Output filename for the plot
        """
        plt.savefig(filename)
        plt.close(self.fig)