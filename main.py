import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from configuration import NeuralNetworkTrainer


class Visualizer:
    @staticmethod
    def plot_training_history(history, save_path='information_visualization/training_history.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        sns.lineplot(data=history['loss'], 
                    label='Training Loss', ax=ax1, linewidth=2.5)
        sns.lineplot(data=history['val_loss'], 
                    label='Validation Loss', ax=ax1, linewidth=2.5)
        ax1.set_title('Model Loss During Training')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        sns.lineplot(data=history['accuracy'], 
                    label='Training Accuracy', ax=ax2, linewidth=2.5)
        sns.lineplot(data=history['val_accuracy'], 
                    label='Validation Accuracy', ax=ax2, linewidth=2.5)
        ax2.set_title('Model Accuracy During Training')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    @staticmethod
    def plot_confusion_matrix(true_labels, pred_labels, save_path='information_visualization/confusion_matrix.png'):
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix on Test Data')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(save_path)
        plt.close()


def main():
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 5)
    
    trainer = NeuralNetworkTrainer()
    trainer.load_and_preprocess_data()
    trainer.build_model()
    trainer.compile_model()
    trainer.train_model()
    
    test_accuracy = trainer.evaluate_model()
    
    history = trainer.get_training_history()
    Visualizer.plot_training_history(history)
    
    test_images, test_labels = trainer.get_test_data()
    pred_labels = trainer.get_predictions()
    Visualizer.plot_confusion_matrix(test_labels, pred_labels)


if __name__ == "__main__":
    main()