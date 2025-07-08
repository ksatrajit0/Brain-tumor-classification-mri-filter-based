import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history):
    tr_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    tr_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(tr_acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, tr_loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'g', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, tr_acc, 'r', label='Training Acc')
    plt.plot(epochs, val_acc, 'g', label='Validation Acc')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
