import matplotlib.pyplot as plt
import numpy as np

def plot_performance(precision, recall, f1, accuracy, labels):
    x = np.arange(len(labels))
    width = 0.23
    fig, ax = plt.subplots(figsize=(14, 8))

    bars = [
        ax.bar(x - 3*width/2, precision, width, label='Precision'),
        ax.bar(x - width/2, recall, width, label='Recall'),
        ax.bar(x + width/2, f1, width, label='F1 Score'),
        ax.bar(x + 3*width/2, accuracy, width, label='Accuracy')
    ]

    for bar in bars:
        for rect in bar:
            height = rect.get_height()
            ax.annotate(f'{height:.0f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    ax.set_ylabel('Scores')
    ax.set_title('Class-wise Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    plt.ylim(90, 100)
    plt.tight_layout()
    plt.show()
