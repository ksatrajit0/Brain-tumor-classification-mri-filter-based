import os
from tensorflow.keras.callbacks import ModelCheckpoint
from src import config
from src.data_loader import get_data_generators
from src.model_train import build_model
from src.utils import plot_training_history

def train_and_save():
    # Load data generators
    train_gen, valid_gen, test_gen, _ = get_data_generators()

    # Build model
    model = build_model(input_shape=config.IMG_SHAPE, num_classes=len(train_gen.class_indices))

    # Set up checkpoint callback
    checkpoint_path = "efficientnetb3_braintumor_weights.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1)

    # Train the model
    history = model.fit(
        train_gen,
        epochs=30,
        validation_data=valid_gen,
        callbacks=[checkpoint],
        verbose=1
    )

    # Plot training results
    plot_training_history(history)

    # Save model structure and weights
    model.save("brain_tumor_model.h5")

    print("Training complete. Model and weights saved.")

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
    train_and_save()
