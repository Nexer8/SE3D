import argparse
import os
import random
from glob import glob

import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam
from volumentations import Compose, Flip

from model import get_model

# Constants
DATA_TYPES = ['flair', 't1', 't1ce', 't2', 'seg']
CLASS_NAMES = ['normal', 'tumor']


def train(base_path: str, model_path: str, fold: int, wandb: bool = False) -> None:
    """
    Train the model.

    Args:
        base_path (str): Root folder of metadata.
        model_path (str): Root folder of the model to evaluate.
        fold (int): Fold to evaluate.
        wandb (bool): Use wandb for logging.
    """
    # Load Environment Variables
    load_dotenv()

    # Project Configuration
    config = {
        'learning_rate': 1e-4,
        'min_learning_rate': 1e-7,
        'epochs': 1,  # 200,
        'batch_size': 8,
        'test_batch_size': 1,
        'img_size': 240,
        'depth': 155,
        'n_classes': 2,
        'fold': fold,
    }

    # Random Seed for Reproducibility
    seed = 27

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)

    # Weights & Biases Initialization
    if wandb:
        import wandb
        wandb.init(project="your_project_name")
        wandb.config = config

        wandb.login(key=os.environ.get('WANDB_API_KEY'))
        run = wandb.init(
            name=f'0-3D-CNN-f{fold}',
            project=os.environ.get('WANDB_BRATS_PROJECT'),
            entity=os.environ.get('WANDB_ENTITY'),
            id=f'0-3D-CNN-f{fold}',
        )

    # Create Output Directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        os.mkdir(f'{model_path}/fold{config["fold"]}')

    # Load Metadata
    df = pd.read_csv(f'{base_path}/metadata.csv')
    print(df['label'].value_counts())

    # Split Data into Train and Test
    labels = np.array(df['label'])
    paths = np.array(df['path'])
    train_indices = df[df['fold'] != config['fold']].index.values.tolist()
    test_indices = df[df['fold'] == config['fold']].index.values.tolist()

    # Data Augmentation
    def get_augmentations() -> Compose:
        return Compose([
            Flip(1, p=0.5),
        ], p=1.0)

    # Data Generator
    class DataGenerator(tf.keras.utils.Sequence):
        """Generates data for Keras"""

        def __init__(self, indices, paths, labels, batch_size=4, dim=(240, 120, 155),
                     n_classes=2, shuffle=True, transform=None):
            """Initialization"""
            self.indexes = None
            self.dim = dim
            self.batch_size = batch_size
            self.paths = paths
            self.labels = labels
            self.indices = indices
            self.n_classes = n_classes
            self.shuffle = shuffle
            self.transform = transform
            self.on_epoch_end()

        def __len__(self) -> int:
            """Denotes the number of batches per epoch"""
            return int(np.floor(len(self.indices) / self.batch_size))

        def __getitem__(self, index):
            """Generate one batch of data"""
            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            # Find list of IDs
            indices_temp = [self.indices[k] for k in indexes]

            # Generate data
            X, y = self.__data_generation(indices_temp)

            return X, y

        def on_epoch_end(self):
            """Updates indexes after each epoch"""
            self.indexes = np.arange(len(self.indices))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indices_temp):
            """Generates data containing batch_size samples"""
            # Initialization
            X = np.empty((self.batch_size, *self.dim, len(DATA_TYPES[:-1])))
            y = np.empty(self.batch_size, dtype=int)

            # Generate data
            for i, ID in enumerate(indices_temp):
                # Store sample
                vol_path = [
                    path for path in glob(f'{base_path}/{self.paths[ID]}/*.npz') if not 'seg' in path
                ][0]
                volume = np.load(vol_path)['data']
                volume = (volume.astype('float32') / 255.0).astype('float32')

                if self.transform is not None:
                    volume = self.transform(**{'image': volume})['image']
                X[i,] = volume

                # # Store class
                y[i] = self.labels[ID]

            return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

    # Initialize Data Generators
    augmentations = get_augmentations()

    training_generator = DataGenerator(
        train_indices,
        paths,
        labels,
        batch_size=config['batch_size'],
        dim=(config['img_size'], config['img_size'] // 2, config['depth']),
        shuffle=True,
        transform=augmentations,
    )

    test_generator = DataGenerator(
        test_indices,
        paths,
        labels,
        batch_size=config['test_batch_size'],
        dim=(config['img_size'], config['img_size'] // 2, config['depth']),
        shuffle=False,
    )

    # Create Model
    model = get_model(height=config['img_size'], width=config['img_size'] // 2,
                      depth=config['depth'], channels=len(DATA_TYPES[:-1]), n_classes=config['n_classes'])

    # Class Weights
    train_labels = [labels[idx] for idx in train_indices]
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',  # balanced for computing weights based on class count
        classes=np.unique(train_labels),
        y=train_labels,
    )
    class_weights = dict(zip(np.unique(labels), class_weights))

    # Compile Model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=config['learning_rate']),
        metrics=['acc'],
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(f'{model_path}/fold{config["fold"]}/best{config["fold"]}.h5',
                        save_best_only=True, monitor='loss', mode='min'),
        ReduceLROnPlateau(
            monitor='loss',
            mode='min',
            patience=7,
            factor=0.5,
            min_lr=config['min_learning_rate'],
        ),
        EarlyStopping(monitor='loss', mode='min',
                      patience=20, restore_best_weights=True)
    ]
    if wandb:
        from wandb.keras import WandbCallback
        callbacks.append(WandbCallback(monitor='loss', mode='min'))

    # Train Model
    model.fit(
        training_generator,
        epochs=config['epochs'],
        class_weight=class_weights,
        verbose=2,
        callbacks=callbacks,
        max_queue_size=20,
        workers=6,
    )

    # Training History
    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()

    for i, metric in enumerate(['acc', 'loss']):
        ax[i].plot(model.history.history[metric])
        ax[i].set_title(f'Model {metric}')
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(metric)
    # save fig as pdf
    fig.savefig(f'{model_path}/fold{config["fold"]}/training_history{config["fold"]}.pdf')

    # Cleanup
    if wandb:
        wandb.finish()

    # Model Evaluation
    model.load_weights(f'{model_path}/fold{config["fold"]}/best{config["fold"]}.h5')
    y_pred = [np.argmax(x) for x in model.predict(test_generator, batch_size=config['test_batch_size'])]
    y_test = [labels[idx] for idx in test_indices]

    # Classification Report
    print(classification_report(y_test, y_pred, digits=3))
    with open(f'{model_path}/fold{config["fold"]}/cr{config["fold"]}.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred, digits=3))

    # Confusion Matrix
    cfsn_matrix = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cfsn_matrix, index=range(config['n_classes']), columns=CLASS_NAMES)
    plt.figure(figsize=(15, 6))
    sn.heatmap(df_cm, annot=True, linewidths=0.5, fmt='d')
    plt.savefig(f'{model_path}/fold{config["fold"]}/cm{config["fold"]}.pdf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_root', type=str, default='.',
                        help="Root folder of metadata.")
    parser.add_argument('--model_path', type=str, default='.',
                        help="Root folder of the model to evaluate.")
    parser.add_argument('--fold', type=int, default=0,
                        help="Fold to evaluate.")
    parser.add_argument('--wandb', type=bool, default=False,
                        help="Log to wandb.")
    args = parser.parse_args()

    train(
        base_path=args.metadata_root,
        model_path=args.model_path,
        fold=args.fold,
        wandb=args.wandb
    )


if __name__ == "__main__":
    main()
