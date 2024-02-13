from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.data_utils import fold_data_list


def compute_class_weights(n_classes, train_loader):
    class_counts = Counter()
    for _, labels in train_loader:
        class_counts.update(labels.numpy())
    total_samples = sum(class_counts.values())
    class_frequencies = {key: value / total_samples for key, value in class_counts.items()}

    # Calculate class weights as the inverse of class frequencies
    class_weights = {key: 1.0 / value for key, value in class_frequencies.items()}

    # Normalize weights so that they sum to the number of classes
    sum_weights = sum(class_weights.values())
    class_weights_normalized = {key: value / sum_weights for key, value in class_weights.items()}
    class_weights_list = [class_weights_normalized[x] for x in range(n_classes)]
    print("Class Counts:", class_counts)
    print("Class Weights:", class_weights_list)
    return class_weights_list


def train_model(ages, config, model, n_classes, train_loader, hidden_dim):
    # Train or load model
    if config['dataset'] != config['model_dataset'] or not config['train']:
        print("Loading model (1/1)")
        model.load_state_dict(torch.load(
            f"{config['models_base_path']}/model_{config['model_dataset']}_fold{config['fold']}_torch{hidden_dim}"))
        model.eval()
    else:
        print("Training model (1/2) - computing class weights")
        class_weights_list = compute_class_weights(n_classes, train_loader)
        print("Training model - training loop (2/2)")
        for idx, epochs in enumerate(ages):
            print(f"Age {idx} with {epochs} epochs")
            criterion = nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights_list).cuda())
            optimizer = optim.AdamW(
                model.parameters(), lr=0.0001, weight_decay=0.01)  # 0.0001
            for _ in range(epochs):
                running_loss = 0
                for step, (inputs, labels) in enumerate(train_loader, start=1):
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    print(
                        f'\r Step: {step} Loss: {(running_loss / step):.4f}', end='')

        # Save model
        save_path_str = f"{config['models_base_path']}/model_{config['dataset']}_fold{config['fold']}_torch{hidden_dim}"
        print(f"Saving model to {save_path_str}")
        torch.save(model.state_dict(), save_path_str)


def evaluate_model(fold, n_folds, data_dict, model):
    model.eval()
    print("Evaluating model's accuracy (1/1)")
    # Test on training data (no augmentation obvs)
    ok = 0
    checked = 0
    checked_p = 0
    checked_n = 0
    for sample in fold_data_list(data_dict, n_folds, fold, split="test"):
        image_in = torch.tensor(sample["image"], dtype=torch.float).cuda()
        if len(image_in.shape) < 5:
            image_in = image_in.unsqueeze(0)
        model_out = model(image_in).cpu().detach().numpy()
        checked += 1
        checked_p = checked_p + 1 if sample['label'] else checked_p
        checked_n = checked_n + 1 if not sample['label'] else checked_n
        if np.argmax(model_out) == sample['label']:
            ok += 1
    print(f"Accuracy: {ok / checked}")
    print(f"There where {checked_p} positive and {checked_n} negative samples")
