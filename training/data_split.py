import os
import random
import torch

def get_split_indices(image_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Generates consistent train, validation, and test split indices for a given directory of images.

    Args:
        image_dir (str): Directory where the images are located.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing the indices for the 'train', 'val', and 'test' splits.
    """

    # Set seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)

    # List and sort images
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

    # Calculate dataset sizes
    total_size = len(images)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # Shuffle and split indices
    all_indices = list(range(total_size))
    random.shuffle(all_indices)
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]

    # Return a dictionary with split indices
    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices
}
