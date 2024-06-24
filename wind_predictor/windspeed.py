import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import os
import glob
import json
import csv
from PIL import Image
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class StormDataset(Dataset):
    """
    Custom dataset class for storm data.

    Args:
        root_dir (str): Root directory containing storm data.
        storm_id (str or list of str): Storm IDs for the dataset.
        sequence_length (int): Length of sequences to extract.
        split (str): Dataset split ('train' or 'test').
        test_size (float): Proportion of data to use for testing (if split is 'train').

    Attributes:
        root_dir (str): Root directory containing storm data.
        sequence_length (int): Length of sequences to extract.
        transform (torchvision.transforms.Compose): Image transformations.
        storm_id (str or list of str): Storm IDs for the dataset.
        sequences (list): List of sequences containing images, features, and labels.

    Methods:
        _load_and_process_data(): Loads and processes storm data.
    """

    def __init__(
        self,
        root_dir,
        storm_id,
        sequence_length=15,
        split=False,
        shuffle_data=True,
        test_size=0.2,
    ):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )
        self.storm_id = storm_id
        self.sequences = []
        self._load_and_process_data()
        if split:
            # Split the dataset into train and test sets
            train_sequences, test_sequences = train_test_split(
                self.sequences, test_size=test_size, random_state=42
            )
            self.sequences = train_sequences if split == "train" else test_sequences

    def _load_and_process_data(self):
        """
        Load and process storm data, extracting sequences of images, features, and labels.
        """
        time_features = []

        storms = self.storm_id

        for storm_id in storms:
            storm_path = os.path.join(self.root_dir, storm_id)
            all_files = os.listdir(storm_path)

            temp_images = []
            temp_features = []
            temp_labels = []

            for file in sorted(all_files):
                if file.endswith(".jpg"):
                    image_path = os.path.join(storm_path, file)
                    image = Image.open(image_path)
                    temp_images.append(self.transform(image))
                elif file.endswith("_features.json") or file.endswith("_label.json"):
                    with open(os.path.join(storm_path, file), "r") as f:
                        data = json.load(f)
                        if file.endswith("_features.json"):
                            temp_features.append(
                                [float(data["relative_time"]), float(data["ocean"])]
                            )
                        else:
                            temp_labels.append(float(data["wind_speed"]))
            max_relative_time = max([f[0] for f in temp_features])
            for feature in temp_features:
                feature[0] /= max_relative_time
            time_features.extend([f[0] for f in temp_features])

            for i in range(len(temp_images) - self.sequence_length):
                self.sequences.append(
                    {
                        "images": temp_images[i : i + self.sequence_length],
                        "features": temp_features[i : i + self.sequence_length],
                        "labels": temp_labels[i : i + self.sequence_length],
                    }
                )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return {
            "images": torch.stack(sequence["images"]),
            "features": torch.tensor(sequence["features"], dtype=torch.float),
            "labels": torch.tensor(sequence["labels"], dtype=torch.float),
        }


class Surprise_StormDataset(Dataset):
    """
    A custom dataset class designed for loading and processing storm-related data for machine learning models.

    This dataset class is tailored to work with images and associated features of storms,
    allowing for the extraction of sequential data that can be used for temporal analysis or prediction models.

    Parameters:
        root_dir (str): The root directory where the storm data is stored.
                        This should contain subdirectories for each storm, each with images and feature files.
        storm_id (str or list of str): A single storm ID or a list of storm IDs that the dataset will cover.
        sequence_length (int): The number of consecutive images (and their features) to include in a single sequence.

    Attributes:
        root_dir (str): Stores the root directory for storm data.
        sequence_length (int): The length of the sequences that will be extracted from the data.
        transform (torchvision.transforms.Compose): A series of torchvision transforms for preprocessing the images.
        storm_id (str or list of str): The list of storm IDs included in the dataset.
        sequences (list): A list where each element is a sequence containing images, features, and optionally labels.
    """

    def __init__(self, root_dir, storm_id=["tst"], sequence_length=15):
        """
        Initializes the dataset object, sets up the transformations for the images, and starts the data loading process.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize images to 224x224
                transforms.Grayscale(
                    num_output_channels=3
                ),  # Convert images to grayscale with 3 channels
                transforms.ToTensor(),  # Convert images to PyTorch tensors
            ]
        )
        self.storm_id = storm_id
        self.sequences = []  # Initialize an empty list to hold the data sequences
        self._load_and_process_data()  # Start loading and processing the data

    def _load_and_process_data(self):
        """
        Loads and processes the storm data by reading the images and their associated features from the filesystem.
        It constructs sequences of a specified length from consecutive images and features.
        """
        for storm_id in self.storm_id:
            storm_path = os.path.join(self.root_dir, storm_id)
            all_files = os.listdir(storm_path)

            temp_images = []
            temp_features = []

            # Loop through each file in the storm's directory
            for file in sorted(all_files):
                if file.endswith(".jpg"):  # For image files
                    image_path = os.path.join(storm_path, file)
                    image = Image.open(image_path)
                    temp_images.append(
                        self.transform(image)
                    )  # Apply transformations and store the image
                elif file.endswith("_features.json"):  # For feature files
                    with open(os.path.join(storm_path, file), "r") as f:
                        data = json.load(f)
                        temp_features.append(
                            [float(data["relative_time"]), float(data["ocean"])]
                        )  # Store features

            # Normalize the 'relative_time' feature by the maximum time for this storm
            max_relative_time = max([f[0] for f in temp_features])
            for feature in temp_features:
                feature[0] /= max_relative_time

            # Construct sequences from the images and features
            for i in range(len(temp_images) - self.sequence_length + 1):
                self.sequences.append(
                    {
                        "images": temp_images[i : i + self.sequence_length],
                        "features": temp_features[i : i + self.sequence_length],
                    }
                )

    def __len__(self):
        """
        Returns the total number of sequences in the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieves a specific sequence from the dataset by index.

        Parameters:
            idx (int): The index of the sequence to retrieve.

        Returns:
            A dictionary containing the sequence's images and features.
        """
        sequence = self.sequences[idx]
        return {
            "images": torch.stack(
                sequence["images"]
            ),  # Stack images into a single tensor
            "features": torch.tensor(
                sequence["features"], dtype=torch.float
            ),  # Convert features to a tensor
        }


class FeatureExtractor(nn.Module):
    def __init__(self):
        """
        Initialize a feature extractor module using a pretrained ResNet18 model.
        """
        super(FeatureExtractor, self).__init__()
        self.output_dim = 256  # Dimension of the output features
        # Load a pretrained ResNet18 model
        self.resnet18 = models.resnet18(pretrained=False)

        # Freeze the parameters of ResNet18 to prevent updates during training
        for param in self.resnet18.parameters():
            param.requires_grad = True

        # Modify the last fully connected layer of ResNet18 to match the desired output dimension
        self.resnet18.fc = nn.Sequential(
            nn.Linear(
                512, self.output_dim
            ),  # Input: 512-dimensional features, Output: output_dim-dimensional features
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Forward pass to extract features from input images.

        Args:
            x (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Extracted features.
        """
        # Forward pass through ResNet18
        x = self.resnet18(x)
        return x


class Modified_STA_LSTM(nn.Module):
    def __init__(
        self, time_feature_dim=2, lstm_hidden_dim=500, out_dim=1, use_gpu=False
    ):
        """
        Initialize a Modified STA-LSTM model.

        Args:
            time_feature_dim (int): Dimensionality of time features.
            lstm_hidden_dim (int): Dimensionality of the LSTM hidden state.
            out_dim (int): Dimensionality of the output.
            use_gpu (bool): Whether to use GPU for computations.
        """
        super(Modified_STA_LSTM, self).__init__()
        # Parameter initialization
        self.feature_extractor = FeatureExtractor()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.out_dim = out_dim
        self.use_gpu = use_gpu

        # Network architecture
        # Assuming the output dimension of feature_extractor is known
        combined_feature_dim = self.feature_extractor.output_dim + time_feature_dim
        self.lstm = nn.LSTM(combined_feature_dim, lstm_hidden_dim, batch_first=True)
        self.extra_fc = nn.Sequential(
            nn.Linear(
                lstm_hidden_dim, 256
            ),  # Input: lstm_hidden_dim-dimensional features, Output: 256-dimensional features
            nn.ReLU(),
            nn.Linear(
                256, 128
            ),  # Input: 256-dimensional features, Output: 128-dimensional features
            nn.ReLU(),
            nn.Linear(
                128, out_dim
            ),  # Input: 128-dimensional features, Output: out_dim-dimensional features
        )

    def forward(self, images, time_features):
        """
        Forward pass of the Modified STA-LSTM model.

        Args:
            images (torch.Tensor): Input images.
            time_features (torch.Tensor): Time-related features.

        Returns:
            torch.Tensor: Model output.
        """
        # Extract image features
        batch_size, sequence_length, C, H, W = images.size()
        cnn_input = images.view(batch_size * sequence_length, C, H, W)
        image_features = self.feature_extractor(cnn_input)
        image_features = image_features.view(batch_size, sequence_length, -1)

        # Ensure the time features have the correct dimension
        time_features = time_features.view(batch_size, sequence_length, -1)

        # Concatenate image features and time features
        combined_features = torch.cat([image_features, time_features], dim=2)

        # Process through LSTM
        lstm_out, _ = self.lstm(combined_features)
        # Take the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]

        # Pass through additional fully connected layers and activation functions
        out = self.extra_fc(last_time_step_out)
        return out
