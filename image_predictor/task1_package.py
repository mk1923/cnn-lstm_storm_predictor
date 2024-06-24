import os
import json
import re
from sympy import false
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from pytorch_msssim import ssim
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_msssim import ssim
import torch
import matplotlib.pyplot as plt


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
        self, root_dir, storm_id, sequence_length=15, split="train", test_size=0.2
    ):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )
        self.storm_id = storm_id
        self.sequences = []
        self._load_and_process_data()

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


# Function to retrieve a list of storm IDs from a specified directory
def get_storm_ids(root_dir):
    """
    Get a list of storm IDs from the specified directory.

    Args:
        root_dir (str): The root directory containing storm data folders.

    Returns:
        list: A list of storm IDs.
    """
    storm_ids = [
        name
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    ]
    return storm_ids


class ConvLSTMCell(nn.Module):
    "References:https://chat.openai.com/share/c739be3f-b7f2-4d02-8ddc-1312ca32f4d1"

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_channel = input_dim
        self.hidden_channel = hidden_dim
        self.kernel_sz = kernel_size
        self.pad = kernel_size[0] // 2, kernel_size[1] // 2
        self.use_bias = bias
        self.conv = nn.Conv2d(
            in_channels=self.input_channel + self.hidden_channel,
            out_channels=4 * self.hidden_channel,
            kernel_size=self.kernel_sz,
            padding=self.pad,
            bias=self.use_bias,
        )

    def forward(self, input_tensor, cur_state):
        h_current, c_current = cur_state
        combined = torch.cat([input_tensor, h_current], dim=1)
        conv_result = self.conv(combined)
        cc_inputgate, cc_forgetgate, cc_outputgate, cc_cellgate = torch.split(
            conv_result, self.hidden_channel, dim=1
        )
        input_gate = torch.sigmoid(cc_inputgate)
        forget_gate = torch.sigmoid(cc_forgetgate)
        output_gate = torch.sigmoid(cc_outputgate)
        cell_gate = torch.tanh(cc_cellgate)
        c_next = forget_gate * c_current + input_gate * cell_gate
        h_next = output_gate * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_channel,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_channel,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class ConvLSTM(nn.Module):
    "References:https://chat.openai.com/share/c739be3f-b7f2-4d02-8ddc-1312ca32f4d1"

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        batch_first=False,
        bias=True,
        return_all_layers=False,
    ):
        super(ConvLSTM, self).__init__()
        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]
                )
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


class SimpleCNN(nn.Module):
    """A simple 3 layer convolution network"""

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.conv_layers(x)


class StormGenerator(nn.Module):
    """
    StormGenerator is a neural network model for generating storm-related images.

    This model includes an encoder-decoder architecture with a ConvLSTM layer for
    processing spatiotemporal features.

    Architecture:
        - Encoder: SimpleCNN for feature extraction.
        - ConvLSTM: Convolutional LSTM layer for capturing spatiotemporal patterns.
        - Decoder: Transpose convolutional layers for generating output images.

    """

    def __init__(self):
        super(StormGenerator, self).__init__()
        self.encoder = SimpleCNN()
        self.conv_lstm = ConvLSTM(
            input_dim=128,
            hidden_dim=[64, 32],
            kernel_size=(3, 3),
            num_layers=2,
            batch_first=True,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, input_imgs):
        batch_size, sequence_len, c, h, w = input_imgs.size()
        c_input = input_imgs.reshape(batch_size * sequence_len, c, h, w)
        c_output = self.encoder(c_input)
        c_output = c_output.view(batch_size, sequence_len, -1, h // 8, w // 8)
        conv_lstm_out, _ = self.conv_lstm(c_output)
        conv_lstm_out = conv_lstm_out[0][:, -1, :, :, :]
        output_image = self.decoder(conv_lstm_out)
        return output_image


def train(model, train_loader, optimizer, device):
    model.train()  # Set the model to training mode
    loss_history = []
    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device)

            input_images = images[
                :, :5, :, :, :
            ]  # Use the first five images as input sequence
            target_image = images[:, 7, :, :, :].squeeze(
                1
            )  # Use the 8th image as target
            target_image = target_image.unsqueeze(1)

            optimizer.zero_grad()

            predicted_image = model(input_images)

            # Calculate loss using SSIM
            loss = 1 - ssim(
                predicted_image, target_image, data_range=1, size_average=True
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)  # Store the average loss for this epoch

        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.show()


def eval(model, data_loader, device):
    # we can choose to use the model parameters which we already trained.
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = StormGenerator().to(device)
    # model.load_state_dict(torch.load('/content/drive/MyDrive/surprise_storm_100_ssim0.25_model.pth'))
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images = batch["images"].to(device)

            input_images = images[
                :, :5, :, :, :
            ]  # Use the first five images as input sequence
            target_image = images[:, 7, :, :, :].squeeze(
                1
            )  # Use the 8th image as target

            # Generate prediction
            predicted_image = model(input_images)

            if batch_idx == 0:  # Just show the first batch
                fig, axs = plt.subplots(2, 1, figsize=(10, 5))

                # Show target image
                axs[0].set_title("Target Image")
                axs[0].imshow(target_image[0].cpu().numpy(), cmap="gray")
                axs[0].axis("off")

                # Show predicted image
                axs[1].set_title("Predicted Image")
                axs[1].imshow(predicted_image[0].squeeze().cpu().numpy(), cmap="gray")
                axs[1].axis("off")

                plt.show()
                break  # only show one set of images


def eval_continuous_prediction(
    model, data_loader, device, start_batch=1, num_predictions=3
):
    # we can use the model we trained before.
    # model = StormGenerator().to(device)
    # model.load_state_dict(torch.load('/content/drive/MyDrive/surprise_storm_100_ssim0.25_model.pth'))
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images = batch["images"].to(device)

            input_sequence = images[:, :5, :, :, :]
            iter = 5
            predicted_images = []
            for _ in range(num_predictions):
                # Generate prediction for the next image
                predicted_image = model(input_sequence)
                predicted_images.append(predicted_image)
                input_sequence = torch.cat(
                    (
                        input_sequence[:, 1:, :, :, :],
                        images[:, iter, :, :, :].unsqueeze(1),
                    ),
                    dim=1,
                )
                iter += 1
            if batch_idx == start_batch:  # Show and save images for the start_batch
                print(batch["images"].shape)
                # Display and save predicted images
                fig, axs = plt.subplots(1, num_predictions, figsize=(15, 5))
                for i in range(num_predictions):
                    # Show predicted image
                    axs[i].set_title(f"Predicted Image {i+1}")
                    predicted_img = predicted_images[i][0].squeeze().cpu().numpy()
                    axs[i].imshow(predicted_img, cmap="gray")
                    axs[i].axis("off")

                    # Save predicted image
                    save_path = f"/content/drive/MyDrive/Yolanda_generatedimages_85to1/tst{i+252}.jpg"
                    plt.imsave(save_path, predicted_img, cmap="gray")
                    print(f"Saved predicted image {i+1} at {save_path}")

                plt.show()
                break
