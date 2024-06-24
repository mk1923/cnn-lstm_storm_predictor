import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import torch
from sklearn.model_selection import train_test_split
import csv
import re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class GenStormDataset(Dataset):
    """
    The first Dataset is designed to be used with a generic model.

    It returns a dictionary:
        'storm_id': sample['storm_id'],
        'sample_number': sample['sample_number'],
        'relative_times': relative_times_tensor,
        'oceans': oceans_tensor,
        'wind_speeds': wind_speeds_tensor,
        'images': storm_images_tensor,
        'outimage': outimage

    """
    def __init__(self, root_dir, window_size=10, train=True, test_size=0.2):
        """
        Initializes the GenStormDataset.

        Args:
            root_dir (str): Root directory containing storm data.
            window_size (int): Size of the window for temporal sequence.
            train (bool): Flag indicating whether to use the training set.
            test_size (float): Proportion of data to use as the test set.
        """
        self.root_dir = root_dir
        self.window_size = window_size
        self.samples = []

        storms = os.listdir(root_dir)
        for storm_id in storms:
            storm_path = os.path.join(root_dir, storm_id)
            sample_files = [file for file in os.listdir(storm_path) if
                            file.endswith('.jpg')]
            for sample_file in sample_files:
                csv.reader("")
                sample_number = int(sample_file.split('_')[1].split('.')[0])
                feature_file = f"{storm_id}_{sample_number:03d}_features.json"
                label_file = f"{storm_id}_{sample_number:03d}_label.json"

                feature_path = os.path.join(storm_path, feature_file)
                label_path = os.path.join(storm_path, label_file)
                image_path = os.path.join(storm_path, sample_file)

                # Load features
                with open(feature_path, 'r') as f:
                    feature_data = json.load(f)

                # Load labels
                with open(label_path, 'r') as f:
                    label_data = json.load(f)

                self.samples.append({
                    'storm_id': storm_id,
                    'sample_number': f"{sample_number:03d}",
                    'feature_data': feature_data,
                    'label_data': label_data,
                    'image_path': image_path
                })

        self.samples.sort(key=lambda x: (x['storm_id'],
                                         int(x['sample_number'])))

        # Splitting the dataset into train and test sets
        train_samples, test_samples = train_test_split(
            self.samples, train_size=(1 - test_size), random_state=42)
        self.samples = train_samples if train else test_samples

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(366)
        ])

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing storm data for the given index.
        """
        storm_images = []
        relative_times = []
        oceans = []
        wind_speeds = []

        start_idx = max(0, idx - self.window_size + 1)

        for sample_idx in range(start_idx, idx + 1):
            if sample_idx < len(self.samples):  # Check index is within bounds
                sample = self.samples[sample_idx]
                feature_data = sample['feature_data']
                label_data = sample['label_data']

                image_path = sample['image_path']
                image = Image.open(image_path)
                image = self.transform(image)

                relative_times.append(int(feature_data.get(
                    'relative_time', None)))
                oceans.append(int(feature_data.get('ocean', None)))
                wind_speeds.append(float(label_data.get('wind_speed', None)))
            else:  # If the index is out of bounds, add placeholders
                image = torch.zeros((1, 366, 366))
                relative_times.append(0)
                oceans.append(0)
                wind_speeds.append(0)

            storm_images.append(image)

        # Ensure the lists have the required window size using placeholders
        for _ in range(self.window_size - len(storm_images)):
            storm_images.append(torch.zeros((1, 366, 366)))
            relative_times.append(0)
            oceans.append(0)
            wind_speeds.append(0)

        storm_images_tensor = torch.stack(storm_images)
        relative_times_tensor = torch.tensor(relative_times, dtype=torch.int32)
        oceans_tensor = torch.tensor(oceans, dtype=torch.int32)
        wind_speeds_tensor = torch.tensor(wind_speeds, dtype=torch.float)

        idx_outside_window = idx+1 if idx + 1 < len(self.samples) else idx
        outimage = Image.open(self.samples[idx_outside_window]['image_path'])
        outimage = self.transform(outimage)

        return {
            'storm_id': sample['storm_id'],
            'sample_number': sample['sample_number'],
            'relative_times': relative_times_tensor,
            'oceans': oceans_tensor,
            'wind_speeds': wind_speeds_tensor,
            'images': storm_images_tensor,
            'outimage': outimage
        }


class WindStormDataset(Dataset):
    """
    Dataset class for wind storm data.

    Args:
        root_dir (str): Root directory containing wind storm data.
        storm_id (str): Specific storm ID, if provided.
        split (str): 'train' or 'test' to indicate dataset split.
        test_size (float): Proportion of data to use as the test set.
    """
    def __init__(self, root_dir, storm_id=None, split='train', test_size=0.2):
        self.root_dir = root_dir
        self.storm_id = storm_id
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.data = []
        self._prepare_data()

        # Splitting the dataset into train and test sets
        train_data, test_data = train_test_split(
            self.data, test_size=test_size, random_state=42)
        self.data = train_data if split == 'train' else test_data

    def _prepare_data(self):
        storm_ids = [self.storm_id] if self.storm_id else os.listdir(
            self.root_dir)
        for storm_id in storm_ids:
            storm_path = os.path.join(self.root_dir, storm_id)
            if not os.path.isdir(storm_path):
                continue

            for file in sorted(os.listdir(storm_path)):
                if file.endswith('.jpg'):
                    image_path = os.path.join(storm_path, file)
                    image = Image.open(image_path)
                    image = self.transform(image)

                    label_file = file.replace('.jpg', '_label.json')
                    label_path = os.path.join(storm_path, label_file)
                    with open(label_path, 'r') as f:
                        label_data = json.load(f)
                    wind_speed = float(label_data['wind_speed'])

                    self.data.append({'image': image, 'label': torch.tensor(
                        wind_speed, dtype=torch.float)})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ImageStormDataset(Dataset):
    def __init__(self, root: str, storm: str, sequence_length: int, alldata=False) -> None:
        self.sequence_length = sequence_length
        self.prefix = root
        self.storm_name = storm

        # 图像预处理转换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])

        # 加载数据并预处理
        self.data, self.preprocessed_images = self._load_data()
        self.data.sort_values(by='Relative_Time', inplace=True)

        # 时间缩放器
        self.time_scaler = MinMaxScaler()
        self.data['Relative_Time'] = self.time_scaler.fit_transform(self.data[['Relative_Time']])
        self.data['Time_Diff'] = self.data['Relative_Time'].diff()
        self.data['Time_Diff'].fillna(0, inplace=True)

    def _load_data(self):
        datalist = []
        preprocessed_images = {}

        storm_dir = os.path.join(self.prefix, self.storm_name)
        if os.path.isdir(storm_dir):
            image_files = [file for file in os.listdir(storm_dir) if file.endswith((".jpg", ".jpeg"))]
            for im in image_files:
                split_names = re.split(r"[_.]", im)
                split_names = [part for part in split_names if part]
                name, num = split_names[0], split_names[1]

                # 加载并预处理图像
                image_path = os.path.join(storm_dir, im)
                image = Image.open(image_path)
                preprocessed_images[num] = self.transform(image)

                # 读取 JSON 数据
                json_data = self._load_json(os.path.join(storm_dir, name + "_" + num))
                json_data["Id"] = num
                datalist.append(json_data)

        return pd.DataFrame(datalist), preprocessed_images

    def _load_json(self, file_prefix):
        with open(file_prefix + "_label.json", "r") as json_file:
            label_data = json.load(json_file)
        with open(file_prefix + "_features.json", "r") as json_file:
            features_data = json.load(json_file)

        return {
            "Storm_Name": features_data["storm_id"],
            "Wind_Speed": int(label_data["wind_speed"]),
            "Relative_Time": int(features_data["relative_time"]),
            "Ocean": int(features_data["ocean"]),
            "Image_Path": file_prefix + ".jpg",
        }

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        if idx + self.sequence_length >= len(self.data):
            idx = len(self.data) - self.sequence_length - 1

        sequence_data = self.data.iloc[idx: idx + self.sequence_length]
        images = [self.preprocessed_images[str(id)] for id in sequence_data["Id"]]

        sample_input = {
            "image": torch.stack(images),
            "relative_time": torch.tensor(sequence_data["Relative_Time"].to_numpy(), dtype=torch.float32),
            "time_diff": torch.tensor(sequence_data["Time_Diff"].to_numpy(), dtype=torch.float32),
            "wind_speed": torch.tensor(sequence_data["Wind_Speed"].to_numpy(), dtype=torch.float32),
        }

        next_image_id = self.data.iloc[idx + self.sequence_length]["Id"]
        next_image = self.preprocessed_images[str(next_image_id)]

        return sample_input, next_image

    def __str__(self):
        class_string = self.__class__.__name__ + "\n\tlen : %d" % self.__len__()
        for key, value in self.__dict__.items():
            if key not in ["data", "preprocessed_images"]:
                class_string += "\n\t" + str(key) + " : " + str(value)
        return class_string


class StormDataLoader(DataLoader):
    """
    DataLoader class for storm data.

    Args:
        *args: Variable-length argument list.
        **kwargs: Arbitrary keyword arguments.
    """
    def __init__(self, *args, **kwargs):
        super(StormDataLoader, self).__init__(*args, **kwargs)
