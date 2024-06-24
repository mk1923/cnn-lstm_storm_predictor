import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import torch
from sklearn.model_selection import train_test_split


class StormDataset(Dataset):
    def __init__(self, root_dir, window_size=10, train=True, test_size=0.2):
        self.root_dir = root_dir
        self.storms = os.listdir(root_dir)
        self.window_size = window_size

        self.samples = []
        for storm_id in self.storms:
            storm_path = os.path.join(root_dir, storm_id)
            sample_files = [file for file in os.listdir(storm_path) if
                            file.endswith('.jpg')]
            for sample_file in sample_files:
                sample_number = int(sample_file.split('_')[1].split('.')[0])
                feature_file = f"{storm_id}_{sample_number:03d}_features.json"
                label_file = f"{storm_id}_{sample_number:03d}_label.json"

                feature_path = os.path.join(storm_path, feature_file)
                label_path = os.path.join(storm_path, label_file)
                image_path = os.path.join(storm_path, sample_file)

                self.samples.append({
                    'storm_id': storm_id,
                    'sample_number': f"{sample_number:03d}",
                    'feature_path': feature_path,
                    'label_path': label_path,
                    'image_path': image_path
                })

        self.samples.sort(key=lambda x: (x['storm_id'],
                                         int(x['sample_number'])))  # sort data

        train_storm, test_storm = train_test_split(self.samples,
                                                   train_size=split_index,
                                                   random_state=42)

        if train:
            self.samples = train_storm
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(366)
            ])
        else:
            self.samples = test_storm
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(366)
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            with open(sample['feature_path'], 'r') as f:
                feature_data = json.load(f)
                storm_id = feature_data.get('storm_id', None)
                relative_time = int(feature_data.get('relative_time', None))
                # relative_time = (relative_time - self.relative_time_mean) / self.relative_time_std  # Normalize relative time
                ocean = int(feature_data.get('ocean', None))
        except Exception as e:
            print(f"Error loading feature data for sample {sample}: {e}")
            return None

        try:
            with open(sample['label_path'], 'r') as f:
                label_data = json.load(f)
                wind_speed = float(label_data.get('wind_speed', None))
        except Exception as e:
            print(f"Error loading label data for sample {sample}: {e}")
            return None

        try:
            image = Image.open(sample['image_path'])
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image for sample {sample}: {e}")
            return None

        # Load all images for the storm
        storm_images = []
        window_indices = []
        for sample_idx in range(idx, idx + self.window_size):
            if sample_idx < 0 or sample_idx >= len(self.samples):
                storm_images.append(torch.zeros((1, 366, 366)))  # Placeholder
                continue  # Skip if index is out of bounds

            try:
                image = Image.open(self.samples[sample_idx]['image_path'])
                image = self.transform(image)
                storm_images.append(image)
            except Exception as e:
                print(f"Error loading image for sample\
                       {self.samples[sample_idx]}: {e}")

        while len(storm_images) < self.window_size:
            storm_images.append(torch.zeros((1, 366, 366))) # Placeholder

        # Convert the list of images to a tensor
        storm_images_tensor = torch.stack(storm_images)

        # # Determine the index outside the window
        if self.samples[sample_idx + self.window_size + 1]['sample_number'] == 0:
            idx += self.window_size

        idx_outside_window = idx+1 if idx + 1 < len(self.samples) else idx

        # Retrieve the first image outside the window
        try:
            outimage = Image.open(self.samples[idx_outside_window]
                                    ['image_path'])
            outimage = self.transform(outimage)
        except Exception as e:
            print("Error loading image for sample", self.samples
                    [idx_outside_window], ":", e)
            outimage = None

        print(f'Window for sample{sample}: {window_indices}')

        return {'storm_id': storm_id,
                'sample_number': sample['sample_number'],
                'relative_time': relative_time,
                'ocean': ocean,
                'wind_speed': wind_speed,
                'image': storm_images_tensor,
                'outimage': outimage}


class StormDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(StormDataLoader, self).__init__(*args, **kwargs)
