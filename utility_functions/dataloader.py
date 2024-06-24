import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json


class StormDataset(Dataset):
    def __init__(self, root_dir, train=True, test_size=0.2, random_state=42):
        self.root_dir = root_dir
        self.storms = os.listdir(root_dir)

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

        train_storm = self.samples[:-3]
        test_storm = self.samples[-3:]

        # Transforms:
        # resize images?
        # normalize images?
        # colour scale?
        # augmentations? (rotation/ flipping/ cropping/ etc.)
        # tensor for images?
        # normalize labels? (z-score/ min-max/ etc.)

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

        return {'storm_id': storm_id,
                'sample_number': sample['sample_number'],
                'relative_time': relative_time,
                'ocean': ocean,
                'wind_speed': wind_speed,
                'image': image}


class StormDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(StormDataLoader, self).__init__(*args, **kwargs)
