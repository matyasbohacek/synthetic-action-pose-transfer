
import os
import random
import argparse
import torch
import hashlib

import sys
import numpy as np
import torchvision.transforms as T

from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.io import read_video
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, TimesformerForVideoClassification, VivitImageProcessor, VivitForVideoClassification


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_fixed_test_set(filepaths, test_size=50):
    random.shuffle(filepaths)
    return filepaths[test_size:], filepaths[:test_size]


def apply_class_cap(filepaths, class_cap_per_dir, video_dir):
    capped_filepaths = []
    class_counts = {}
    for video_path, label in filepaths:
        if video_path.startswith(video_dir):
            class_counts[label] = class_counts.get(label, 0)
            if class_counts[label] < class_cap_per_dir.get(video_dir, float('inf')):
                capped_filepaths.append((video_path, label))
                class_counts[label] += 1
    return capped_filepaths


def sample_frame_indices_vivit(clip_len, frame_sample_rate, total_frames):
    start_idx = 0
    end_idx = min(total_frames, clip_len * frame_sample_rate)
    indices = torch.linspace(start_idx, end_idx - 1, steps=clip_len).long()
    return indices


def preprocess_video_vivit(videos, processor, device):
    batch_size, num_frames, channels, height, width = videos.shape
    flattened_frames = videos.view(-1, channels, height, width).to(device)
    processed = processor(images=list(flattened_frames), return_tensors="pt", return_pixel_values=True)
    pixel_values = processed["pixel_values"].view(batch_size, num_frames, channels, height, width)
    inputs = {"pixel_values": pixel_values.to(device)}
    return inputs


def setup_logging(experiment_name):
    rand_hash = hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
    log_filename = f"{experiment_name}_{rand_hash}.txt"
    log_filepath = os.path.join(os.getcwd(), log_filename)
    sys.stdout = open(log_filepath, "w")
    sys.stderr = sys.stdout
    print(f"start time: {datetime.now()}")
    print(f"log file: {log_filepath}")
    return log_filepath


class HumanActionVideoDataset(Dataset):

    def __init__(self, video_dirs, num_frames=16, transforms=None, class_cap_per_dir=None, frame_sample_rate=4,
                 dataset=""):
        self.video_dirs = video_dirs if isinstance(video_dirs, list) else [video_dirs]
        self.num_frames = num_frames
        self.transforms = transforms
        self.frame_sample_rate = frame_sample_rate
        self.dataset = dataset
        self.resize = T.Resize((480, 640))
        self.class_cap_per_dir = class_cap_per_dir or {}
        self.classes, self.filepaths = self._load_video_paths()

        if dataset == "vivit":
            self.resize = T.Resize((224, 224))
            self.to_tensor = T.ToTensor()

    def _load_video_paths(self):
        classes = set()
        filepaths = []
        for video_dir in self.video_dirs:
            print(f"loading dir: {video_dir}")
            current_classes = sorted([d.name for d in Path(video_dir).iterdir() if d.is_dir()])
            classes.update(current_classes)
            for cls in current_classes:
                class_videos = os.listdir(os.path.join(video_dir, cls))
                filepaths.extend([(os.path.join(video_dir, cls, video), cls) for video in class_videos])
        return sorted(classes), filepaths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        if self.dataset == "vivit":
            video_path, label = self.filepaths[idx]

            try:
                video, _, _ = read_video(video_path, pts_unit="sec")

            except Exception as e:
                print(f"error reading video at {video_path}, {e}")
                dummy_frame = torch.zeros((self.num_frames, 3, 224, 224))
                return dummy_frame, -1

            if video.numel() == 0:
                print(f"empty video at {video_path}, skipping...")
                dummy_frame = torch.zeros((self.num_frames, 3, 224, 224))
                return dummy_frame, -1

            video = video.permute(0, 3, 1, 2)
            total_frames = video.shape[0]

            indices = sample_frame_indices_vivit(self.num_frames, self.frame_sample_rate, total_frames)
            indices = indices[indices < total_frames]
            frames = video[indices]

            frames = torch.stack([self.resize(frame) for frame in frames])

            if self.transforms:
                frames = self.transforms(frames)

            label_idx = self.classes.index(label)
            return frames, label_idx

        video_path, label = self.filepaths[idx]
        video, _, _ = read_video(video_path, pts_unit="sec")

        if video.numel() == 0:
            print(f"empty video at {video_path}, skipping...")
            dummy_frame = torch.zeros((self.num_frames, 3, 480, 640))
            return dummy_frame, -1

        video = video.permute(0, 3, 1, 2)

        total_frames = video.shape[0]
        frame_indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        if total_frames < self.num_frames:
            frame_indices = torch.linspace(0, total_frames - 1, total_frames).long()

        video = video[frame_indices]
        video = torch.stack([self.resize(frame) for frame in video])

        if self.transforms:
            video = self.transforms(video)

        label_idx = self.classes.index(label)
        return video, label_idx


resize_transform = T.Resize((480, 640))


def preprocess_video(videos, processor, device):
    batch_size, num_frames, channels, height, width = videos.shape

    if height != 480 or width != 640:
        print(f"resizing from {height}x{width} to 480x640")
        videos = torch.stack([resize_transform(frame) for frame in videos.view(-1, channels, height, width)])
        videos = videos.view(batch_size, num_frames, channels, 480, 640)

    frames_list = videos.view(-1, channels, 480, 640).unbind(0)
    processed = processor(images=list(frames_list), return_tensors="pt")

    num_channels = processed["pixel_values"].shape[-3]
    new_height = processed["pixel_values"].shape[-2]
    new_width = processed["pixel_values"].shape[-1]

    inputs = {
        k: v.view(batch_size, num_frames, num_channels, new_height, new_width).to(device)
        for k, v in processed.items()
    }

    return inputs


def main():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--baseline_model", type=str, default="MCG-NJU/videomae-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--real_test_size", type=int, default=50)
    parser.add_argument("--training_sources", type=str, nargs="+", default=["real", "white", "background"])

    parser.add_argument("--dataset_dir__real", type=str, default="./RANDOM-People/real")
    parser.add_argument("--class_cap_per_dir__real", type=int, default=50)
    parser.add_argument("--dataset_dir__background", type=str, default="./RANDOM-People/background")
    parser.add_argument("--class_cap_per_dir__background", type=int, default=225)
    parser.add_argument("--dataset_dir__white", type=str, default="./RANDOM-People/white")
    parser.add_argument("--class_cap_per_dir__white", type=int, default=100)

    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"])

    args = parser.parse_args()

    set_seed(args.seed)

    log_filepath = setup_logging(args.experiment_name)
    print(f"parsed args: {args}")

    class_cap_per_dir = {
        args.dataset_dir__real: args.class_cap_per_dir__real,
        args.dataset_dir__background: args.class_cap_per_dir__background,
        args.dataset_dir__white: args.class_cap_per_dir__white,
    }

    num_frames = 16
    frame_sample_rate = None
    load_ds = None

    if args.baseline_model in ["MCG-NJU/videomae-base", "facebook/timesformer-base-finetuned-k400"]:
        num_frames = 16
        frame_sample_rate = None
        load_ds = None

    elif args.baseline_model in ["google/vivit-b-16x2-kinetics400"]:
        num_frames = 32
        frame_sample_rate = 4
        load_ds = "vivit"

    customToyota_dataset = HumanActionVideoDataset([args.dataset_dir__real], num_frames=num_frames, frame_sample_rate=frame_sample_rate, dataset=load_ds)
    remaining_filepaths, test_filepaths = select_fixed_test_set(customToyota_dataset.filepaths, args.real_test_size)
    train_filepaths = apply_class_cap(remaining_filepaths, class_cap_per_dir, args.dataset_dir__real)
    train_dataset = Subset(customToyota_dataset, [customToyota_dataset.filepaths.index(fp) for fp in train_filepaths])
    test_dataset = Subset(customToyota_dataset, [customToyota_dataset.filepaths.index(fp) for fp in test_filepaths])

    print(f"customToyota split into {len(train_dataset)} train and {len(test_dataset)} test sample")

    datasets = []
    if "real" in args.training_sources:
        datasets.append(train_dataset)

    if "background" in args.training_sources:
        videos_foldered_dataset = HumanActionVideoDataset([args.dataset_dir__background], num_frames=num_frames, class_cap_per_dir=class_cap_per_dir, frame_sample_rate=frame_sample_rate, dataset=load_ds)
        datasets.append(videos_foldered_dataset)

    if "white" in args.training_sources:
        random_people_dataset = HumanActionVideoDataset([args.dataset_dir__white], num_frames=num_frames, class_cap_per_dir=class_cap_per_dir, frame_sample_rate=frame_sample_rate, dataset=load_ds)
        datasets.append(random_people_dataset)

    train_dataset = torch.utils.data.ConcatDataset(datasets)
    print(f"combined training dataset size: {len(train_dataset)}")

    if args.baseline_model == "MCG-NJU/videomae-base":
        processor = VideoMAEImageProcessor.from_pretrained(args.baseline_model)
        model = VideoMAEForVideoClassification.from_pretrained(args.baseline_model, num_labels=16)

    elif args.baseline_model == "facebook/timesformer-base-finetuned-k400":
        processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400", num_labels=16, ignore_mismatched_sizes=True)

    elif args.baseline_model == "google/vivit-b-16x2-kinetics400":
        processor = VivitImageProcessor.from_pretrained(args.baseline_model)
        model = VivitForVideoClassification.from_pretrained(args.baseline_model, num_labels=16, ignore_mismatched_sizes=True)

    else:
        print("model not supported")
        return

    device = torch.device(args.device)
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for videos, labels in train_loader:
            if (labels == -1).any():
                continue
            videos, labels = videos.to(device), labels.to(device)
            if load_ds == "vivit":
                inputs = preprocess_video_vivit(videos, processor, device)
            else:
                inputs = preprocess_video(videos, processor, device)
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"epoch {epoch + 1}/{args.epochs} - loss: {running_loss / len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels in test_loader:
            if (labels == -1).any():
                continue
            videos, labels = videos.to(device), labels.to(device)
            if load_ds == "vivit":
                inputs = preprocess_video_vivit(videos, processor, device)
            else:
                inputs = preprocess_video(videos, processor, device)
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"testing accuracy: {accuracy:.2f}%")
    print("training and evaluation complete")


if __name__ == "__main__":
    main()
