import cv2
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--use_colmap', dest='use_colmap', action='store_true')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

# get path
args = parse_args()
root_path = args.root_path

# Path to the video file
video_path = os.path.join(root_path, 'video.mp4')
print(video_path)

# Initialize the video capture
cap = cv2.VideoCapture(video_path)


# Get the total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Close the video capture
cap.release()

all_txt = os.path.join(root_path, 'frame_list_all.txt')
train_txt = os.path.join(root_path, 'frame_list_train.txt')
test_txt = os.path.join(root_path, 'frame_list_test.txt') 

# Create and write to frame_list_all.txt
with open(all_txt, 'w') as f_all:
    for i in range(total_frames):
        f_all.write(f"{i}\n")

# Create and write to frame_list_train.txt (every 5th frame)
with open(train_txt, 'w') as f_train:
    for i in range(4, total_frames, 6):  # Start from frame 4, then take every 6th frame (4, 10, 16,...)
        f_train.write(f"{i}\n")

with open(test_txt, 'w') as f_train:
    for i in range(2, total_frames, 3):  # Start from frame 4, then take every 6th frame (4, 10, 16,...)
        f_train.write(f"{i}\n")


print(f"Total frames: {total_frames}")
print("frame_list_all.txt and frame_list_train.txt generated successfully.")
