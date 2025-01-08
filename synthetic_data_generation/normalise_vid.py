import cv2
import numpy as np
import os
import argparse

def speed_up_video(root_path):
    input_video_path = os.path.join(root_path, 'video.mp4')
    temp_video_path = root_path + '.temp.mp4'  


    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return


    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / original_fps


    if original_fps <= 25:
        output_fps = original_fps
    else:
        output_fps = 25


    max_duration = 20  # seconds
    desired_frame_count = int(output_fps * max_duration)


    if duration <= max_duration:

        print(f"Video duration is {duration:.2f} seconds. No adjusment needed")

    else:

        print(f"Video duration is {duration:.2f} seconds. Speeding up to {max_duration} seconds and adjusting frame rate to {output_fps} fps.")


        frame_indices = np.linspace(0, frame_count - 1, num=desired_frame_count, endpoint=True)
        frame_indices = set(frame_indices.astype(int))


        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, output_fps, (width, height))

        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame in frame_indices:
                out.write(frame)
            current_frame += 1


        cap.release()
        out.release()


        os.replace(temp_video_path, input_video_path)
        print(f"Video has been sped up and saved at {input_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Speed up video if longer than 18 seconds and adjust frame rate to 18 fps.')
    parser.add_argument('--root_path', type=str, required=True, help='Path to the video file.')
    args = parser.parse_args()

    speed_up_video(args.root_path)
