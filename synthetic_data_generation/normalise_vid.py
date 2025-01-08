import cv2
import numpy as np
import os
import argparse

def speed_up_video(root_path):
    input_video_path = os.path.join(root_path, 'video.mp4')
    temp_video_path = root_path + '.temp.mp4'  # Temporary file for processing

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / original_fps

    # Set the desired output frame rate
    if original_fps <= 25:
        output_fps = original_fps
    else:
        output_fps = 25

    # Calculate the total number of frames for the output video (18 seconds * 18 fps)
    max_duration = 20  # seconds
    desired_frame_count = int(output_fps * max_duration)

    # Check if video needs to be sped up or just adjust the frame rate
    if duration <= max_duration:
        # Video is shorter than or equal to 18 seconds; adjust frame rate
        print(f"Video duration is {duration:.2f} seconds. No adjusment needed")

        # Prepare to write the output video with new frame rate
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #out = cv2.VideoWriter(temp_video_path, fourcc, output_fps, (width, height))

        #while True:
        #    ret, frame = cap.read()
        #    if not ret:
        #        break
        #    out.write(frame)

        # Release resources
        #cap.release()
        #out.release()

        # Replace the original video with the processed video
        #os.replace(temp_video_path, input_video_path)
        #print(f"Video frame rate adjusted and saved at {input_video_path}")

    else:
        # Video is longer than 18 seconds; need to speed up
        print(f"Video duration is {duration:.2f} seconds. Speeding up to {max_duration} seconds and adjusting frame rate to {output_fps} fps.")

        # Generate frame indices to sample frames evenly across the original video
        frame_indices = np.linspace(0, frame_count - 1, num=desired_frame_count, endpoint=True)
        frame_indices = set(frame_indices.astype(int))

        # Prepare to write the output video
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

        # Release resources
        cap.release()
        out.release()

        # Replace the original video with the processed video
        os.replace(temp_video_path, input_video_path)
        print(f"Video has been sped up and saved at {input_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Speed up video if longer than 18 seconds and adjust frame rate to 18 fps.')
    parser.add_argument('--root_path', type=str, required=True, help='Path to the video file.')
    args = parser.parse_args()

    speed_up_video(args.root_path)
