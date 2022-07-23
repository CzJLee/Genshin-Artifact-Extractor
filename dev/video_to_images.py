import cv2
import numpy as np
import pathlib
import time

artifact_video_file_path = "artifacts.mp4"
output_dir = pathlib.Path("frames/")

# vid = cv2.VideoCapture(artifact_video_file_path)
# success, image = vid.read()
# count = 0
# while success:
#     cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file      
#     success, image = vid.read()
#     print('Read a new frame: ', success)
#     count += 1

def video_to_frames(input_file_path, output_dir):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_file_path: Input video file.
        output_dir: Output directory to save the frames.
    Returns:
        None
    """
    try:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_file_path)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            # Continue if bad frame
            continue
        # Write the results back to output location.
        print(output_dir / f"{(count+1):0>4d}.jpg")
        cv2.imwrite(str(output_dir / f"{(count+1):0>4d}.jpg"), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds for conversion." % (time_end-time_start))
            break

video_to_frames(artifact_video_file_path, output_dir)
