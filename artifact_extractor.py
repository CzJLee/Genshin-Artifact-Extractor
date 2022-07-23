import cv2
import pathlib
import numpy as np
import json
import time
import shutil
import pytesseract
import string
import tempfile
from fastprogress import fastprogress

# MAGIC NUMBERS
roi = (1217, 153, 612, 1135)

rois = {
    "artifact_name" :   (  28,   77,  365,   46),
    "type" :            (  26,  183,  254,   37),
    "value" :           (  26,  219,  208,   62),
    "level" :           (  38,  385,   65,   30),
    "rarity":           (  26,  290,  254,   45),
    "substats_4" :      (  59,  440,  428,  192),
    "set_name_4" :      (  26,  629,  504,   53),
    "substats_3" :      (  59,  440,  428,  145),
    "set_name_3" :      (  26,  581,  504,   53),
    "equipped" :        ( 100, 1071,  511,   63)
}

# Constants
light_text = ["artifact_name", "level", "type", "value"]
dark_text = ["equipped", "set_name_3", "set_name_4", "substats_3", "substats_4"]

whitelist = set(string.ascii_letters + string.digits + string.whitespace + ".,+%\':")

with open("ArtifactInfo.json") as f:
    artifact_info = json.loads(f.read())

class Artifact():
    def __init__(self):
        self.artifact_name = None
        self.level = None
        self.type = None
        self.value = None
        self.set_name = None
        self.substats = None
        self.equipped = None

        self.file_path = None

    def to_dict(self):
        return {
            "artifact_name" : self.artifact_name,
            "level" : self.level,
            "type" : self.type,
            "value" : self.value,
            "set_name" : self.set_name,
            "substats" : self.substats,
            "equipped" : self.equipped
        }

def video_to_frames(input_file_path, output_dir, verbose = True):
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
    cap = cv2.VideoCapture(str(input_file_path))
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    if verbose:
        print ("Number of frames: ", video_length)
    count = 0
    if verbose:
        print ("Converting video..\n")
    # Start converting the video
    success, frame = cap.read()

    while success:
        # Extract the frame
        # ret, frame = cap.read()
        # if not ret:
        #     # Continue if bad frame
        #     count = count + 1

        #     continue
        # Write the results back to output location.
        if verbose:
            # print(output_dir / f"{(count+1):0>4d}.jpg")
            print(f"Reading frame {(count+1):0>4d} / {video_length}")
        frame = crop_roi(frame, roi)
        cv2.imwrite(str(output_dir / f"{(count+1):0>4d}.jpg"), frame)
        count = count + 1
        success, frame = cap.read()

    # # If there are no more frames left
    # if (count > (video_length-1)):
    # Log the time again
    time_end = time.time()
    # Release the feed
    cap.release()
    # Print stats
    if verbose:
        print ("Done extracting frames.\n%d frames extracted" % count)
        print ("It took %d seconds for conversion." % (time_end-time_start))
    # break

def crop_roi(image, roi):
    return image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

def crop_frames(frames_dir, output_dir):
    # Crop all images
    frames_dir = pathlib.Path(frames_dir)
    output_dir = pathlib.Path(output_dir)
    for img_path in sorted(frames_dir.iterdir()):
        if img_path.suffix == ".jpg":
            print(img_path.name)
            image = cv2.imread(str(img_path))
            cropped_img = crop_roi(image, roi)
            cv2.imwrite(str(output_dir / img_path.name), cropped_img)

def remove_duplicate_frames(cropped_frames_dir, output_dir):
    output_dir = pathlib.Path(output_dir)
    hashes = set()
    valid_frames = []

    cropped_frames_dir = pathlib.Path(cropped_frames_dir)
    for img_path in sorted(cropped_frames_dir.iterdir()):
        if img_path.suffix == ".jpg":
            image = cv2.imread(str(img_path))
            img_hash = cv2.img_hash.pHash(image)
            img_hash = tuple(img_hash[0])

            if img_hash not in hashes:
                valid_frames.append(img_path)
                hashes.add(img_hash)
                print(f"{img_path.name} : {img_hash}")

    print(f"Found {len(hashes)} artifacts")

    for img_path in valid_frames:
        shutil.copy(img_path, output_dir / img_path.name)

def get_artifact_components(frames_dir, output_dir):
    # Crop all images
    frames_dir = pathlib.Path(frames_dir)
    output_dir = pathlib.Path(output_dir)
    for img_path in sorted(frames_dir.iterdir()):
        if img_path.suffix == ".jpg":
            print(img_path.name)
            image = cv2.imread(str(img_path))
            img_crop_dir = output_dir / img_path.stem
            img_crop_dir.mkdir(exist_ok=True, parents=True)
            for key, roi in rois.items():
                cropped_img = crop_roi(image, roi)
                cv2.imwrite(str(img_crop_dir / f"{key}.jpg"), cropped_img)

def get_rarity_blob_detector():
    # We are looking for star shapes
    # Set up the detector with default parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Filter by occupancy Threshold
    params.minThreshold = 200
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.1
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)
    return detector

def process_rarity(file_path):
    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    pad = 20
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, borderType = cv2.BORDER_CONSTANT, value=255)
    # Dilating helps with light text
    kernel = np.ones((3,3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    
    # Detect blobs.
    detector = get_rarity_blob_detector()
    keypoints = detector.detect(image)
    return len(keypoints)

def process_light(file_path):
    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    pad = 20
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, borderType = cv2.BORDER_CONSTANT, value=255)
    # Dilating helps with light text
    # kernel = np.ones((2,2), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)
    cv2.imwrite(str(file_path.parent / f"{file_path.stem}_thresh.jpg"), image)
    return image

def process_dark(file_path):
    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

    _, image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
    pad = 20
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, borderType = cv2.BORDER_CONSTANT, value=255)
    # kernel = np.ones((2,2), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)
    cv2.imwrite(str(file_path.parent / f"{file_path.stem}_thresh.jpg"), image)
    return image

def get_ocr_text(image):
    # Config: https://stackoverflow.com/questions/44619077/pytesseract-ocr-multiple-config-options
    config="--psm 6"
    ocr_text = pytesseract.image_to_string(image, lang = "genshin", config=config)
    ocr_text = ocr_text.strip()
    ocr_text = "".join(char for char in ocr_text if char in whitelist)
    return ocr_text

def ocr_artifact(artifact_dir):
    artifact_dir = pathlib.Path(artifact_dir)
    artifact_text = {}
    for artifact_component in light_text:
        processed_image = process_light(artifact_dir / (artifact_component + ".jpg"))
        ocr_text = get_ocr_text(processed_image)
        artifact_text[artifact_component] = ocr_text

    for artifact_component in dark_text:
        processed_image = process_dark(artifact_dir / (artifact_component + ".jpg"))
        ocr_text = get_ocr_text(processed_image)
        artifact_text[artifact_component] = ocr_text

    rarity = process_rarity(artifact_dir / ("rarity" + ".jpg"))
    artifact_text["rarity"] = rarity

    # Split substats newlines into list (if double new line, don't keep a blank line)
    artifact_text["substats_3"] = [x for x in artifact_text["substats_3"].split("\n") if x]
    artifact_text["substats_4"] = [x for x in artifact_text["substats_4"].split("\n") if x]
    return artifact_text

def extract_text_dir(ocr_dir, verbose = True):
    ocr_dir = pathlib.Path(ocr_dir)
    artifacts = {}
    for artifact_dir in sorted(ocr_dir.iterdir()):
        if artifact_dir.is_dir():
            artifact_text = ocr_artifact(artifact_dir)
            artifacts[artifact_dir.stem] = artifact_text
            if verbose:
                print(artifact_text)

    return artifacts

def write_json(artifacts, save_file_path = "artifacts.json"):
    save_file_path = pathlib.Path(save_file_path)
    with open(save_file_path, "w") as f:
       json.dump(artifacts, f, indent=4)

verbose = True
def main(video_path, artifact_dir = None):
    video_path = pathlib.Path(video_path)

    if artifact_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        artifact_dir = pathlib.Path(temp_dir.name)
        if verbose:
            print(f"Created temporary directory at {artifact_dir}")
    else:
        artifact_dir = pathlib.Path(artifact_dir)
        artifact_dir.mkdir(exist_ok=True, parents=True)


    # Extract frames from video
    video_output_dir = artifact_dir / "video_output"
    video_output_dir.mkdir(exist_ok=True, parents=True)
    video_to_frames(video_path, video_output_dir)

    # # Crop extracted frames
    # cropped_frames_dir = temp_dir / "cropped_frames"
    # cropped_frames_dir.mkdir(exist_ok=True, parents=True)
    # crop_frames(video_output_dir, cropped_frames_dir)

    valid_frames_dir = artifact_dir / "valid_frames"
    valid_frames_dir.mkdir(exist_ok=True, parents=True)
    remove_duplicate_frames(video_output_dir, valid_frames_dir)

    ocr_dir = artifact_dir / "artifacts"
    ocr_dir.mkdir(exist_ok=True, parents=True)
    get_artifact_components(valid_frames_dir, ocr_dir)

    artifacts = extract_text_dir(ocr_dir)
    write_json(artifacts)

if __name__ == "__main__":
    main("artifacts.mp4", artifact_dir="artifacts")


    # get_artifact_components("artifacts/valid_frames", "artifacts/artifacts")

    # artifacts = extract_text_dir("artifacts/artifacts")
    # write_json(artifacts)

    # ocr_dir = pathlib.Path("artifacts/artifacts/")

    # artifacts = extract_text_dir(ocr_dir)
    # write_json(artifacts)

    # test_artifact = {'artifact_name': 'Flower of Life', 'level': '+0', 'type': 'HP', 'value': 'Wi', 'equipped': 'er to make wishes come true.', 'set_name_3': "Shimenawa's Reminiscence:", 'set_name_4': ' 2Piece Set: ATK +18%,', 'substats_3': 'Energy Recharge+5.8%\nCRIT DMG+5.4%\nATK+19', 'substats_4': "Energy Recharge+5.8%\nCRIT DMG+5.4%\n\nATK+19\n\n1imenawa's Reminiscence:"}
    # write_json(test_artifact)
    # test_artifact = ocr_artifact("artifacts/artifacts/1460")
    # write_json(test_artifact)
    # print(test_artifact)

