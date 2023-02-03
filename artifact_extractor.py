import cv2
import pathlib
import numpy as np
import time
import shutil
import pytesseract
import string
import tempfile
import re
import os
from tqdm import tqdm

import artifact

from typing import Dict, List, Union, Any, Optional
PathLike = Union[str, os.PathLike]

from utils import write_json, load_json

# MAGIC NUMBERS
roi = (1217, 153, 612, 1135)

# Crop Region of Interest for iPad
rois = {
    "artifact_type" :   (  28,   77,  320,   46),
    "main_stat" :       (  26,  183,  300,   37),
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
light_text = ["artifact_type", "level", "main_stat", "value"]
dark_text = ["equipped", "set_name_3", "set_name_4", "substats_3", "substats_4"]

whitelist = set(string.ascii_letters + string.digits + string.whitespace + ".,+-%\':")



def video_to_frames(input_file_path: PathLike, output_dir: PathLike, verbose = True) -> None:
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_file_path: Input video file.
        output_dir: Output directory to save the frames.
    Returns:
        None
    """
    try:
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
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
        print ("Converting video..")
    # Start converting the video
    success, frame = cap.read()

    with tqdm(total = video_length) as pbar:
        while success:
            # Extract the frame
            # ret, frame = cap.read()
            # if not ret:
            #     # Continue if bad frame
            #     count = count + 1

            #     continue
            # Write the results back to output location.
            # if verbose:
            #     # print(output_dir / f"{(count+1):0>4d}.jpg")
            #     print(f"Reading frame {(count+1):0>4d} / {video_length}")
            frame = crop_roi(frame, roi)
            cv2.imwrite(str(output_dir / f"{(count+1):0>4d}.jpg"), frame)
            count = count + 1
            pbar.update()
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

def crop_roi(image: np.ndarray, roi: List[int]) -> np.ndarray:
    return image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

# def crop_frames(frames_dir, output_dir):
#     # Crop all images
#     frames_dir = pathlib.Path(frames_dir)
#     output_dir = pathlib.Path(output_dir)
#     for img_path in sorted(frames_dir.iterdir()):
#         if img_path.suffix == ".jpg":
#             print(img_path.name)
#             image = cv2.imread(str(img_path))
#             cropped_img = crop_roi(image, roi)
#             cv2.imwrite(str(output_dir / img_path.name), cropped_img)

def remove_duplicate_frames(cropped_frames_dir: PathLike, output_dir: PathLike, verbose = True) -> None:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = set()
    valid_frames = []
    
    previous_hash = None
    cropped_frames_dir = pathlib.Path(cropped_frames_dir)
    if verbose:
        print("Removing duplicate frames...")
    for img_path in tqdm(sorted(cropped_frames_dir.iterdir())):
        if img_path.suffix == ".jpg":
            image = cv2.imread(str(img_path))
            img_hash = cv2.img_hash.pHash(image)
            img_hash = tuple(img_hash[0])

            # Eliminate frames if the current img hash is equal to the previous hash.
            if img_hash != previous_hash:
                valid_frames.append(img_path)
                # print(f"{img_path.name} : {img_hash}")
                previous_hash = img_hash

    print(f"Found {len(valid_frames)} artifacts")

    for img_path in valid_frames:
        shutil.copy(img_path, output_dir / img_path.name)

def get_artifact_components(frames_dir: PathLike, output_dir: PathLike, verbose = True) -> None:
    # Crop all images
    frames_dir = pathlib.Path(frames_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("Getting artifact component crops...")
    for img_path in tqdm(sorted(frames_dir.iterdir())):
        if img_path.suffix == ".jpg":
            # print(img_path.name)
            image = cv2.imread(str(img_path))
            img_crop_dir = output_dir / img_path.stem
            img_crop_dir.mkdir(exist_ok=True, parents=True)
            for key, roi in rois.items():
                cropped_img = crop_roi(image, roi)
                cv2.imwrite(str(img_crop_dir / f"{key}.jpg"), cropped_img)

def _get_rarity_blob_detector():
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

def process_rarity(file_path: PathLike) -> int:
    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    pad = 20
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, borderType = cv2.BORDER_CONSTANT, value=255)
    # Dilating helps with light text
    kernel = np.ones((3,3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    
    # Detect blobs.
    detector = _get_rarity_blob_detector()
    keypoints = detector.detect(image)
    return len(keypoints)

def _process_light_text(file_path: PathLike) -> np.ndarray:
    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    pad = 20
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, borderType = cv2.BORDER_CONSTANT, value=255)
    # Dilating helps with light text
    # kernel = np.ones((2,2), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)
    cv2.imwrite(str(file_path.parent / f"{file_path.stem}_thresh.jpg"), image)
    return image

def _process_dark_text(file_path: PathLike) -> np.ndarray:
    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

    _, image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
    pad = 20
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, borderType = cv2.BORDER_CONSTANT, value=255)
    # kernel = np.ones((2,2), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)
    cv2.imwrite(str(file_path.parent / f"{file_path.stem}_thresh.jpg"), image)
    return image

def _get_ocr_text(image: np.ndarray) -> str:
    # Config: https://stackoverflow.com/questions/44619077/pytesseract-ocr-multiple-config-options
    config="--psm 6"
    ocr_text = pytesseract.image_to_string(image, lang = "genshin", config=config)
    ocr_text = ocr_text.strip()
    ocr_text = "".join(char for char in ocr_text if char in whitelist)
    return ocr_text

def ocr_artifact(artifact_dir: PathLike) -> Dict[str, Union[str, list]]:
    artifact_dir = pathlib.Path(artifact_dir)
    artifact_text = {}
    for artifact_component in light_text:
        processed_image = _process_light_text(artifact_dir / (artifact_component + ".jpg"))
        ocr_text = _get_ocr_text(processed_image)
        artifact_text[artifact_component] = ocr_text

    for artifact_component in dark_text:
        processed_image = _process_dark_text(artifact_dir / (artifact_component + ".jpg"))
        ocr_text = _get_ocr_text(processed_image)
        artifact_text[artifact_component] = ocr_text

    rarity = process_rarity(artifact_dir / ("rarity" + ".jpg"))
    artifact_text["rarity"] = rarity

    # Split substats newlines into list (if double new line, don't keep a blank line)
    artifact_text["substats_3"] = [x for x in artifact_text["substats_3"].split("\n") if x]
    artifact_text["substats_4"] = [x for x in artifact_text["substats_4"].split("\n") if x]
    return artifact_text

def extract_text_dir(artifact_component_dir: PathLike, ocr_output_dir: PathLike, verbose = True) -> Dict[str, Dict[str, list]]:
    print("Running OCR...")
    artifact_component_dir = pathlib.Path(artifact_component_dir)
    ocr_output_dir = pathlib.Path(ocr_output_dir).mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for artifact_dir in tqdm(sorted(artifact_component_dir.iterdir())):
        if artifact_dir.is_dir():
            artifact_text = ocr_artifact(artifact_dir)
            artifacts[artifact_dir.stem] = artifact_text
            # if verbose:
            #     print(artifact_text)

    write_json(artifacts, ocr_output_dir / "artifacts.json")
    return artifacts

# def write_json(artifacts, save_file_path = "artifacts.json"):
#     save_file_path = pathlib.Path(save_file_path)
#     with open(save_file_path, "w") as f:
#        json.dump(artifacts, f, indent=4)

# def load_json(save_file_path = "artifacts.json"):
#     with open(save_file_path) as f:
#         return json.loads(f.read())

def replace_artifacts(gi_data_path: PathLike, 
                      all_artifacts_json = "artifacts_good_format.json", 
                      updated_gi_data_path = "gi_data_updated.json",
                      verbose = True
    ) -> None:

    gi_data = load_json(gi_data_path)

    all_artifacts = load_json(all_artifacts_json)

    # Replace data
    gi_data["artifacts"] = all_artifacts["artifacts"]

    write_json(gi_data, updated_gi_data_path)

    if verbose:
        print(f"Created updated GI Database: {updated_gi_data_path}")

def remove_duplicate_artifacts(artifacts: Dict[str, Dict[str, list]]) -> List[artifact.Artifact]:
    all_artifacts = []
    previous_artifact = None
    for id, ocr_json in artifacts.items():
        sample_artifact = artifact.Artifact.from_ocr_json(ocr_json)
        if sample_artifact != previous_artifact:
            all_artifacts.append(sample_artifact)
            previous_artifact = sample_artifact

    return all_artifacts

def main(video_path = "artifacts.MOV", artifact_dir: Optional[PathLike] = None, verbose = True) -> None:
    video_path = pathlib.Path(video_path)

    # Create a directory to store intermediate results.
    # If program crashes, try to detect existing directory to continue where it crashed.
    artifact_dir_name = "_artifact_temp"
    if artifact_dir is None:
        artifact_dir = pathlib.Path().cwd() / artifact_dir_name
    else:
        artifact_dir = pathlib.Path(artifact_dir) / artifact_dir_name
    artifact_dir.mkdir(exist_ok=True, parents=True)

    # if artifact_dir is None:
    #     temp_dir = tempfile.TemporaryDirectory()
    #     artifact_dir = pathlib.Path(temp_dir.name)
    #     if verbose:
    #         print(f"Created temporary directory at {artifact_dir}")
    # else:
    #     artifact_dir = pathlib.Path(artifact_dir)
    #     artifact_dir.mkdir(exist_ok=True, parents=True)

    video_output_dir_name = "video_output"
    valid_frames_dir_name = "valid_frames"
    artifact_components_dir_name = "artifact_components"
    ocr_output_dir_name = "ocr_output"


    # Extract frames from video
    if not valid_frames_dir_name.exists():
        # The next step directory will exist if this step has been completed.
        video_output_dir = artifact_dir / video_output_dir_name
        video_to_frames(video_path, video_output_dir)

    # Remove duplicate frames using image hashes
    if not artifact_components_dir_name.exists():
        # The next step directory will exist if this step has been completed.
        valid_frames_dir = artifact_dir / valid_frames_dir_name
        remove_duplicate_frames(video_output_dir, valid_frames_dir)

    # Get artifact component crops
    if not ocr_output_dir.exists():
        # The next step directory will exist if this step has been completed.
        artifact_components_dir = artifact_dir / artifact_components_dir_name
        get_artifact_components(valid_frames_dir, artifact_components_dir)

    # Run OCR
    ocr_output_dir = artifact_dir / ocr_output_dir_name
    artifacts = extract_text_dir(artifact_components_dir, ocr_output_dir)

    # Remove duplicate artifacts
    all_artifacts = remove_duplicate_artifacts(artifacts=artifacts)
    print(f"Found {len(all_artifacts)} total artifacts")

    # Save artifacts to good format
    artifact.artifact_list_to_good_format_json(all_artifacts, output_path="artifacts_good_format.json")

    # Find most recently downloaded GI Database
    gi_data_path = get_most_recent_gi_database(download_dir)
    # Replace artifacts and write to updated_gi_data_path
    replace_artifacts(gi_data_path = gi_data_path, 
        all_artifacts_json="artifacts_good_format.json", 
        updated_gi_data_path="gi_data_updated.json")

def get_most_recent_gi_database(search_dir) -> Optional[PathLike]:
    # Find most recently downloaded GI Database
    database_files = []
    for file in search_dir.iterdir():
        if re.fullmatch(r"Database_\d+_([-_\d]+).json", file.name):
            database_files.append(file)
    database_files.sort()
    if database_files:
        return database_files[-1]



if __name__ == "__main__":
    download_dir = pathlib.Path("~/Downloads").expanduser()

    # Run artifact extractor on artifacts.MOV located in download directory
    main(download_dir / "artifacts.MOV")
 