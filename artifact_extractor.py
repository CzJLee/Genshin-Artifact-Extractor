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
from tqdm.auto import tqdm
import concurrent.futures
import shutil
from matplotlib import pyplot as plt
import utils
import artifact
import dataclasses
import constants
import datetime

from typing import Union, Any, Optional

PathLike = str | os.PathLike

write_json = utils.write_json
load_json = utils.load_json

ROI = constants.ROI
ROIS = constants.ROIS
LIGHT_TEXT = constants.LIGHT_TEXT
DARK_TEXT = constants.DARK_TEXT
WHITELIST = constants.WHITELIST


def copy_tesseract_font(
    source: PathLike = "genshin.traineddata",
    destination: PathLike = "/opt/homebrew/share/tessdata/genshin.traineddata",
) -> None:
    """Copy Tesseract font training data to its required directory incase it is missing."""
    shutil.copy(source, destination)


def extract_video_frames(
    input_video_path: PathLike, output_dir: PathLike, verbose=True
) -> None:
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_video_path: Input video file.
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
    cap = cv2.VideoCapture(str(input_video_path))
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    if verbose:
        print("Number of frames: ", video_length)
    count = 0
    if verbose:
        print("Converting video..")
    # Start converting the video
    success, frame = cap.read()

    with tqdm(total=video_length) as pbar:
        while success:
            frame = crop_roi(frame, ROI)
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
        print(f"Done extracting frames.\n {count} frames extracted")
        print(f"It took {(time_end - time_start)} seconds for conversion.")
    # break


def crop_roi(image: np.ndarray, roi: constants.CropROI) -> np.ndarray:
    return image[
        int(roi.top) : int(roi.top + roi.height),
        int(roi.left) : int(roi.left + roi.width),
    ]


def _get_img_hash(img_path: PathLike, img_hash_algorithm=cv2.img_hash.pHash) -> tuple:
    image = cv2.imread(str(img_path))
    # Try cropping the image block to focus on the text difference.
    image = image[70:670, :]
    img_hash = img_hash_algorithm(image)
    img_hash = tuple(img_hash[0])
    return img_hash


def remove_duplicate_frames(
    cropped_frames_dir: PathLike, output_dir: PathLike, verbose: bool = True
) -> None:
    """
    Use image hashes to remove duplicate frames.

    Args:
        cropped_frames_dir (PathLike): _description_
        output_dir (PathLike): _description_
        verbose (bool, optional): _description_. Defaults to True.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_frames = []

    previous_hash = None
    cropped_frames_dir = pathlib.Path(cropped_frames_dir)
    if verbose:
        print("Removing duplicate frames...")
    for img_path in tqdm(sorted(cropped_frames_dir.iterdir())):
        if img_path.suffix == ".jpg":
            img_hash = _get_img_hash(img_path)

            # Eliminate frames if the current img hash is equal to the previous hash.
            if img_hash != previous_hash:
                valid_frames.append(img_path)
                previous_hash = img_hash

    print(f"Found {len(valid_frames)} artifacts")

    for img_path in valid_frames:
        shutil.copy(img_path, output_dir / img_path.name)


def get_artifact_components(
    frames_dir: PathLike, output_dir: PathLike, verbose: bool = True
) -> None:
    # Crop all images
    frames_dir = pathlib.Path(frames_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Getting artifact component crops...")
    for img_path in tqdm(sorted(frames_dir.iterdir())):
        if img_path.suffix == ".jpg":
            image = cv2.imread(str(img_path))
            img_crop_dir = output_dir / img_path.stem
            img_crop_dir.mkdir(exist_ok=True, parents=True)
            for key, roi in ROIS.items():
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
    image = cv2.copyMakeBorder(
        image, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=255
    )
    # Dilating helps with light text
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)

    # Detect blobs.
    detector = _get_rarity_blob_detector()
    keypoints = detector.detect(image)
    return len(keypoints)


def _process_light_text(file_path: PathLike) -> np.ndarray:
    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    pad = 20
    image = cv2.copyMakeBorder(
        image, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=255
    )
    # Dilating helps with light text
    # kernel = np.ones((2,2), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)
    cv2.imwrite(str(file_path.parent / f"{file_path.stem}_thresh.jpg"), image)
    return image


def _process_dark_text(file_path: PathLike) -> np.ndarray:
    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

    _, image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
    pad = 20
    image = cv2.copyMakeBorder(
        image, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=255
    )
    # kernel = np.ones((2,2), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)
    cv2.imwrite(str(file_path.parent / f"{file_path.stem}_thresh.jpg"), image)
    return image


def _get_ocr_text(image: np.ndarray) -> str:
    # Config: https://stackoverflow.com/questions/44619077/pytesseract-ocr-multiple-config-options
    config = "--psm 6"
    ocr_text = pytesseract.image_to_string(image, lang="genshin", config=config)
    ocr_text = ocr_text.strip()
    ocr_text = "".join(char for char in ocr_text if char in WHITELIST)
    return ocr_text


def ocr_artifact(artifact_dir: PathLike) -> dict[str, Union[str, list]]:
    artifact_dir = pathlib.Path(artifact_dir)
    artifact_text = {}
    artifact_text["artifact_id"] = artifact_dir.stem
    for artifact_component in LIGHT_TEXT:
        processed_image = _process_light_text(
            artifact_dir / (artifact_component + ".jpg")
        )
        ocr_text = _get_ocr_text(processed_image)
        artifact_text[artifact_component] = ocr_text

    for artifact_component in DARK_TEXT:
        processed_image = _process_dark_text(
            artifact_dir / (artifact_component + ".jpg")
        )
        ocr_text = _get_ocr_text(processed_image)
        artifact_text[artifact_component] = ocr_text

    rarity = process_rarity(artifact_dir / ("rarity" + ".jpg"))
    artifact_text["rarity"] = rarity

    # Split substats newlines into list (if double new line, don't keep a blank line)
    artifact_text["substats_3"] = [
        x for x in artifact_text["substats_3"].split("\n") if x
    ]
    artifact_text["substats_4"] = [
        x for x in artifact_text["substats_4"].split("\n") if x
    ]
    return artifact_text


def run_ocr_on_artifact_components(
    artifact_component_dir: PathLike, ocr_output_dir: PathLike, verbose: bool = True
) -> dict[str, dict[str, list]]:
    if verbose:
        print("Running OCR...")

    artifact_component_dir = pathlib.Path(artifact_component_dir)
    artifact_component_dirs = sorted(artifact_component_dir.iterdir())
    ocr_output_dir = pathlib.Path(ocr_output_dir)
    ocr_output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {}
    for artifact_dir in tqdm(artifact_component_dirs):
        if artifact_dir.is_dir():
            artifact_text = ocr_artifact(artifact_dir)
            artifacts[artifact_dir.stem] = artifact_text

    write_json(artifacts, ocr_output_dir / "artifacts.json")
    return artifacts


def run_ocr_on_artifact_components_multiprocess(
    artifact_component_dir: PathLike, ocr_output_dir: PathLike, verbose=True
) -> dict[str, dict[str, list]]:
    if verbose:
        print("Running OCR...")

    artifact_component_dir = pathlib.Path(artifact_component_dir)
    artifact_component_dirs = sorted(artifact_component_dir.iterdir())
    ocr_output_dir = pathlib.Path(ocr_output_dir)
    ocr_output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {}
    artifact_dirs = [dir for dir in artifact_component_dirs if dir.is_dir()]
    progress_bar = tqdm(total=len(artifact_dirs))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for artifact_dir in artifact_dirs:
            future = executor.submit(ocr_artifact, artifact_dir)
            # Add callback up update tqdm progress bar.
            future.add_done_callback(lambda _: progress_bar.update())
            futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            artifact_text = future.result()
            artifacts[artifact_text["artifact_id"]] = artifact_text

    write_json(artifacts, ocr_output_dir / "artifacts.json")
    return artifacts


def replace_artifacts(
    gi_data_path: PathLike,
    all_artifacts_json="artifacts_good_format.json",
    updated_gi_data_path="gi_data_updated.json",
    verbose=True,
) -> None:
    gi_data = load_json(gi_data_path)
    all_artifacts = load_json(all_artifacts_json)

    # Replace data
    gi_data["artifacts"] = all_artifacts["artifacts"]

    write_json(gi_data, updated_gi_data_path)

    if verbose:
        print(f"Created updated GI Database: {updated_gi_data_path}")


def remove_duplicate_artifacts(
    artifacts: dict[str, dict[str, list]]
) -> list[artifact.Artifact]:
    all_artifacts = []
    previous_artifact = None
    for _id, ocr_json in sorted(artifacts.items()):
        try:
            next_artifact = artifact.Artifact.from_ocr_json(ocr_json)
        except artifact.ArtifactError:
            print(_id)
            raise
        if next_artifact != previous_artifact:
            all_artifacts.append(next_artifact)
            previous_artifact = next_artifact

    return all_artifacts


def locate_template(
    image: np.ndarray,
    template_image_path: PathLike = constants.TEMPLATE_IMAGE_PATH,
) -> tuple[int, int]:
    """Find the pixel coordinates of a recognized +0 icon in an image of a new artifact.

    Args:
        image: Image of a new artifact.
        template_image_path: Path to the +0 template image.

    Returns:
        Top left corner pixel coordinates of the recognized +0 icon.
    """
    template = cv2.imread(str(template_image_path))
    res = cv2.matchTemplate(image, template, method=cv2.TM_CCOEFF)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    return max_loc


def resize_artifact(
    image: np.ndarray,
    artifact_expected_width: int = constants.ARTIFACT_EXPECTED_WIDTH,
) -> np.ndarray:
    """Resizes an artifact image crop to the expected width for OCR to work."""
    # Only width matters.
    scaling_factor = artifact_expected_width / image.shape[1]
    new_height = int(image.shape[0] * scaling_factor)
    return cv2.resize(
        image, (artifact_expected_width, new_height), interpolation=cv2.INTER_CUBIC
    )


def locate_and_crop_template(image: artifact.ArtifactImage) -> np.ndarray:
    """
    Locate the +0 template in an artifact image and crop the image to the Artifact box.

    Args:
        image (artifact.ArtifactImage): Artifact Image representing a new artifact.

    Raises:
        ValueError: If the image dimensions are unrecognized.

    Returns:
        np.ndarray: Cropped and resized artifact crop image.
    """
    # Get image coordinates to determine with template match to use.
    image_dimensions = f"{image.width}x{image.height}"
    template_image_path = pathlib.Path("templates") / f"{image_dimensions}.png"

    if not template_image_path.exists():
        raise ValueError(
            f"No template found for Artifact image with dimensions {image_dimensions}"
        )

    match_w, match_h = locate_template(image.image, template_image_path)

    # Lookup dict to get the correct crop ROI offset for the image dimensions.
    offset = constants.OFFSET_FOR_IMAGE_DIMENSIONS[(image.width, image.height)]
    roi = constants.CropROI(
        left=match_w - offset.horizontal_offset,
        top=match_h - offset.vertical_offset,
        width=offset.artifact_width,
        height=offset.artifact_height,
    )

    # Crop the image.
    image_crop = crop_roi(image.image, roi)

    # Resize to expected dimensions.
    return resize_artifact(image_crop)


# def crop_from_template_match(
#     image: np.ndarray, template_coordinates: tuple[int, int]
# ) -> np.ndarray:
#     """Given the coordinates of a recognized +0 icon, crop the artifact image."""
#     match_w, match_h = template_coordinates
#     artifact_left = match_w - constants.HORIZONTAL_OFFSET
#     artifact_top = match_h - constants.VERTICAL_OFFSET

#     artifact_width = 874
#     artifact_height = 1731

#     roi = constants.CropROI(
#         left=artifact_left,
#         top=artifact_top,
#         width=artifact_width,
#         height=artifact_height,
#     )

#     return crop_roi(image, roi)


def crop_new_artifact(artifact_image_path: PathLike, output_dir: PathLike) -> None:
    """
    Read a new artifact image from disk and generate a cropped image.

    Args:
        artifact_image_path: Path to the new artifact image.
        output_dir: Directory to write cropped image.
    """
    artifact_image_path = pathlib.Path(artifact_image_path)
    image = artifact.ArtifactImage(artifact_image_path)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    image_crop = locate_and_crop_template(image)

    # TODO: Modify new file creation time.

    # Get image file creation time.
    creation_timestamp = pathlib.Path(artifact_image_path).stat().st_birthtime
    creation_time = datetime.datetime.fromtimestamp(creation_timestamp)

    # Format file name simply by tacking on creation time at the end.
    new_image_name = (
        artifact_image_path.stem + " "
        + creation_time.strftime(constants.STRTIME_FORMAT)
        + ".jpg"
    )
    new_image_path = output_dir / new_image_name
    cv2.imwrite(str(new_image_path), image_crop)


def crop_new_artifacts_multiprocess(
    artifact_dir: PathLike, output_dir: PathLike
) -> None:
    """Runs crop_new_artifact as a concurrent multiprocess for an entire directory of new artifacts.

    Args:
        artifact_dir: Directory containing new artifact images.
        output_dir: Directory to write cropped images.
    """
    artifact_image_paths = list(pathlib.Path(artifact_dir).iterdir())

    progress_bar = tqdm(total=len(artifact_image_paths))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for artifact_path in artifact_image_paths:
            future = executor.submit(crop_new_artifact, artifact_path, output_dir)
            # Add callback up update tqdm progress bar.
            future.add_done_callback(lambda _: progress_bar.update())
            futures.append(future)

        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)


def main(
    download_dir: str,
    video_file_name: str = "artifacts.MOV",
    artifact_dir: Optional[PathLike] = None,
    artifact_dir_name: str = "_artifact_temp",
) -> None:
    video_file_name = pathlib.Path(video_file_name)

    # Create a directory to store intermediate results.
    # If program crashes, try to detect existing directory to continue where it crashed.
    if artifact_dir is None:
        artifact_dir = pathlib.Path().cwd() / artifact_dir_name
    else:
        artifact_dir = pathlib.Path(artifact_dir) / artifact_dir_name
    artifact_dir.mkdir(exist_ok=True, parents=True)

    video_output_dir_name = "video_output"
    video_output_dir = artifact_dir / video_output_dir_name

    valid_frames_dir_name = "valid_frames"
    valid_frames_dir = artifact_dir / valid_frames_dir_name

    artifact_components_dir_name = "artifact_components"
    artifact_components_dir = artifact_dir / artifact_components_dir_name

    ocr_output_dir_name = "ocr_output"
    ocr_output_dir = artifact_dir / ocr_output_dir_name

    # Extract each video frame and crop the Artifact region.
    if not valid_frames_dir.exists():
        # The next step directory will exist if this step has been completed.
        extract_video_frames(video_file_name, video_output_dir)

    # Remove duplicate frames using image hashes.
    if not artifact_components_dir.exists():
        # The next step directory will exist if this step has been completed.
        remove_duplicate_frames(video_output_dir, valid_frames_dir)

    # Get artifact component crops
    if not ocr_output_dir.exists():
        # The next step directory will exist if this step has been completed.
        get_artifact_components(valid_frames_dir, artifact_components_dir)

    # Run OCR
    if not (ocr_output_dir / "artifacts.json").exists():
        artifacts = run_ocr_on_artifact_components_multiprocess(
            artifact_components_dir, ocr_output_dir
        )

    # Remove duplicate artifacts
    artifacts = load_json(ocr_output_dir / "artifacts.json")
    all_artifacts = remove_duplicate_artifacts(artifacts=artifacts)
    print(f"Found {len(all_artifacts)} total artifacts")

    # Save artifacts to good format
    artifact.artifact_list_to_good_format_json(
        all_artifacts, output_path="artifacts_good_format.json"
    )

    # Find most recently downloaded GI Database
    gi_data_path = get_most_recent_gi_database(download_dir)
    # Replace artifacts and write to updated_gi_data_path
    replace_artifacts(
        gi_data_path=gi_data_path,
        all_artifacts_json="artifacts_good_format.json",
        updated_gi_data_path="gi_data_updated.json",
    )

    # shutil.rmtree(artifact_dir)


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
    copy_tesseract_font()
    download_dir = pathlib.Path("~/Downloads").expanduser()

    # Run artifact extractor on artifacts.MOV located in download directory
    main(
        download_dir=download_dir,
        video_file_name=download_dir / "artifacts.MOV",
    )
