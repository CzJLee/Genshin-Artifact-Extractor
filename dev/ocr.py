import cv2
import pytesseract
import pathlib
import numpy as np

# ocr_dir = pathlib.Path("crop/0477/")

# text1 = pytesseract.image_to_string(str(ocr_dir / "set_name_4.jpg"))
# print(text1)

# allowed_char = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+%'0123456789"

def process_light(file_path):
    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    pad = 20
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, borderType = cv2.BORDER_CONSTANT, value=255)
    # Dilating helps with light text
    kernel = np.ones((2,2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
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


light_text = ["artifact_type", "level", "type", "value"]
dark_text = ["equipped", "set_name_3", "set_name_4", "substats_3", "substats_4"]

# for file_stem in light_text:
#     processed_image = process_light(ocr_dir / (file_stem + ".jpg"))
#     print(f"{file_stem} : {pytesseract.image_to_string(processed_image, config='--psm 6')}")

# for file_stem in dark_text:
#     processed_image = process_dark(ocr_dir / (file_stem + ".jpg"))
#     print(f"{file_stem} : {pytesseract.image_to_string(processed_image, config='--psm 6')}")

# process_type(ocr_dir/ "type.jpg")
# print(pytesseract.image_to_string(process_type(ocr_dir/ "level.jpg"), config='--psm 7'))

# for file_path in ocr_dir.iterdir():
#     if file_path.suffix == ".jpg":
#         processed_image = process_type(file_path)
#         print(f"{file_path.stem} : {pytesseract.image_to_string(processed_image)}")


# def process_dark(file_path):
#     image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

#     _, image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
#     pad = 20
#     image = cv2.copyMakeBorder(image, pad, pad, pad, pad, borderType = cv2.BORDER_CONSTANT, value=255)
#     # kernel = np.ones((2,2), np.uint8)
#     # image = cv2.dilate(image, kernel, iterations=1)
#     # image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#     cv2.imwrite(str(file_path.parent / f"{file_path.stem}_thresh.jpg"), image)
#     return image

# file_path = "crop/0412/set_name_4.jpg"
# for i in range(0, 255, 10):
#     image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
#     _, image = cv2.threshold(image, i, 255, cv2.THRESH_BINARY)
#     pad = 20
#     image = cv2.copyMakeBorder(image, pad, pad, pad, pad, borderType = cv2.BORDER_CONSTANT, value=255)
#     # kernel = np.ones((2,2), np.uint8)
#     # image = cv2.dilate(image, kernel, iterations=1)
#     # image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#     cv2.imwrite(str(f"crop/0412/set_name_4_{i}.jpg"), image)
import string
whitelist = set(string.ascii_letters + string.digits + string.whitespace + ".,+%\':")

def get_ocr_text(image):
    # Config: https://stackoverflow.com/questions/44619077/pytesseract-ocr-multiple-config-options
    config="--psm 6"
    ocr_text = pytesseract.image_to_string(image, config=config)
    ocr_text.strip()
    ocr_text = "".join(char for char in ocr_text if char in whitelist)
    return ocr_text

ocr_dir = pathlib.Path("crop/")
for artifact_dir in ocr_dir.iterdir():
    if artifact_dir.is_dir():
        for file_stem in light_text:
            processed_image = process_light(artifact_dir / (file_stem + ".jpg"))
            print(f"{file_stem} : {get_ocr_text(processed_image)}")

        for file_stem in dark_text:
            processed_image = process_dark(artifact_dir / (file_stem + ".jpg"))
            print(f"{file_stem} : {get_ocr_text(processed_image)}")