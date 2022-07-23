import cv2
import pathlib
import numpy as np
import shutil

# https://docs.opencv.org/3.4/de/d85/namespacecv_1_1img__hash.html

# hashes = set()

# # Crop all images
# cropped_frames_dir = pathlib.Path("cropped_frames/")
# for img_path in sorted(cropped_frames_dir.iterdir()):
#     if img_path.suffix == ".jpg":
#         image = cv2.imread(str(img_path))
#         img_hash = cv2.img_hash.pHash(image)
#         # Add hash to set
#         hashes.add(tuple(img_hash[0]))
#         print(f"{img_path.name} : {img_hash}")

# print(f"Found {len(hashes)} unique hashes")

# img_path1 = "cropped_frames/0030.jpg"
# img_path2 = "cropped_frames/0031.jpg"

# image1 = cv2.imread(img_path1)
# image2 = cv2.imread(img_path2)

# img_hash1 = cv2.img_hash.pHash(image1)
# img_hash2 = cv2.img_hash.pHash(image2)

# print(tuple(img_hash1[0]))

output_dir = pathlib.Path("valid_frames/")
hashes = set()
valid_frames = []

cropped_frames_dir = pathlib.Path("cropped_frames/")
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

###

# output_dir = pathlib.Path("valid_frames/")
# # hashes = set()
# valid_frames = []

# # Valid if different than previous by more than 1 unit

# cropped_frames_dir = pathlib.Path("cropped_frames/")
# prev_hash = np.zeros(8)
# for img_path in sorted(cropped_frames_dir.iterdir()):
#     if img_path.suffix == ".jpg":
#         image = cv2.imread(str(img_path))
#         img_hash = cv2.img_hash.pHash(image)
#         img_hash = img_hash[0]

#         # If the difference has more than 1 different elements, it is a valid frame
#         if np.count_nonzero(img_hash - prev_hash) > 1:
#             valid_frames.append(img_path)
#             print(f"{img_path.name} : {img_hash}")
#             prev_hash = img_hash


# print(f"Found {len(valid_frames)} artifacts")

# for img_path in valid_frames:
#     shutil.copy(img_path, output_dir / img_path.name)

