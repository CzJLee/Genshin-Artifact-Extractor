import cv2
import pathlib

# image_path = "valid_frames/0471.py"
# image = cv2.imread(image_path)

def crop_roi(image, roi):
    return image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

# roi = None
# if roi is None:
#     # Sample example image
#     img_path="valid_frames/1350.jpg"

#     #read image
#     image = cv2.imread(img_path)

#     #select ROI function
#     roi = cv2.selectROIs("window", image)

#     #print rectangle points of selected roi
#     print(roi)

#     # #Crop selected roi from raw image
#     # roi_cropped = crop_roi(image, roi)

#     # #show cropped image
#     # cv2.imshow("ROI", roi_cropped)

#     # cv2.imwrite("crop.jpg", roi_cropped)

#     #hold window
#     cv2.waitKey(0)

# print(roi)

rois = {
    "artifact_name" :   (  28,   77,  365,   46),
    "type" :            (  26,  183,  254,   37),
    "value" :           (  32,  219,  202,   62),
    "level" :           (  38,  385,   65,   30),
    "substats_4" :      (  59,  440,  428,  192),
    "set_name_4" :      (  26,  629,  504,   53),
    "substats_3" :      (  59,  440,  428,  145),
    "set_name_3" :      (  26,  581,  504,   53),
    "equipped" :        ( 100, 1071,  511,   63)
}


# # Sample example image
# img_path="valid_frames/1350.jpg"

# #read image
# image = cv2.imread(img_path)

# for key, roi in rois.items():
#     cropped_img = crop_roi(image, roi)
#     cv2.imwrite(f"crop/{key}.jpg", cropped_img)

# Crop all images
frames_dir = pathlib.Path("valid_frames/")
output_dir = pathlib.Path("crop/")
for img_path in sorted(frames_dir.iterdir()):
    if img_path.suffix == ".jpg":
        print(img_path.name)
        image = cv2.imread(str(img_path))
        for key, roi in rois.items():
            cropped_img = crop_roi(image, roi)
            img_crop_dir = output_dir / img_path.stem
            img_crop_dir.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(img_crop_dir / f"{key}.jpg"), cropped_img)
