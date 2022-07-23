import cv2
import numpy as np
import pathlib

def crop_roi(image, roi):
    return image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

roi = (1217, 153, 612, 1135)

if roi is None:
    # Sample example image
    img_path="frames/0191.jpg"

    #read image
    image = cv2.imread(img_path)

    #select ROI function
    roi = cv2.selectROI(image)

    #print rectangle points of selected roi
    print(roi)

    #Crop selected roi from raw image
    roi_cropped = crop_roi(image, roi)

    #show cropped image
    cv2.imshow("ROI", roi_cropped)

    cv2.imwrite("crop.jpg", roi_cropped)

    #hold window
    cv2.waitKey(0)

# Crop all images
frames_dir = pathlib.Path("frames/")
output_dir = pathlib.Path("cropped_frames/")
for img_path in sorted(frames_dir.iterdir()):
    if img_path.suffix == ".jpg":
        print(img_path.name)
        image = cv2.imread(str(img_path))
        cropped_img = crop_roi(image, roi)
        cv2.imwrite(str(output_dir / img_path.name), cropped_img)
