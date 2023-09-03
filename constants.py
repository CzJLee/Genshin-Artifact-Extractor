import dataclasses
import string


@dataclasses.dataclass()
class CropROI:
    left: int
    top: int
    width: int
    height: int


# Pixel distance between template coordinate and artifact top left coordinate.
VERTICAL_OFFSET = 547
HORIZONTAL_OFFSET = 50

# Artifact image height and width expected for OCR to work.
ARTIFACT_EXPECTED_WIDTH = 612
ARTIFACT_EXPECTED_HEIGHT = 1135

# Path to "+0" template image to identify location for new artifacts.
TEMPLATE_IMAGE_PATH = "template.png"

# MAGIC NUMBERS
ROI = CropROI(left=1217, top=153, width=612, height=1135)

# Crop Region of Interest for iPad
# fmt: off
ROIS = {
    "artifact_type" :   CropROI(  28,   77,  320,   46),
    "main_stat" :       CropROI(  26,  183,  300,   37),
    "value" :           CropROI(  26,  219,  208,   62),
    "level" :           CropROI(  38,  385,   65,   30),
    "rarity":           CropROI(  26,  290,  254,   45),
    "substats_4" :      CropROI(  59,  440,  428,  192),
    "set_name_4" :      CropROI(  26,  629,  504,   53),
    "substats_3" :      CropROI(  59,  440,  428,  145),
    "set_name_3" :      CropROI(  26,  581,  504,   53),
    "equipped" :        CropROI( 100, 1071,  511,   63),
}
# fmt: on

# Constants
LIGHT_TEXT = ["artifact_type", "level", "main_stat", "value"]
DARK_TEXT = ["equipped", "set_name_3", "set_name_4", "substats_3", "substats_4"]

WHITELIST = set(string.ascii_letters + string.digits + string.whitespace + r".,+-%\':")
