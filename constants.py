import dataclasses
import string
import json


@dataclasses.dataclass()
class CropROI:
    left: int
    top: int
    width: int
    height: int


@dataclasses.dataclass()
class CropOffset:
    """Pixel dimensions for ROI offset from +0 template to Artifact.

    Attributes:
        horizontal_offset: The pixel distance between the left edge of the artifact and the template coordinate.
        vertical_offset: The pixel distance between the top edge of the artifact and the template coordinate.
        artifact_width: The pixel width of the artifact box.
        artifact_height: The pixel height of the artifact box.
    """

    horizontal_offset: int
    vertical_offset: int
    artifact_width: int
    artifact_height: int


# Maps the dimensions of an image to the expected crop offset for the Artifact
# box relative to the +0 template coordinates.
OFFSET_FOR_IMAGE_DIMENSIONS = {
    (2732, 2048): CropOffset(
        horizontal_offset=50,
        vertical_offset=547,
        artifact_width=874,
        artifact_height=1731,
    ),
    (3840, 2160): CropOffset(
        horizontal_offset=47,
        vertical_offset=510,
        artifact_width=819,
        artifact_height=1620,
    ),
}

# Artifact image height and width expected for OCR to work.
ARTIFACT_EXPECTED_WIDTH = 612
ARTIFACT_EXPECTED_HEIGHT = 1135

# Path to "+0" template image to identify location for new artifacts.
TEMPLATE_IMAGE_PATH = "templates/2732x2048.png"

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
    "set_name_4" :      CropROI(  26,  629,  540,   53),
    "substats_3" :      CropROI(  59,  440,  428,  145),
    "set_name_3" :      CropROI(  26,  581,  540,   53),
    "equipped" :        CropROI( 100, 1071,  511,   63),
}
# fmt: on

# Constants
LIGHT_TEXT = ["artifact_type", "level", "main_stat", "value"]
DARK_TEXT = ["equipped", "set_name_3", "set_name_4", "substats_3", "substats_4"]

WHITELIST = set(string.ascii_letters + string.digits + string.whitespace + r".,+-%\':")

STRTIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


VALID_ARTIFACT_TYPES = {
    "Flower of Life",
    "Plume of Death",
    "Sands of Eon",
    "Goblet of Eonothem",
    "Circlet of Logos",
}

VALID_ARTIFACT_LEVEL = set(range(1, 21))  # [1, 20]

VALID_ARTIFACT_RARITY = set(range(1, 6))  # [1, 5]

VALID_ARTIFACT_MAIN_STATS = {
    "HP",
    "ATK",
    "DEF",
    "HP%",
    "ATK%",
    "DEF%",
    "Physical DMG Bonus",
    "Pyro DMG Bonus",
    "Electro DMG Bonus",
    "Cryo DMG Bonus",
    "Hydro DMG Bonus",
    "Anemo DMG Bonus",
    "Geo DMG Bonus",
    "Dendro DMG Bonus",
    "Elemental Mastery",
    "Energy Recharge",
    "CRIT Rate",
    "CRIT DMG",
    "Healing Bonus",
}

VALID_ARTIFACT_SET_NAMES = {
    "Instructor",
    "Gladiator's Finale",
    "Wanderer's Troupe",
    "Thundersoother",
    "Thundering Fury",
    "Maiden Beloved",
    "Viridescent Venerer",
    "Crimson Witch of Flames",
    "Lavawalker",
    "Noblesse Oblige",
    "Bloodstained Chivalry",
    "Archaic Petra",
    "Retracing Bolide",
    "Blizzard Strayer",
    "Heart of Depth",
    "Tenacity of the Millelith",
    "Pale Flame",
    "Emblem of Severed Fate",
    "Shimenawa's Reminiscence",
    "Husk of Opulent Dreams",
    "Ocean-Hued Clam",
    "Echoes of an Offering",
    "Vermillion Hereafter",
    "Deepwood Memories",
    "Gilded Dreams",
    "Flower of Paradise Lost",
    "Desert Pavilion Chronicle",
    "Nymph's Dream",
    "Vourukasha's Glow",
    "Marechaussee Hunter",
    "Golden Troupe",
    "Nighttime Whispers in the Echoing",
    "Song of Days Past",
}

VALID_ARTIFACT_SUBSTATS = {
    "HP",
    "ATK",
    "DEF",
    "HP%",
    "ATK%",
    "DEF%",
    "Elemental Mastery",
    "Energy Recharge",
    "CRIT Rate",
    "CRIT DMG",
}

# Not currently used.
VALID_CHARACTER_NAMES = {
    "Albedo",
    "Aloy",
    "Alhaitham",
    "Amber",
    "Arataki Itto",
    "Baizhu",
    "Barbara",
    "Beidou",
    "Bennett",
    "Candace",
    "Charlotte",
    "Chevreuse",
    "Chiori",
    "Chongyun",
    "Collei",
    "Cyno",
    "Diluc",
    "Diona",
    "Eula",
    "Faruzan",
    "Fischl",
    "Freminet",
    "Furina",
    "Gaming",
    "Ganyu",
    "Gorou",
    "Hu Tao",
    "Jean",
    "Kaedehara Kazuha",
    "Kaeya",
    "Kamisato Ayaka",
    "Kamisato Ayato",
    "Kaveh",
    "Keqing",
    "Kirara",
    "Klee",
    "Kujou Sara",
    "Kuki Shinobu",
    "Layla",
    "Lisa",
    "Mona",
    "Nahida",
    "Nilou",
    "Ningguang",
    "Noelle",
    "Qiqi",
    "Raiden Shogun",
    "Razor",
    "Rosaria",
    "Sangonomiya Kokomi",
    "Sayu",
    "Shikanoin Heizou",
    "Shenhe",
    "Sucrose",
    "Tartaglia",
    "Thoma",
    "Tighnari",
    "Crazy",
    "Venti",
    "Wanderer",
    "Xiangling",
    "Xianyun",
    "Xiao",
    "Xingqiu",
    "Xinyan",
    "Yanfei",
    "Yelan",
    "Yae Miko",
    "Yoimiya",
    "Yun Jin",
    "Zhongli",
}


def get_artifact_max_substat_roll_values(
    artifact_info_json_file: str = "ArtifactInfo.json",
) -> dict[str, float]:
    with open(artifact_info_json_file, "r", encoding="utf-8") as f:
        artifact_info = json.loads(f.read())

    max_roll_values: dict[str, float] = {}
    for artifact_stat in artifact_info["ArtifactTiers"][0]["data"]["Substats"]:
        stat_name = list(artifact_stat["name"].keys())[0]
        max_roll_value = artifact_stat["rolls"][-1]
        max_roll_values[stat_name] = max_roll_value

    return max_roll_values


MAX_ARTIFACT_SUBSTAT_ROLL_VALUES = get_artifact_max_substat_roll_values()

print(MAX_ARTIFACT_SUBSTAT_ROLL_VALUES)
