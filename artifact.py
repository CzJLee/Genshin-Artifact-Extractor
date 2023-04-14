import cv2
import pathlib
import numpy as np
import json
import time
import shutil
import pytesseract
import string
import re
import tempfile
from fastprogress import fastprogress

from typing import Optional

import utils
import good_format


class ArtifactError(Exception):
    pass


class InvalidArtifactTypeError(ArtifactError):
    # artifact_type
    pass


class InvalidLevelError(ArtifactError):
    pass


class InvalidRarityError(ArtifactError):
    pass


class InvalidMainStatError(ArtifactError):
    pass


class InvalidValueError(ArtifactError):
    pass


class InvalidSetNameError(ArtifactError):
    pass


class InvalidSubstatsError(ArtifactError):
    pass


class InvalidEquippedError(ArtifactError):
    pass


with open("ArtifactInfo.json") as f:
    artifact_info = json.loads(f.read())

# Valid artifact_type
_valid_artifact_type = {
    "Flower of Life", "Plume of Death", "Sands of Eon", "Goblet of Eonothem",
    "Circlet of Logos"
}

# Valid level
_valid_level = set(range(1, 21))  # [1, 20]

# Valid rarity
_valid_rarity = set(range(1, 6))  # [1, 5]

# Valid main_stat
_valid_main_stat = {
    "HP", "ATK", "DEF", "HP%", "ATK%", "DEF%", "Physical DMG Bonus",
    "Pyro DMG Bonus", "Electro DMG Bonus", "Cryo DMG Bonus", "Hydro DMG Bonus",
    "Anemo DMG Bonus", "Geo DMG Bonus", "Dendro DMG Bonus",
    "Elemental Mastery", "Energy Recharge", "CRIT Rate", "CRIT DMG",
    "Healing Bonus"
}

# Valid set_name
_valid_set_name = {
    "Gladiator's Finale", "Wanderer's Troupe", "Thundersoother",
    "Thundering Fury", "Maiden Beloved", "Viridescent Venerer",
    "Crimson Witch of Flames", "Lavawalker", "Noblesse Oblige",
    "Bloodstained Chivalry", "Archaic Petra", "Retracing Bolide",
    "Blizzard Strayer", "Heart of Depth", "Tenacity of the Millelith",
    "Pale Flame", "Emblem of Severed Fate", "Shimenawa's Reminiscence",
    "Husk of Opulent Dreams", "Ocean-Hued Clam", "Echoes of an Offering",
    "Vermillion Hereafter", "Retracing Bolide", "Deepwood Memories",
    "Gilded Dreams", "Flower of Paradise Lost", "Desert Pavilion Chronicle",
    "Nymph's Dream", "Vourukasha's Glow"
}

# Valid substats
_valid_substats = {
    "HP", "ATK", "DEF", "HP%", "ATK%", "DEF%", "Elemental Mastery",
    "Energy Recharge", "CRIT Rate", "CRIT DMG"
}
# for substat_names in artifact_info["ArtifactTiers"][0]["data"]["Substats"]:
#     substat_name = list(substat_names["name"].keys())[0]
#     _valid_substats.append(substat_name)

# Valid character names
_valid_character_names = {
    "Albedo", "Aloy", "Amber", "Arataki Itto", "Baizhu", "Barbara", "Beidou",
    "Bennett", "Candace", "Chongyun", "Collei", "Cyno", "Diluc", "Diona",
    "Eula", "Faruzan", "Fischl", "Ganyu", "Gorou", "Hu Tao", "Jean",
    "Kaedehara Kazuha", "Kaeya", "Kamisato Ayaka", "Kamisato Ayato", "Kaveh",
    "Keqing", "Kirara", "Klee", "Kujou Sara", "Kuki Shinobu", "Layla", "Lisa",
    "Mona", "Nahida", "Nilou", "Ningguang", "Noelle", "Qiqi", "Raiden Shogun",
    "Razor", "Rosaria", "Sangonomiya Kokomi", "Sayu", "Shikanoin Heizou",
    "Shenhe", "Sucrose", "Tartaglia", "Thoma", "Tighnari", "Crazy", "Venti",
    "Wanderer", "Xiangling", "Xiao", "Xingqiu", "Xinyan", "Yanfei", "Yelan",
    "Yae Miko", "Yoimiya", "Yun Jin", "Zhongli"
}
# for character in artifact_info["Characters"]:
#     character_name = list(character["name"].keys())[0]
#     _valid_character_names.append(character_name)

valid_values = {
    "artifact_type": _valid_artifact_type,
    "main_stat": _valid_main_stat,
    "set_name": _valid_set_name,
    "substats": _valid_substats,
    "equipped": _valid_character_names
}

_whitelist_names = set(string.ascii_letters + string.whitespace + "-\'")
whitelist = set(string.ascii_letters + string.digits + string.whitespace +
                ".,+%\':")
blacklist = set(".,+%\':")


def filter_chars(word, whitelist=None, blacklist=None):
    if whitelist:
        word = "".join(char for char in word if char in whitelist)
    if blacklist:
        word = "".join(char for char in word if char not in blacklist)

    return word


class Artifact():

    def __init__(self,
                 artifact_type=None,
                 level=None,
                 rarity=None,
                 main_stat=None,
                 value=None,
                 set_name=None,
                 substats=None,
                 equipped=None,
                 set_name_3=None,
                 set_name_4=None,
                 substats_3=None,
                 substats_4=None,
                 file_path=None):

        self.artifact_type = self._format_artifact_type(artifact_type)
        self.level = self._format_level(level)
        self.rarity = self._format_rarity(rarity)
        self.main_stat = self._format_main_stat(main_stat)
        self.value = self._format_value(value)
        self.num_substats = None
        if set_name:
            self.set_name = set_name
        else:
            self.set_name = self._format_set_name(set_name_3, set_name_4)

        if substats:
            self.substats = substats
        else:
            self.substats = self._format_substats(substats_3, substats_4)

        self.equipped = self._format_equipped(equipped)

        self.file_path = file_path

    # def to_dict(self):
    #     return {
    #         "artifact_type" : self.artifact_type,
    #         "level" : self.level,
    #         "rarity" : self.rarity,
    #         "main_stat" : self.main_stat,
    #         "value" : self.value,
    #         "set_name" : self.set_name,
    #         "substats" : self.substats,
    #         "equipped" : self.equipped
    #     }

    def __str__(self):
        formatted_str = """
            {artifact_type} ({rarity}*)

            {main_stat} (+{level})
            {value}

            {substats}

            {set_name}
            Equipped: {equipped}
            """

        substats = []
        for substat_name, substat_value in self.substats.items():
            if substat_name.endswith("%"):
                substat_name = substat_name.strip("%")
                substat_value = str(substat_value) + "%"
            substats.append(f"{substat_name}+{substat_value}")
        substats_formatted = "\n".join(substats)

        formatted_str = formatted_str.format(artifact_type=self.artifact_type,
                                             rarity=self.rarity,
                                             main_stat=self.main_stat,
                                             level=self.level,
                                             value=self.value,
                                             substats=substats_formatted,
                                             set_name=self.set_name,
                                             equipped=self.equipped)

        # Remove leading whitespace on each
        formatted_str = formatted_str.strip()
        formatted_str_lines = formatted_str.split("\n")
        formatted_str_lines = [line.strip() for line in formatted_str_lines]
        return "\n".join(formatted_str_lines)

    def __eq__(self, other: Optional["Artifact"]):
        if other is None:
            return False

        return all([
            self.artifact_type == other.artifact_type,
            self.level == other.level, self.rarity == other.rarity,
            self.main_stat == other.main_stat, self.value == other.value,
            self.set_name == other.set_name, self.substats == other.substats,
            self.equipped == other.equipped
        ])

    def to_good_format(self):
        # https://frzyc.github.io/genshin-optimizer/#/doc

        substat_list = []
        for substat_name, substat_value in self.substats.items():
            # percent_stat = substat_value.endswith("%")
            # substat_value = float(substat_value.strip("%"))
            # if percent_stat:
            #     substat_name += "%"

            substat_dict = {
                "key": good_format.statKey[substat_name],
                "value": substat_value
            }
            substat_list.append(substat_dict)

        # Handle % values for main stat
        main_stat_is_percent_stat = self.value.endswith("%")
        main_stat = self.main_stat
        if main_stat_is_percent_stat:
            main_stat = self.main_stat + "%"

        return {
            "setKey": good_format.setKey[self.set_name],
            "slotKey": good_format.slotKey[self.artifact_type],
            "level": self.level,
            "rarity": self.rarity,
            "mainStatKey": good_format.statKey[main_stat],
            "location":
            good_format.location[self.equipped] if self.equipped else "",
            "lock": True,
            "substats": substat_list
        }

    def _format_artifact_type(self, artifact_type):
        artifact_type = filter_chars(artifact_type, whitelist=_whitelist_names)
        artifact_type = artifact_type.strip()
        if artifact_type in _valid_artifact_type:
            return artifact_type
        else:
            print(_valid_artifact_type)
            raise InvalidArtifactTypeError(
                f"Can not match artifact type to expected format: {artifact_type}"
            )

    def _format_level(self, level):
        level = level.strip()
        if re.match("\+\d+", level):
            return int(level[1:])
        else:
            raise InvalidLevelError(
                f"Can not match level to expected format: {level}")

    def _format_rarity(self, rarity):
        if 1 <= rarity <= 5:
            return int(rarity)
        else:
            raise InvalidRarityError(
                f"Can not match rarity to expected format: {rarity}")

    def _format_main_stat(self, main_stat):
        main_stat = filter_chars(main_stat, whitelist=_whitelist_names)
        main_stat = main_stat.strip()

        if main_stat in _valid_main_stat:
            return main_stat
        else:
            for valid_main_stat in _valid_main_stat:
                if valid_main_stat in main_stat:
                    return valid_main_stat

        print(_valid_main_stat)
        raise InvalidMainStatError(
            f"Can not match main_stat to expected value: >{main_stat}<")

    def _format_value(self, value):
        value = value.strip()
        return value

    def _format_set_name(self, set_name, set_name_4=""):
        set_name = filter_chars(set_name, whitelist=_whitelist_names)
        set_name_4 = filter_chars(set_name_4, whitelist=_whitelist_names)

        if set_name in _valid_set_name:
            self.num_substats = 3
            return set_name
        elif set_name_4 in _valid_set_name:
            self.num_substats = 4
            return set_name_4
        else:
            raise InvalidSetNameError(
                f"Can not match set name to expected value: >{set_name}< or >{set_name_4}<"
            )

    def _format_substats(self, substats, substats_4=None):
        if substats_4 is None:
            substats = substats
        elif self.num_substats == 4:
            substats = substats_4
        elif self.num_substats == 3:
            substats = substats

        substat_values = {}

        for substat in substats:
            substat = substat.strip()
            match = re.match("([\w\s]+)\+([\d.%]+)", substat)
            substat_type = match.group(1)
            substat_value = match.group(2)
            if substat_value.endswith("%"):
                substat_value = substat_value.strip("%")
                substat_type += "%"
            substat_value = float(substat_value)
            substat_values[substat_type] = substat_value

        # self.num_substats = len(substat_values)
        return substat_values

    def _format_equipped(self, equipped):
        equipped = filter_chars(equipped, whitelist=_whitelist_names)
        equipped = equipped.strip()
        match = re.match("(Equipped)\s([\w\s]+)", equipped)
        if match:
            return match.group(2)
        else:
            return None

    @classmethod
    def from_ocr_json(cls, ocr_json: dict):
        return cls(artifact_type=ocr_json["artifact_type"],
                   level=ocr_json["level"],
                   rarity=ocr_json["rarity"],
                   main_stat=ocr_json["main_stat"],
                   value=ocr_json["value"],
                   set_name_3=ocr_json["set_name_3"],
                   set_name_4=ocr_json["set_name_4"],
                   substats_3=ocr_json["substats_3"],
                   substats_4=ocr_json["substats_4"],
                   equipped=ocr_json["equipped"])


def artifact_list_to_good_format_json(artifact_list: list[Artifact],
                                      output_path="artifacts_good_format.json",
                                      verbose=True):
    artifact_list_good_format = []
    for artifact in artifact_list:
        artifact_list_good_format.append(artifact.to_good_format())

    artifacts_good_format = {
        "format": "GOOD",
        "version": 1,
        "source": "Genshin Artifact Extractor",
        "artifacts": artifact_list_good_format
    }

    utils.write_json(artifacts_good_format, output_path)
    # if verbose:
    #     print(f"Created updated GI Database: {output_path}")