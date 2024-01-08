from __future__ import annotations

import json
import string
import re
from collections.abc import Sequence
from typing import Any
import editdistance
import functools

from collections.abc import Collection, Sequence
from typing import Optional
import os
import constants
import cv2
import utils
import pathlib
import datetime
import good_format

PathLike = os.PathLike | str


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


with open("ArtifactInfo.json", "r", encoding="utf-8") as f:
    artifact_info = json.loads(f.read())


_whitelist_names = set(string.ascii_letters + string.whitespace + r"-'")
whitelist = set(string.ascii_letters + string.digits + string.whitespace + r".,+%':")
blacklist = set(r".,+%':")


def filter_chars(word, whitelist=None, blacklist=None):
    if whitelist:
        word = "".join(char for char in word if char in whitelist)
    if blacklist:
        word = "".join(char for char in word if char not in blacklist)

    return word


class ArtifactImage:
    """Image of a new artifact."""

    def __init__(self, image_path: PathLike):
        self.image_path = pathlib.Path(image_path)
        self.image = cv2.imread(str(self.image_path))
        self.height, self.width, *_ = self.image.shape


class Artifact:
    def __init__(
        self,
        artifact_type: str = None,
        level: str = None,
        rarity: int = None,
        main_stat: str = None,
        value: str = None,
        set_name: str = None,
        substats: dict[str, float] = None,
        equipped: str = None,
        set_name_3: str = None,
        set_name_4: str = None,
        substats_3: Sequence[str] = None,
        substats_4: Sequence[str] = None,
        artifact_id: str = None,
        file_path: str = None,
        creation_time: datetime.datetime = None,
    ):
        self.artifact_id = artifact_id
        self.file_path = file_path

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

        self.creation_time = creation_time

    def to_dict(self):
        return {
            "artifact_type": self.artifact_type,
            "level": self.level,
            "rarity": self.rarity,
            "main_stat": self.main_stat,
            "value": self.value,
            "set_name": self.set_name,
            "substats": self.substats,
            "roll_value": self.roll_value,
            "crit_value": self.crit_value,
            "equipped": self.equipped,
            "artifact_id": self.artifact_id,
            "creation_time": datetime.datetime.strptime(
                " ".join(self.artifact_id.split(" ")[-2:]), "%Y-%m-%d %H:%M:%S"
            ),
        }

    def __repr__(self) -> str:
        return f"""
Artifact(
    artifact_type="{self.artifact_type}",
    level="{self.level}",
    rarity="{self.rarity}",
    main_stat="{self.main_stat}",
    value="{self.value}",
    set_name="{self.set_name}",
    substats="{self.substats}",
    equipped="{self.equipped}",
    artifact_id="{self.artifact_id}",
    file_path="{self.file_path}",
)
""".strip()

    def __str__(self) -> str:
        formatted_str = """
            {artifact_id}
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

        formatted_str = formatted_str.format(
            artifact_type=self.artifact_type,
            rarity=self.rarity,
            main_stat=self.main_stat,
            level=self.level,
            value=self.value,
            substats=substats_formatted,
            set_name=self.set_name,
            equipped=self.equipped,
            artifact_id=self.artifact_id,
        )

        # Remove leading whitespace on each
        formatted_str = formatted_str.strip()
        formatted_str_lines = formatted_str.split("\n")
        formatted_str_lines = [line.strip() for line in formatted_str_lines]
        return "\n".join(formatted_str_lines)

    def __eq__(self, other: Optional["Artifact"]) -> bool:
        if other is None:
            return False

        return all(
            [
                self.artifact_type == other.artifact_type,
                self.level == other.level,
                self.rarity == other.rarity,
                self.main_stat == other.main_stat,
                self.value == other.value,
                self.set_name == other.set_name,
                self.substats == other.substats,
                self.equipped == other.equipped,
            ]
        )

    def to_good_format(self) -> dict[str, Any]:
        # https://frzyc.github.io/genshin-optimizer/#/doc

        substat_list = []
        for substat_name, substat_value in self.substats.items():
            # percent_stat = substat_value.endswith("%")
            # substat_value = float(substat_value.strip("%"))
            # if percent_stat:
            #     substat_name += "%"

            substat_dict = {
                "key": good_format.statKey[substat_name],
                "value": substat_value,
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
            "location": good_format.location[self.equipped] if self.equipped else "",
            "lock": True,
            "substats": substat_list,
        }

    def _format_artifact_type(self, artifact_type: str) -> str:
        artifact_type = filter_chars(artifact_type, whitelist=_whitelist_names)
        artifact_type = artifact_type.strip()
        if artifact_type in constants.VALID_ARTIFACT_TYPES:
            return artifact_type
        else:
            print(constants.VALID_ARTIFACT_TYPES)
            raise InvalidArtifactTypeError(
                f"Can not match artifact type to expected format: {artifact_type}"
            )

    def _format_level(self, level: str) -> int:
        level = level.strip()
        if re.match("\+\d+", level):
            return int(level[1:])
        else:
            raise InvalidLevelError(f"Can not match level to expected format: {level}")

    def _format_rarity(self, rarity: int) -> int:
        if 1 <= rarity <= 5:
            return int(rarity)
        else:
            raise InvalidRarityError(
                f"Can not match rarity to expected format: {rarity}"
            )

    def _format_main_stat(self, main_stat: str) -> str:
        main_stat = filter_chars(main_stat, whitelist=_whitelist_names)
        main_stat = main_stat.strip()

        if main_stat in constants.VALID_ARTIFACT_MAIN_STATS:
            return main_stat
        else:
            for valid_main_stat in constants.VALID_ARTIFACT_MAIN_STATS:
                if valid_main_stat in main_stat:
                    return valid_main_stat

        print(constants.VALID_ARTIFACT_MAIN_STATS)
        raise InvalidMainStatError(
            f"Can not match main_stat to expected value: >{main_stat}<"
        )

    def _format_value(self, value: str) -> str:
        value = value.strip()
        return value

    def _format_set_name(self, set_name: str, set_name_4: str = "") -> str:
        set_name = filter_chars(set_name, whitelist=_whitelist_names)
        set_name_4 = filter_chars(set_name_4, whitelist=_whitelist_names)

        if set_name in constants.VALID_ARTIFACT_SET_NAMES:
            self.num_substats = 3
            return set_name
        elif set_name_4 in constants.VALID_ARTIFACT_SET_NAMES:
            self.num_substats = 4
            return set_name_4
        else:
            raise InvalidSetNameError(
                f"{self.artifact_id}: Can not match set name to expected value: >{set_name}< or >{set_name_4}<"
            )

    def _format_substats(
        self, substats: Sequence[str], substats_4: Sequence[str] | None = None
    ) -> dict[str, float]:
        if substats_4 is None:
            substats = substats
        elif self.num_substats == 4:
            substats = substats_4
        elif self.num_substats == 3:
            substats = substats

        substat_values = {}

        for substat in substats:
            substat = substat.strip()
            match = re.match(r"([\w\s]+)\+([\d.%]+)", substat)
            substat_type = match.group(1)
            substat_value = match.group(2)
            if substat_value.endswith("%"):
                substat_value = substat_value.strip("%")
                substat_type += "%"
            substat_value = float(substat_value)
            substat_values[substat_type] = substat_value

        # self.num_substats = len(substat_values)
        return substat_values

    def _format_equipped(self, equipped: str) -> str:
        equipped = filter_chars(equipped, whitelist=_whitelist_names)
        equipped = equipped.strip()
        match = re.match(r"(Equipped)\s([\w\s]+)", equipped)
        if match:
            return match.group(2)
        else:
            return None

    @property
    def roll_value(self) -> int:
        """Returns the current artifact roll value."""
        return roll_value(self.substats)
    
    @property
    def crit_value(self) -> float:
        """Returns the current artifact crit value."""
        return crit_value(self.substats)

    @classmethod
    def from_ocr_json(cls, ocr_json: dict) -> Artifact:
        return cls(
            artifact_type=ocr_json["artifact_type"],
            level=ocr_json["level"],
            rarity=ocr_json["rarity"],
            main_stat=ocr_json["main_stat"],
            value=ocr_json["value"],
            set_name_3=ocr_json["set_name_3"],
            set_name_4=ocr_json["set_name_4"],
            substats_3=ocr_json["substats_3"],
            substats_4=ocr_json["substats_4"],
            equipped=ocr_json["equipped"],
            artifact_id=ocr_json["artifact_id"],
        )

def roll_value(substats: dict[str, float]) -> int:
    """Calculates the current artifact roll value."""
    total_roll_value = 0
    for substat_name, value in substats.items():
        total_roll_value += round(
            value / constants.MAX_ARTIFACT_SUBSTAT_ROLL_VALUES_5_STAR[substat_name], 1
        )
    return round(100 * total_roll_value)

def crit_value(substats: dict[str, float]) -> float:
    """Calculates the current artifact crit value."""
    total_crit_value = 0
    for substat_name, value in substats.items():
        if substat_name == "CRIT Rate%":
            total_crit_value += value * 2
        elif substat_name == "CRIT DMG%":
            total_crit_value += value
    return total_crit_value

def artifact_list_to_good_format_json(
    artifact_list: list[Artifact],
    output_path="artifacts_good_format.json",
    verbose=True,
) -> None:
    artifact_list_good_format = []
    for artifact in artifact_list:
        try:
            artifact_list_good_format.append(artifact.to_good_format())
        except Exception:
            print(artifact.artifact_id)
            raise

    artifacts_good_format = {
        "format": "GOOD",
        "version": 1,
        "source": "Genshin Artifact Extractor",
        "artifacts": artifact_list_good_format,
    }

    utils.write_json(artifacts_good_format, output_path)
    # if verbose:
    #     print(f"Created updated GI Database: {output_path}")


@functools.cache
def edit_distance(a: str, b: str) -> int:
    """Returns the minimum number of operations required to convert a to b."""
    return editdistance.distance(a, b)


def find_closest_match(text: str, valid_values: Collection[str], max_distance: int = 4):
    """Returns the closest match to text in valid_values.

    Args:
        text: The text to match.
        valid_values: The values to match against.
        max_distance: The maximum edit distance allowed.

    Returns:
        The closest match to text in values.
    """
    closest_match = None
    for valid_value in valid_values:
        distance = edit_distance(text, valid_value)
        if distance < max_distance:
            closest_match = valid_value
            max_distance = distance

    return closest_match
