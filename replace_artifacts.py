import utils

gi_data_path = "Database_3_2022-07-24_04-21-02.json"
gi_data = utils.load_json(gi_data_path)

all_artifacts_json = "artifacts_good_format.json"
all_artifacts = utils.load_json(all_artifacts_json)

# Replace data
gi_data["artifacts"] = all_artifacts["artifacts"]

updated_gi_data_path = "gi_data_updated.json"
utils.write_json(gi_data, updated_gi_data_path)