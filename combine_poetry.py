import json
import os

data = []

all_poetry_file_path = [os.path.join(
    "./poetry", file_path) for file_path in os.listdir('./poetry')]

for file_path in all_poetry_file_path:
    with open(file_path, "r", encoding="utf-8") as f:
        json_file = json.load(f)
        print("current data size: ", len(json_file))
        data.extend(json_file)
        print("total data size: ", len(data))

with open("./data/all_poetry.json", "w+")as f:
    json.dump(data, f)
