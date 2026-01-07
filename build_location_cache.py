#!/usr/bin/env python
import os
import json
import argparse

import tqdm

from modules.logger import logger
from modules.utils import load_gtfs, build_stop_hierarchy, CACHEDIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--hst-json", help="json containing all stops", required=True, type=str, dest="stopjson")
    parser.add_argument("-i", "--input", help="Input folder(s)", action="extend", nargs="+", required=True, dest="input")
    args = parser.parse_args()

    stops = []

    for folder in tqdm.tqdm(args.input, desc="Loading data", unit=" folders", ascii=True, dynamic_ncols=True):
        if os.path.isdir(folder):
            stops.extend(load_gtfs(folder, "stops"))
        else:
            logger.error("Input folder %s does not exist", folder)

    stop_hierarchy = build_stop_hierarchy(stops)

    with open(args.stopjson, "r", encoding="utf-8") as f:
        stop_json = json.loads(f.read())

    id_mapping = {}
    for stop in stop_json["features"]:
        stop_id = stop["properties"]["stg_globid"]
        stop_name = stop["properties"]["stg_name"]
        id_mapping[stop_id] = stop_name

    loccache = os.path.join(CACHEDIR, "location_cache")
    with open(loccache, "r", encoding="utf-8") as f:
        location_cache = json.loads(f.read())

    for parent_stop in stop_hierarchy.values():
        stop_id: str = parent_stop["stop_id"]
        try:
            location_cache[stop_id] = id_mapping[stop_id.removeprefix("Parent").removeprefix("P")]
        except Exception:
            pass

    with open(loccache, "w", encoding="utf-8") as f:
        location_cache = f.write(json.dumps(location_cache, indent=2))


if __name__ == "__main__":
    main()
