#!/usr/bin/env python

import os
import logging
import asyncio
import argparse

import tqdm
import PIL.Image
import questionary

from rich_argparse import RichHelpFormatter

import modules.utils as utils
import modules.db as db

from modules.map import MAP_PROVIDERS
from modules.logger import setup_logger, rotate_log_file
from modules.compute import compute

# Constants for file paths and exclusions
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__)).removesuffix(__package__ if __package__ else "")
PIL.Image.MAX_IMAGE_PIXELS = 9331200000


custom_style = questionary.Style(
    [
        ("question", "fg:#ff0000 bold"),
        ("answer", "fg:#00ff00 bold"),
        ("pointer", "fg:#0000ff bold"),
        ("highlighted", "fg:#ffff00 bold"),
        ("completion-menu", "bg:#000000"),
        ("completion-menu.completion.current", "bg:#444444"),
    ]
)


async def main():
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter)
    parser.add_argument("-i", "--input", help="Input folder(s)", action="extend", nargs="+", default=[], required=False, dest="input")
    parser.add_argument("-c", "--color", help="Timetable color", type=str, required=False, dest="color", default="random")
    parser.add_argument("-o", "--output", help="Output file", type=str, required=False, dest="output", default="fahrplan.pdf")
    parser.add_argument("-m", "--map", help="Generate maps", action="store_true", dest="map")
    parser.add_argument("-s", "--stop-name-csv", help="Stop name mapping csv folder", required=False, type=str, dest="mapping_csv")
    parser.add_argument("-r", "--reset-db", help="Reset local database", action="store_true", dest="reset_db")
    parser.add_argument("--dpi", help="map dpi", type=int, dest="map_dpi")
    parser.add_argument("--no-logo", action="store_false", dest="logo")
    parser.add_argument(
        "--map-provider",
        type=str,
        choices=MAP_PROVIDERS.keys(),
        default="BasemapAT",
        help="Map provider for the basemap (default: BasemapAT)",
        dest="map_provider",
    )
    args = parser.parse_args()

    rotate_log_file(compress=True)
    setup_logger()
    logger = logging.getLogger(name=os.path.basename(SCRIPTDIR))

    if args.reset_db:
        append = False
    else:
        append = True

    if args.mapping_csv:
        await db.load_hst_csv(args.mapping_csv, append=append)

    if len(args.input) > 0:
        for aid, folder in tqdm.tqdm(enumerate(args.input), total=len(args.input), desc="Loading data", unit="folder", ascii=True, dynamic_ncols=True):
            if os.path.isdir(folder):
                await db.load_gtfs(folder, aid, append=append)
                append = True
            else:
                logger.error("Input folder %s does not exist", folder)

    stops = await db.get_table_data("stops")

    stop_hierarchy = await utils.build_stop_hierarchy()
    stop_hierarchy = await utils.query_stop_names(stop_hierarchy)
    destinations = await utils.build_dest_list()
    stop_id_mapping = {}
    for stop in stop_hierarchy.values():
        if stop.stop_name not in stop_id_mapping:
            stop_id_mapping[stop.stop_name] = [stop.stop_id]
        else:
            stop_id_mapping[stop.stop_name].append(stop.stop_id)
    stops = utils.build_list_index(stops, "stop_id")

    logger.info("Loaded data")

    choices = sorted(stop_id_mapping.keys())
    while True:
        try:
            choice = await asyncio.to_thread(
                lambda: questionary.autocomplete("Haltestelle/Bahnhof: ", choices=choices, match_middle=True, validate=lambda val: val in choices, style=custom_style).ask()
            )
        except KeyboardInterrupt:
            print()
            break
        if choice is None:
            print()
            break
        if len(stop_id_mapping[choice]) == 1:
            ourstop = [stop_hierarchy[stop_id_mapping[choice][0]]]
            if ourstop[0].children is not None:
                ourstop.extend(ourstop[0].children)
        else:
            ourstop = []
            for stop_id in stop_id_mapping[choice]:
                stop = stop_hierarchy[stop_id]
                if stop.children is not None:
                    ourstop.extend(stop.children)
                ourstop.append(stop)

        await compute(ourstop, stops, args, destinations)


if __name__ == "__main__":
    asyncio.run(main())
