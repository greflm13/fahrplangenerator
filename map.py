import os
import csv
import math
import argparse

from functools import cache

import questionary
import contextily as cx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import shape

questionary_style = questionary.Style(
    [
        ("question", "fg:#ff0000 bold"),
        ("answer", "fg:#00ff00 bold"),
        ("pointer", "fg:#0000ff bold"),
        ("highlighted", "fg:#ffff00 bold"),
        ("completion-menu", "bg:#000000"),
        ("completion-menu.completion.current", "bg:#444444"),
    ]
)
termwith = os.get_terminal_size().columns


def parse_args():
    parser = argparse.ArgumentParser(description="Generate map from GTFS data")
    parser.add_argument("-i", "--input", action="extend", nargs="+", required=True, help="Input directory containing GTFS shapes.txt", dest="gtfs")
    parser.add_argument("-z", "--zoom", type=int, default=-1, help="Zoom level for the basemap", dest="zoom")
    parser.add_argument("-c", "--color", type=str, default="green", help="Color of the shapes on the map", dest="color")
    parser.add_argument("-l", "--linewidth", type=float, default=None, help="Line width of the shapes on the map", dest="linewidth")
    return parser.parse_args()


def compare_coords(lat1, lon1, lat2, lon2, tolerance=1e-9):
    return math.isclose(float(lat1), float(lat2), abs_tol=tolerance) and math.isclose(float(lon1), float(lon2), abs_tol=tolerance)


@cache
def find_stop_by_coords(stopshash, lat, lon):
    stops = stopshash.split("$")
    tolerance = 1e-9
    while True:
        for stop in stops:
            lat_s, lon_s, name_s = stop.split("%")
            if compare_coords(lat_s, lon_s, lat, lon, tolerance):
                return name_s
        tolerance *= 10


def main():
    choicelist, shapes, trips, stops = [], [], [], []
    shapedict = {}
    routes = set()
    stopshash = ""

    args = parse_args()

    print("Loading GTFS data...")
    for folder in args.gtfs:
        if os.path.isdir(folder):
            with open(os.path.join(folder, "shapes.txt"), mode="r", encoding="utf-8-sig") as f:
                shapes.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
            with open(os.path.join(folder, "trips.txt"), mode="r", encoding="utf-8-sig") as f:
                trips.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
            with open(os.path.join(folder, "stops.txt"), mode="r", encoding="utf-8-sig") as f:
                stops.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
    print(f"Loaded {len(shapes)} points, {len(trips)} trips, and {len(stops)} stops from GTFS data.")

    stopshash = "$".join(f"{stop['stop_lat']}%{stop['stop_lon']}%{stop['stop_name']}" for stop in stops)
    print("Associating start stops with shapes...")
    for shapeline in shapes:
        if shapeline["shape_id"] not in shapedict:
            print(f"Processing shape ID: {shapeline['shape_id']}".ljust(termwith), end="\r")
            shapedict[shapeline["shape_id"]] = {"start_stop": None, "end_stop": None, "points": []}
        if shapeline["shape_pt_sequence"] == "1":
            shapedict[shapeline["shape_id"]]["start_stop"] = find_stop_by_coords(stopshash, shapeline["shape_pt_lat"], shapeline["shape_pt_lon"])
        shapedict[shapeline["shape_id"]]["points"].append([shapeline["shape_pt_lon"], shapeline["shape_pt_lat"]])
    print("Associating end stops with shapes...".ljust(termwith))
    for key, shap in shapedict.items():
        if not shap["end_stop"]:
            print(f"Finding end stop for shape ID: {key}".ljust(termwith), end="\r")
            last_point = shap["points"][-1]
            shap["end_stop"] = find_stop_by_coords(stopshash, last_point[1], last_point[0])
    print("Preparing route choices...".ljust(termwith))
    for trip in trips:
        if trip["shape_id"] in shapedict:
            routes.add((trip["shape_id"], shapedict[trip["shape_id"]]["start_stop"], shapedict[trip["shape_id"]]["end_stop"]))
    routes = sorted(routes, key=lambda x: x[2])
    for route in routes:
        choicelist.append(f"{route[2]} - {route[1]}")
    while True:
        geodata = []
        choice = questionary.autocomplete("Haltestelle/Bahnhof: ", choices=choicelist, match_middle=True, validate=lambda val: val in choicelist, style=questionary_style).ask()
        for idx, route in enumerate(routes):
            if choice == f"{route[2]} - {route[1]}":
                choice = idx
                break
        if choice is None:
            print("No route selected, exiting.")
            break
        route = routes[choice]
        geodata.append(shape({"type": "LineString", "coordinates": shapedict[route[0]]["points"]}))

        gdf = gpd.GeoDataFrame({"geometry": geodata}, crs="EPSG:4326")
        ax = gdf.plot(facecolor="none", edgecolor=args.color, linewidth=args.linewidth, figsize=(10, 10))
        ax.set_axis_off()
        print("Generating map...")
        cx.add_basemap(ax=ax, crs="EPSG:4326", source=cx.providers.BasemapAT.basemap, zoom=args.zoom if args.zoom >= 0 else "auto")
        plt.savefig(f"{route[2].replace('/', '')} - {route[1].replace('/', '')}_map.png", dpi=1200, bbox_inches="tight", pad_inches=0)
        plt.show()


if __name__ == "__main__":
    main()
