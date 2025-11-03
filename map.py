import os
import csv
import math
import argparse
from functools import cache
from typing import Dict, List, Tuple, Optional, Union

import questionary
import contextily as cx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import shape

from modules.logger import logger

logger.level = 10


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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate map from GTFS data")
    parser.add_argument("-i", "--input", action="extend", nargs="+", required=True, help="Input directory(s) containing GTFS shapes.txt", dest="gtfs")
    parser.add_argument("-z", "--zoom", type=int, default=-1, help="Zoom level for the basemap", dest="zoom")
    parser.add_argument("-c", "--color", type=str, default="green", help="Color of the shapes on the map", dest="color")
    parser.add_argument("-l", "--linewidth", type=float, default=None, help="Line width of the shapes on the map", dest="linewidth")
    return parser.parse_args()


def compare_coords(lat1: Union[str, float], lon1: Union[str, float], lat2: Union[str, float], lon2: Union[str, float], tolerance: float = 1e-9) -> bool:
    """Compare two coordinates with given absolute tolerance."""
    return math.isclose(float(lat1), float(lat2), abs_tol=tolerance) and math.isclose(float(lon1), float(lon2), abs_tol=tolerance)


@cache
def find_stop_by_coords(stopshash: str, lat: str, lon: str, max_tolerance: float = 1e-1) -> Optional[str]:
    """Find a stop name matching given coordinates.

    The original algorithm progressively increased tolerance until a match was found. To avoid
    potential infinite loops, we cap the tolerance and return None if no match is found.
    """
    stops = stopshash.split("$")
    tolerance = 1e-9
    while tolerance <= max_tolerance:
        for stop in stops:
            lat_s, lon_s, name_s = stop.split("%")
            if compare_coords(lat_s, lon_s, lat, lon, tolerance):
                return name_s
        tolerance *= 10
    logger.debug("No stop found for coords (%s, %s) within tolerance %s", lat, lon, max_tolerance)
    return None


def load_gtfs(dirs: List[str]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load shapes, trips and stops from given GTFS directories."""
    shapes: List[Dict] = []
    trips: List[Dict] = []
    stops: List[Dict] = []
    for folder in dirs:
        if os.path.isdir(folder):
            shapes_path = os.path.join(folder, "shapes.txt")
            trips_path = os.path.join(folder, "trips.txt")
            stops_path = os.path.join(folder, "stops.txt")
            if os.path.exists(shapes_path):
                with open(shapes_path, mode="r", encoding="utf-8-sig") as f:
                    shapes.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
            if os.path.exists(trips_path):
                with open(trips_path, mode="r", encoding="utf-8-sig") as f:
                    trips.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
            if os.path.exists(stops_path):
                with open(stops_path, mode="r", encoding="utf-8-sig") as f:
                    stops.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
    return shapes, trips, stops


def build_stop_hash(stops: List[Dict]) -> str:
    """Create a compact searchable string from stops for the cached lookup function."""
    return "$".join(f"{stop['stop_lat']}%{stop['stop_lon']}%{stop['stop_name']}" for stop in stops)


def build_shapedict(shapes: List[Dict], stopshash: str) -> Dict[str, Dict]:
    """Build a mapping shape_id -> {start_stop, end_stop, points}.

    Uses find_stop_by_coords but tolerates missing matches.
    """
    shapedict: Dict[str, Dict] = {}
    logger.info("Associating start stops with shapes...")
    for shapeline in shapes:
        sid = shapeline["shape_id"]
        if sid not in shapedict:
            logger.debug("Processing shape ID: %s", sid)
            shapedict[sid] = {"start_stop": None, "end_stop": None, "points": []}
        if shapeline.get("shape_pt_sequence") == "1":
            name = find_stop_by_coords(stopshash, shapeline["shape_pt_lat"], shapeline["shape_pt_lon"]) or ""
            shapedict[sid]["start_stop"] = name
        shapedict[sid]["points"].append([shapeline["shape_pt_lon"], shapeline["shape_pt_lat"]])

    logger.info("Associating end stops with shapes...")
    for key, shap in shapedict.items():
        if not shap["end_stop"] and shap["points"]:
            last_point = shap["points"][-1]
            name = find_stop_by_coords(stopshash, last_point[1], last_point[0]) or ""
            shap["end_stop"] = name
    return shapedict


def build_routes(trips: List[Dict], shapedict: Dict[str, Dict]) -> List[Tuple[str, str, str]]:
    """Return a sorted list of routes (shape_id, start_stop, end_stop)."""
    routes = set()
    for trip in trips:
        sid = trip.get("shape_id")
        if sid in shapedict:
            routes.add((sid, shapedict[sid].get("start_stop", ""), shapedict[sid].get("end_stop", "")))
    return sorted(routes, key=lambda x: x[2])


def main():
    args = parse_args()

    logger.info("Loading GTFS data...")
    shapes, trips, stops = load_gtfs(args.gtfs)
    logger.info("Loaded %d points, %d trips, and %d stops from GTFS data.", len(shapes), len(trips), len(stops))

    stopshash = build_stop_hash(stops)
    shapedict = build_shapedict(shapes, stopshash)
    routes = build_routes(trips, shapedict)

    choicelist: List[str] = [f"{route[1]} - {route[2]}" for route in routes]

    while True:
        geodata = []
        choice = questionary.autocomplete("Haltestelle/Bahnhof: ", choices=choicelist, match_middle=True, validate=lambda val: val in choicelist, style=questionary_style).ask()
        if choice is None:
            logger.info("No route selected, exiting.")
            break
        # find index of chosen route
        idx = next((i for i, route in enumerate(routes) if choice == f"{route[1]} - {route[2]}"), None)
        if idx is None:
            logger.warning("Selected choice not found in routes: %s", choice)
            continue
        route = routes[idx]
        sid = route[0]
        points = shapedict[sid]["points"]
        geodata.append(shape({"type": "LineString", "coordinates": points}))

        gdf = gpd.GeoDataFrame({"geometry": geodata}, crs="EPSG:4326")
        ax = gdf.plot(facecolor="none", edgecolor=args.color, linewidth=args.linewidth, figsize=(10, 10))
        ax.set_axis_off()
        logger.info("Generating map for %s -> %s...", route[1], route[2])
        cx.add_basemap(ax=ax, crs="EPSG:4326", source=cx.providers.BasemapAT.basemap, zoom=args.zoom if args.zoom >= 0 else "auto")
        outname = f"{route[1].replace('/', '')} - {route[2].replace('/', '')}_map.png"
        plt.savefig(outname, dpi=1200, bbox_inches="tight", pad_inches=1)
        plt.show()


if __name__ == "__main__":
    main()
