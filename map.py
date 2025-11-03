import os
import csv
import math
import argparse
from functools import cache
from typing import Dict, List, Tuple, Optional, Union

import questionary
import contextily as cx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import geopandas as gpd
from shapely.geometry import shape, Point

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
    parser.add_argument("-s", "--stops", action="store_true", dest="plot_stops", help="Plot stops along the selected route")
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
                logger.debug("Matched stop '%s' for coords (%s,%s) with tolerance %s", name_s, lat, lon, tolerance)
                return name_s
        tolerance *= 10
    logger.debug("No stop found for coords (%s, %s) within tolerance %s", lat, lon, max_tolerance)
    return None


def load_gtfs(dirs: List[str]) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Load shapes, trips and stops from given GTFS directories."""
    shapes: List[Dict] = []
    trips: List[Dict] = []
    stops: List[Dict] = []
    stop_times: List[Dict] = []
    logger.debug("Scanning %d directories for GTFS files", len(dirs))
    for folder in dirs:
        logger.debug("Checking folder: %s", folder)
        if os.path.isdir(folder):
            shapes_path = os.path.join(folder, "shapes.txt")
            trips_path = os.path.join(folder, "trips.txt")
            stops_path = os.path.join(folder, "stops.txt")
            logger.info("Loading GTFS from %s", folder)
            if os.path.exists(shapes_path):
                with open(shapes_path, mode="r", encoding="utf-8-sig") as f:
                    before = len(shapes)
                    shapes.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
                    logger.debug("Loaded %d shapes from %s", len(shapes) - before, shapes_path)
            else:
                logger.debug("No shapes.txt in %s", folder)
            if os.path.exists(trips_path):
                with open(trips_path, mode="r", encoding="utf-8-sig") as f:
                    before = len(trips)
                    trips.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
                    logger.debug("Loaded %d trips from %s", len(trips) - before, trips_path)
            else:
                logger.debug("No trips.txt in %s", folder)
            if os.path.exists(stops_path):
                with open(stops_path, mode="r", encoding="utf-8-sig") as f:
                    before = len(stops)
                    stops.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
                    logger.debug("Loaded %d stops from %s", len(stops) - before, stops_path)
            else:
                logger.debug("No stops.txt in %s", folder)
            stop_times_path = os.path.join(folder, "stop_times.txt")
            if os.path.exists(stop_times_path):
                with open(stop_times_path, mode="r", encoding="utf-8-sig") as f:
                    before = len(stop_times)
                    stop_times.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
                    logger.debug("Loaded %d stop_times from %s", len(stop_times) - before, stop_times_path)
            else:
                logger.debug("No stop_times.txt in %s", folder)
    return shapes, trips, stops, stop_times


def build_stop_hash(stops: List[Dict]) -> str:
    """Create a compact searchable string from stops for the cached lookup function."""
    logger.debug("Building stop hash from %d stops", len(stops))
    res = "$".join(f"{stop['stop_lat']}%{stop['stop_lon']}%{stop['stop_name']}" for stop in stops)
    logger.debug("Stop hash length: %d", len(res))
    return res


def build_stops_index(stops: List[Dict]) -> Dict[str, Dict]:
    """Index stops by stop_id for quick lookup."""
    idx: Dict[str, Dict] = {}
    for s in stops:
        sid = s.get("stop_id")
        if sid:
            idx[sid] = s
    logger.debug("Built stops index with %d entries", len(idx))
    return idx


def build_stop_times_index(stop_times: List[Dict]) -> Dict[str, List[Dict]]:
    """Index stop_times by trip_id for quick lookup."""
    idx: Dict[str, List[Dict]] = {}
    for st in stop_times:
        tid = st.get("trip_id")
        if not tid:
            continue
        idx.setdefault(tid, []).append(st)
    logger.debug("Built stop_times index with %d trips", len(idx))
    return idx


def plot_stops_on_ax(
    ax,
    sid: str,
    trips: List[Dict],
    stops_index: Dict[str, Dict],
    stop_times_index: Dict[str, List[Dict]],
    line_color: str = "green",
    marker_size: int = 30,
    label_fontsize: int = 7,
) -> int:
    """Plot stops for a given shape_id onto the provided axes and annotate stop names.

    The stop marker color is derived from the line color but made visually distinct by
    darkening it.

    Returns the number of plotted stops.
    """
    # derive stop color from line color (darker)
    try:
        rgb = mcolors.to_rgb(line_color)
        stop_rgb = tuple(max(0.0, c * 0.5) for c in rgb)
        stop_color = stop_rgb
    except Exception:
        # fallback to red
        stop_color = (1.0, 0.0, 0.0)

    # find trip_ids that reference this shape_id
    trip_ids = [t["trip_id"] for t in trips if t.get("shape_id") == sid and t.get("trip_id")]
    stop_ids = set()
    for tid in trip_ids:
        for st in stop_times_index.get(tid, []):
            if st.get("stop_id"):
                stop_ids.add(st.get("stop_id"))

    stop_points = []
    stop_rows = []
    for stopid in sorted(stop_ids):
        s = stops_index.get(stopid)
        if not s:
            continue
        lon_s = s.get("stop_lon")
        lat_s = s.get("stop_lat")
        if lon_s is None or lat_s is None:
            continue
        try:
            lon = float(lon_s)
            lat = float(lat_s)
        except Exception:
            continue
        stop_points.append(Point(lon, lat))
        stop_rows.append({"stop_id": stopid, "stop_name": s.get("stop_name", "")})

    if not stop_points:
        return 0

    gdf_stops = gpd.GeoDataFrame(stop_rows, geometry=stop_points, crs="EPSG:4326")
    try:
        gdf_stops.plot(ax=ax, color=stop_color, markersize=marker_size, zorder=5)
    except Exception as exc:
        logger.warning("Failed to plot stops GeoDataFrame: %s", exc)
        return 0

    # annotate stop names next to each point
    for idx, row in gdf_stops.iterrows():
        pt = row.geometry
        name = row.get("stop_name", "")
        if not name:
            continue
        try:
            txt = ax.annotate(
                name,
                xy=(pt.x, pt.y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=label_fontsize,
                color="black",
                ha="left",
                va="bottom",
                zorder=6,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "black", "linewidth": 0.4, "alpha": 0.9},
                clip_on=False,
            )
            # Add a slight stroke to improve legibility on variable backgrounds
            try:
                txt.set_path_effects([pe.withStroke(linewidth=0.5, foreground="white")])
            except Exception:
                pass
        except Exception:
            # annotation can fail in some backends; ignore gracefully
            continue

    return len(gdf_stops)


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
    routes_list = sorted(routes, key=lambda x: x[2])
    logger.info("Discovered %d routes", len(routes_list))
    return routes_list


def main():
    args = parse_args()

    logger.info("Loading GTFS data...")
    shapes, trips, stops, stop_times = load_gtfs(args.gtfs)
    logger.info("Loaded %d points, %d trips, %d stops and %d stop_times from GTFS data.", len(shapes), len(trips), len(stops), len(stop_times))

    stopshash = build_stop_hash(stops)
    shapedict = build_shapedict(shapes, stopshash)
    routes = build_routes(trips, shapedict)

    # Index stops & stop_times for plotting
    stops_index = build_stops_index(stops)
    stop_times_index = build_stop_times_index(stop_times)

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
        # Add basemap and handle potential provider issues gracefully
        zoom_param = args.zoom if args.zoom >= 0 else "auto"
        try:
            logger.debug("Adding basemap with zoom=%s and provider=BasemapAT", zoom_param)
            cx.add_basemap(ax=ax, crs="EPSG:4326", source=cx.providers.BasemapAT.basemap, zoom=zoom_param)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to add basemap: %s", exc)
        outname = f"{route[1].replace('/', '')} - {route[2].replace('/', '')}_map.png"
        # If requested, plot stops for this route
        if args.plot_stops:
            logger.info("Plotting stops for route %s -> %s", route[1], route[2])
            try:
                n = plot_stops_on_ax(ax, sid, trips, stops_index, stop_times_index, line_color=args.color)
                logger.info("Plotted %d stops for route", n)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Exception while plotting stops: %s", exc)

        try:
            plt.savefig(outname, dpi=1200, bbox_inches="tight", pad_inches=1)
            logger.info("Saved map to %s", outname)
        except Exception as exc:  # pragma: no cover - I/O error handling
            logger.error("Failed to save map to %s: %s", outname, exc)
        plt.show()


if __name__ == "__main__":
    main()
