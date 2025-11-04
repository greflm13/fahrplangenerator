import os
import csv
import math
import argparse
from typing import Dict, List, Tuple, Optional, Union

import questionary
import contextily as cx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import geopandas as gpd
from shapely.geometry import shape, Point
from matplotlib.axes import Axes
from xyzservices import TileProvider
import pickle
import hashlib
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

from modules.logger import logger

if __package__ is None:
    PACKAGE = ""
else:
    PACKAGE = __package__
SCRIPTDIR = os.path.abspath(os.path.dirname(__file__).removesuffix(PACKAGE))


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
    parser.add_argument(
        "--map-provider",
        type=str,
        choices=["BasemapAT", "OPNVKarte", "OSM", "OSMDE", "ORM"],
        default="BasemapAT",
        help="Map provider for the basemap (default: BasemapAT)",
        dest="map_provider",
    )
    return parser.parse_args()


def compare_coords(lat1: Union[str, float], lon1: Union[str, float], lat2: Union[str, float], lon2: Union[str, float], tolerance: float = 1e-9) -> bool:
    """Compare two coordinates with given absolute tolerance."""
    return math.isclose(float(lat1), float(lat2), abs_tol=tolerance) and math.isclose(float(lon1), float(lon2), abs_tol=tolerance)


def compute_cache_key(file_paths: List[str]) -> str:
    """Compute a cache key based on file paths and their mtimes/sizes."""
    h = hashlib.sha256()
    for p in sorted(file_paths):
        try:
            stat = os.stat(p)
            h.update(p.encode())
            h.update(str(stat.st_mtime_ns).encode())
            h.update(str(stat.st_size).encode())
        except Exception:
            h.update(p.encode())
    return h.hexdigest()


def get_cache_path(key: str) -> str:
    cache_dir = os.path.join(SCRIPTDIR, "__pycache__", "spatial_index")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"stop_index_{key}.pkl")


def build_spatial_index(stops: List[Dict[str, str]]) -> Dict:
    """Build KDTree and auxiliary arrays from stops list.

    Returns a dict: { 'kdtree': KDTree, 'coords': np.ndarray, 'names': List[str], 'stop_ids': List[str] }
    """
    coords = []
    names = []
    stop_ids = []
    for s in stops:
        lat = s.get("stop_lat", "")
        lon = s.get("stop_lon", "")
        sid = s.get("stop_id")
        name = s.get("stop_name", "")
        try:
            latf = float(lat)
            lonf = float(lon)
        except Exception:
            continue
        coords.append((latf, lonf))
        names.append(name)
        stop_ids.append(sid)
    if not coords:
        return {"kdtree": None, "coords": np.array([]), "names": [], "stop_ids": []}
    arr = np.array(coords)
    if KDTree is None:
        logger.warning("scipy not available; spatial index disabled")
        return {"kdtree": None, "coords": arr, "names": names, "stop_ids": stop_ids}
    tree = KDTree(arr)
    return {"kdtree": tree, "coords": arr, "names": names, "stop_ids": stop_ids}


def load_or_build_spatial_index(stops: List[Dict], cache_key: Optional[str] = None) -> Dict:
    """Load spatial index from cache if available, otherwise build and cache it."""
    if cache_key:
        cache_path = get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    idx = pickle.load(f)
                    logger.info("Loaded spatial index from cache %s", cache_path)
                    return idx
            except Exception:
                logger.debug("Failed to load cache, rebuilding")
    idx = build_spatial_index(stops)
    if cache_key:
        try:
            with open(get_cache_path(cache_key), "wb") as f:
                pickle.dump(idx, f)
                logger.info("Cached spatial index to %s", get_cache_path(cache_key))
        except Exception as exc:
            logger.debug("Failed to write spatial index cache: %s", exc)
    return idx


def find_nearest_stop(spatial_index: Dict, lat: str, lon: str) -> Optional[str]:
    """Find nearest stop name using KDTree spatial index. Returns None if not found."""
    try:
        latf = float(lat)
        lonf = float(lon)
    except Exception:
        return None
    tree = spatial_index.get("kdtree")
    if tree is None:
        names = spatial_index.get("names", [])
        coords = spatial_index.get("coords", np.array([]))
        if coords.size == 0:
            return None
        d2 = np.sum((coords - np.array([latf, lonf])) ** 2, axis=1)
        idx = int(np.argmin(d2))
        return names[idx]
    dist, idx = tree.query([latf, lonf])
    names = spatial_index.get("names", [])
    if idx < len(names):
        return names[int(idx)]
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
                before = len(shapes)
                if pd is not None:
                    try:
                        df = pd.read_csv(shapes_path, dtype=str)
                        df = df.fillna("")
                        shapes.extend(df.to_dict(orient="records"))
                    except Exception as exc:
                        logger.debug("pandas failed to read %s: %s; falling back to csv", shapes_path, exc)
                        with open(shapes_path, mode="r", encoding="utf-8-sig") as f:
                            shapes.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
                else:
                    with open(shapes_path, mode="r", encoding="utf-8-sig") as f:
                        shapes.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
                logger.debug("Loaded %d shapes from %s", len(shapes) - before, shapes_path)
            else:
                logger.debug("No shapes.txt in %s", folder)
            if os.path.exists(trips_path):
                before = len(trips)
                if pd is not None:
                    try:
                        df = pd.read_csv(trips_path, dtype=str)
                        df = df.fillna("")
                        trips.extend(df.to_dict(orient="records"))
                    except Exception as exc:
                        logger.debug("pandas failed to read %s: %s; falling back to csv", trips_path, exc)
                        with open(trips_path, mode="r", encoding="utf-8-sig") as f:
                            trips.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
                else:
                    with open(trips_path, mode="r", encoding="utf-8-sig") as f:
                        trips.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
                logger.debug("Loaded %d trips from %s", len(trips) - before, trips_path)
            else:
                logger.debug("No trips.txt in %s", folder)
            if os.path.exists(stops_path):
                before = len(stops)
                if pd is not None:
                    try:
                        df = pd.read_csv(stops_path, dtype=str)
                        df = df.fillna("")
                        stops.extend(df.to_dict(orient="records"))
                    except Exception as exc:
                        logger.debug("pandas failed to read %s: %s; falling back to csv", stops_path, exc)
                        with open(stops_path, mode="r", encoding="utf-8-sig") as f:
                            stops.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
                else:
                    with open(stops_path, mode="r", encoding="utf-8-sig") as f:
                        stops.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
                logger.debug("Loaded %d stops from %s", len(stops) - before, stops_path)
            else:
                logger.debug("No stops.txt in %s", folder)
            stop_times_path = os.path.join(folder, "stop_times.txt")
            if os.path.exists(stop_times_path):
                before = len(stop_times)
                if pd is not None:
                    try:
                        df = pd.read_csv(stop_times_path, dtype=str)
                        df = df.fillna("")
                        stop_times.extend(df.to_dict(orient="records"))
                    except Exception as exc:
                        logger.debug("pandas failed to read %s: %s; falling back to csv", stop_times_path, exc)
                        with open(stop_times_path, mode="r", encoding="utf-8-sig") as f:
                            stop_times.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
                else:
                    with open(stop_times_path, mode="r", encoding="utf-8-sig") as f:
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
    ax: Axes,
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
    try:
        rgb = mcolors.to_rgb(line_color)
        stop_rgb = tuple(max(0.0, c * 0.5) for c in rgb)
        stop_color = stop_rgb
    except Exception:
        stop_color = (1.0, 0.0, 0.0)

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

    for _, row in gdf_stops.iterrows():
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
            try:
                txt.set_path_effects([pe.withStroke(linewidth=0.5, foreground="white")])
            except Exception:
                pass
        except Exception:
            continue

    return len(gdf_stops)


def build_shapedict(shapes: List[Dict], stop_index: Dict[int, Dict[Tuple[float, float], List[str]]]) -> Dict[str, Dict]:
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
            name = find_nearest_stop(stop_index, shapeline["shape_pt_lat"], shapeline["shape_pt_lon"]) or ""
            shapedict[sid]["start_stop"] = name
        shapedict[sid]["points"].append([shapeline["shape_pt_lon"], shapeline["shape_pt_lat"]])

    logger.info("Associating end stops with shapes...")
    for _, shap in shapedict.items():
        if not shap["end_stop"] and shap["points"]:
            last_point = shap["points"][-1]
            name = find_nearest_stop(stop_index, last_point[1], last_point[0]) or ""
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


def get_provider_source(provider_name: str) -> TileProvider:
    """Get the contextily provider source based on the provider name."""
    provider_map = {
        "BasemapAT": cx.providers.BasemapAT.basemap,
        "OPNVKarte": cx.providers.OPNVKarte,
        "OSM": cx.providers.OpenStreetMap.Mapnik,
        "OSMDE": cx.providers.OpenStreetMap.DE,
        "ORM": cx.providers.OpenRailwayMap,
        "TomTom": cx.providers.TomTom.Hybrid,
    }
    return provider_map.get(provider_name, cx.providers.BasemapAT.basemap)


def main():
    args = parse_args()

    logger.info("Loading GTFS data...")
    shapes, trips, stops, stop_times = load_gtfs(args.gtfs)
    logger.info("Loaded %d points, %d trips, %d stops and %d stop_times from GTFS data.", len(shapes), len(trips), len(stops), len(stop_times))

    stops_paths = [os.path.join(folder, "stops.txt") for folder in args.gtfs if os.path.exists(os.path.join(folder, "stops.txt"))]
    cache_key = compute_cache_key(stops_paths) if stops_paths else None
    spatial_index = load_or_build_spatial_index(stops, cache_key)
    shapedict = build_shapedict(shapes, spatial_index)
    routes = build_routes(trips, shapedict)

    stops_index = build_stops_index(stops)
    stop_times_index = build_stop_times_index(stop_times)

    choicelist: List[str] = [f"{route[1]} - {route[2]}" for route in routes]

    while True:
        geodata = []
        choice = questionary.autocomplete("Haltestelle/Bahnhof: ", choices=choicelist, match_middle=True, validate=lambda val: val in choicelist, style=questionary_style).ask()
        if choice is None:
            logger.info("No route selected, exiting.")
            break
        idx = next((i for i, route in enumerate(routes) if choice == f"{route[1]} - {route[2]}"), None)
        if idx is None:
            logger.warning("Selected choice not found in routes: %s", choice)
            continue
        route = routes[idx]
        sid = route[0]
        points = shapedict[sid]["points"]
        geodata.append(shape({"type": "LineString", "coordinates": points}))

        gdf = gpd.GeoDataFrame({"geometry": geodata}, crs="EPSG:4326")
        gminx, gmaxx, gminy, gmaxy = gdf.total_bounds
        gbox_w = gmaxx - gminx
        gbox_h = gmaxy - gminy
        gbox_aspect = gbox_w / gbox_h
        if gbox_aspect >= 1.0:
            figsize = (10 * math.sqrt(2), 10)
        else:
            figsize = (10, 10 * math.sqrt(2))
        ax = gdf.plot(facecolor="none", edgecolor=args.color, linewidth=args.linewidth, figsize=figsize)

        if args.plot_stops:
            logger.info("Plotting stops for route %s -> %s", route[1], route[2])
            try:
                n = plot_stops_on_ax(ax, sid, trips, stops_index, stop_times_index, line_color=args.color)
                logger.info("Plotted %d stops for route", n)
            except Exception as exc:
                logger.warning("Exception while plotting stops: %s", exc)

        ax.set_axis_off()
        minx, maxx = ax.get_xlim()
        miny, maxy = ax.get_ylim()
        bbox_w = maxx - minx
        bbox_h = maxy - miny
        bbox_aspect = bbox_w / bbox_h

        if bbox_aspect >= 1.0:
            target_aspect = math.sqrt(2)
            target_ysize = bbox_w / target_aspect
            increase_y = (target_ysize - bbox_h) / 2.0
            ax.set_ylim(miny - increase_y, maxy + increase_y)
        else:
            target_aspect = 1.0 / math.sqrt(2)
            target_xsize = bbox_h * target_aspect
            increase_x = (target_xsize - bbox_w) / 2.0
            ax.set_xlim(minx - increase_x, maxx + increase_x)

        logger.info("Generating map for %s -> %s...", route[1], route[2])
        zoom_param = args.zoom if args.zoom >= 0 else "auto"
        done = False
        while not done:
            try:
                logger.debug("Adding basemap with zoom=%s and provider=%s", zoom_param, args.map_provider)
                cx.add_basemap(ax=ax, crs="EPSG:4326", source=get_provider_source(args.map_provider), zoom=zoom_param, reset_extent=True)
                done = True
            except Exception as exc:
                logger.warning("Failed to add basemap: %s", exc)
                if isinstance(zoom_param, int) and zoom_param > 0:
                    zoom_param -= 1
                    logger.info("Retrying with lower zoom level: %s", zoom_param)
                else:
                    logger.error("Cannot add basemap, giving up.")
                    done = True
        outname = f"{route[1].replace('/', '')} - {route[2].replace('/', '')} Map.png"

        try:
            plt.savefig(outname, dpi=1200, bbox_inches="tight", pad_inches=0)
            logger.info("Saved map to %s", outname)
        except Exception as exc:
            logger.error("Failed to save map to %s: %s", outname, exc)
        plt.show()


if __name__ == "__main__":
    main()
