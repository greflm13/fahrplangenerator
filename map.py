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
from matplotlib.figure import Figure
import geopandas as gpd
from shapely.geometry import shape, Point
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
    parser.add_argument("-l", "--linewidth", type=float, default=2.0, help="Line width of the shapes on the map", dest="linewidth")
    parser.add_argument("--label-rotation", type=int, default=30, help="Fixed rotation angle (degrees) to apply to all stop labels")
    parser.add_argument("-s", "--stops", action="store_true", help="Plot stops along the selected route", dest="plot_stops")
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
    cache_dir = os.path.join(SCRIPTDIR, "__pycache__")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"stop_index_{key}.pkl")


def build_spatial_index(stops: List[Dict]) -> Dict:
    """Build KDTree and auxiliary arrays from stops list.

    Returns a dict: { 'kdtree': KDTree, 'coords': np.ndarray, 'names': List[str], 'stop_ids': List[str] }
    """
    coords = []
    names = []
    stop_ids = []
    for s in stops:
        lat = s.get("stop_lat")
        lon = s.get("stop_lon")
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


def find_stop_by_coords(spatial_index: Dict, lat: str, lon: str) -> Optional[str]:
    """Compatibility wrapper for older calls: find nearest stop using spatial index."""
    return find_nearest_stop(spatial_index, lat, lon)


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
    ax,
    sid: str,
    trips: List[Dict],
    stops_index: Dict[str, Dict],
    stop_times_index: Dict[str, List[Dict]],
    line_color: str = "green",
    marker_size: int = 30,
    label_fontsize: int = 7,
    target_crs: Optional[str] = None,
    label_rotation: int = 30,
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
    if target_crs:
        try:
            gdf_stops = gdf_stops.to_crs(target_crs)
        except Exception:
            logger.debug("Failed to transform stops to %s, keeping original CRS", target_crs)

    try:
        gdf_stops.plot(ax=ax, color=stop_color, markersize=marker_size, zorder=5)
    except Exception as exc:
        logger.warning("Failed to plot stops GeoDataFrame: %s", exc)
        return 0

    fig = ax.figure
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        renderer = None

    used_bboxes = []
    offsets = [(4, 4), (-4, 4), (4, -4), (-4, -4), (8, 0), (-8, 0), (0, 8), (0, -8)]
    # Use a single fixed rotation for all labels so the map looks consistent.
    # This rotation is tried first for each label; only if placement fails we
    # reduce fontsize. Change this angle to taste (degrees).
    fixed_angle = int(label_rotation)
    angles = [fixed_angle]

    for idx, row in gdf_stops.iterrows():
        pt = row.geometry
        name = row.get("stop_name", "")
        if not name:
            continue
        fontsize = label_fontsize
        placed = False

        while fontsize >= 5 and not placed:
            # Try the single fixed rotation with different offsets first.
            for angle in angles:
                for off in offsets:
                    try:
                        txt = ax.annotate(
                            name,
                            xy=(pt.x, pt.y),
                            xytext=off,
                            textcoords="offset points",
                            fontsize=fontsize,
                            color="black",
                            ha="left",
                            va="bottom",
                            rotation=angle,
                            rotation_mode="anchor",
                            zorder=6,
                            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "black", "linewidth": 0.4, "alpha": 0.9},
                            clip_on=False,
                        )
                        try:
                            txt.set_path_effects([pe.withStroke(linewidth=0.5, foreground="white")])
                        except Exception:
                            pass
                        if renderer is None:
                            placed = True
                            break
                        bbox = txt.get_window_extent(renderer)
                        overlap = False
                        for ub in used_bboxes:
                            if ub is not None and bbox.overlaps(ub):
                                overlap = True
                                break
                        if not overlap:
                            used_bboxes.append(bbox)
                            placed = True
                            break
                        else:
                            txt.remove()
                    except Exception:
                        try:
                            txt.remove()
                        except Exception:
                            pass
                        continue
                if placed:
                    break
            if not placed:
                # only reduce fontsize after all offsets with the fixed rotation failed
                fontsize -= 1

        if not placed:
            try:
                ax.annotate(
                    name,
                    xy=(pt.x, pt.y),
                    xytext=offsets[0],
                    textcoords="offset points",
                    fontsize=max(5, label_fontsize - 2),
                    color="black",
                    ha="left",
                    va="bottom",
                    zorder=6,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "black", "linewidth": 0.4, "alpha": 0.9},
                    clip_on=False,
                )
            except Exception:
                pass

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
            name = find_stop_by_coords(stop_index, shapeline["shape_pt_lat"], shapeline["shape_pt_lon"]) or ""
            shapedict[sid]["start_stop"] = name
        shapedict[sid]["points"].append([shapeline["shape_pt_lon"], shapeline["shape_pt_lat"]])

    logger.info("Associating end stops with shapes...")
    for _, shap in shapedict.items():
        if not shap["end_stop"] and shap["points"]:
            last_point = shap["points"][-1]
            name = find_stop_by_coords(stop_index, last_point[1], last_point[0]) or ""
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

        unit = 10.0
        default_portrait = (unit, unit * math.sqrt(2))
        default_landscape = (unit * math.sqrt(2), unit)

        gdf = gpd.GeoDataFrame({"geometry": geodata}, crs="EPSG:4326")
        try:
            gdf_3857 = gdf.to_crs(epsg=3857)
        except Exception:
            gdf_3857 = gdf

        minx, miny, maxx, maxy = gdf_3857.total_bounds
        bbox_w = maxx - minx if maxx > minx else 1.0
        bbox_h = maxy - miny if maxy > miny else 1.0

        bbox_aspect = bbox_w / bbox_h
        if bbox_aspect >= 1.0:
            figsize = default_landscape
            fig_aspect = figsize[0] / figsize[1]
        else:
            figsize = default_portrait
            fig_aspect = figsize[0] / figsize[1]

        target_w = bbox_h * fig_aspect
        target_h = bbox_w / fig_aspect
        if target_w > bbox_w:
            extra = (target_w - bbox_w) / 2.0
            minx -= extra
            maxx += extra
        elif target_h > bbox_h:
            extra = (target_h - bbox_h) / 2.0
            miny -= extra
            maxy += extra

        # Create a fresh figure/axes sized to the target aspect. We'll render
        # the basemap image to this axes and then plot the route on top. Using
        # plt.subplots prevents GeoPandas from autoscaling the axes and
        # collapsing the image/geometry into a tiny blob.
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_axis_off()
        ax.set_aspect("equal", adjustable="box")

        logger.info("Generating map for %s -> %s...", route[1], route[2])

        target_dpi = 300
        target_px_w = int(figsize[0] * target_dpi)
        target_px_h = int(figsize[1] * target_dpi)

        try:
            provider = cx.providers.BasemapAT.basemap
        except Exception:
            provider = getattr(cx.providers, "Stamen", None)
            if provider is not None and hasattr(provider, "TonerLite"):
                provider = provider.TonerLite
            else:
                provider = list(cx.providers.values())[0]

        if args.zoom >= 0:
            try_zoom = int(args.zoom)
        else:
            R = 6378137.0
            bbox_w_m = max(bbox_w, 1.0)
            est = target_px_w * 2 * math.pi * R / (256.0 * bbox_w_m)
            if est <= 0:
                try_zoom = 0
            else:
                try_zoom = max(0, int(math.floor(math.log2(est))))
            try_zoom = max(0, min(19, try_zoom))

        img = None
        img_ext = None
        # For each zoom attempt, fetch tiles and expand the bbox in the shorter
        # direction until the returned image aspect >= desired figure aspect.
        for z in range(try_zoom, min(19, try_zoom + 3) + 1):
            logger.debug("Attempting basemap fetch at zoom=%d", z)
            # start with the route bbox
            cur_minx, cur_miny, cur_maxx, cur_maxy = minx, miny, maxx, maxy
            best_arr = None
            best_ext = None
            # allow a few expansion iterations to pull additional tiles in the short direction
            for expand_iter in range(6):
                try:
                    arr, arr_ext = cx.bounds2img(cur_minx, cur_miny, cur_maxx, cur_maxy, z, source=provider)
                except Exception as exc:
                    logger.debug("bounds2img failed at zoom %d expand_iter %d: %s", z, expand_iter, exc)
                    arr = None
                    arr_ext = None
                if arr is None:
                    break
                h, w = arr.shape[0], arr.shape[1]
                img_aspect = (w / h) if h > 0 else 1.0
                logger.debug("Got basemap image %dx%d (aspect %.3f) at zoom %d (iter %d)", w, h, img_aspect, z, expand_iter)
                # keep the highest resolution image seen so far
                if best_arr is None or (w > best_arr.shape[1] and h > best_arr.shape[0]):
                    best_arr = arr
                    best_ext = arr_ext
                # if this image already matches/exceeds desired figure aspect and pixels, pick it
                if img_aspect >= fig_aspect and w >= target_px_w and h >= target_px_h:
                    img = arr
                    img_ext = arr_ext
                    logger.info("Selected basemap zoom=%d (image %dx%d) after %d expansions", z, w, h, expand_iter)
                    break
                # If image aspect is less than desired, expand width (short direction)
                if img_aspect < fig_aspect:
                    need_factor = fig_aspect / max(img_aspect, 1e-6)
                    cur_width = cur_maxx - cur_minx
                    new_width = cur_width * need_factor
                    extra = (new_width - cur_width) / 2.0
                    cur_minx -= extra
                    cur_maxx += extra
                    logger.debug("Expanding bbox width by factor %.3f (iter %d)", need_factor, expand_iter)
                    continue
                # If image aspect is greater than desired, expand height
                if img_aspect > fig_aspect:
                    need_factor = img_aspect / max(fig_aspect, 1e-6)
                    cur_height = cur_maxy - cur_miny
                    new_height = cur_height * need_factor
                    extra = (new_height - cur_height) / 2.0
                    cur_miny -= extra
                    cur_maxy += extra
                    logger.debug("Expanding bbox height by factor %.3f (iter %d)", need_factor, expand_iter)
                    continue
                # fallback small expansion to try to increase resolution
                cur_minx -= 0.05 * (cur_maxx - cur_minx)
                cur_maxx += 0.05 * (cur_maxx - cur_minx)
                cur_miny -= 0.05 * (cur_maxy - cur_miny)
                cur_maxy += 0.05 * (cur_maxy - cur_miny)
            # if we did not find ideal arr, use the best_arr we collected
            if img is None and best_arr is not None:
                img = best_arr
                img_ext = best_ext
                logger.info("Using best-available basemap at zoom=%d (image %dx%d)", z, img.shape[1], img.shape[0])
            if img is not None:
                break

        # track final extent we'll use for plotting overlays
        final_minx, final_miny, final_maxx, final_maxy = minx, miny, maxx, maxy

        if img is not None and img_ext is not None:
            try:
                # Crop the fetched tile image to the exact figure aspect centered on the route bbox.
                west, south, east, north = img_ext
                w_px = img.shape[1]
                h_px = img.shape[0]

                # desired bbox centered on original route center with figure aspect
                cx0 = 0.5 * (minx + maxx)
                cy0 = 0.5 * (miny + maxy)
                desired_w = bbox_h * fig_aspect
                desired_h = bbox_w / fig_aspect
                # ensure desired dims are not larger than available extent; clamp if necessary
                avail_w = east - west
                avail_h = north - south
                if desired_w > avail_w:
                    desired_w = avail_w
                    desired_h = desired_w / fig_aspect
                if desired_h > avail_h:
                    desired_h = avail_h
                    desired_w = desired_h * fig_aspect

                des_minx = cx0 - desired_w / 2.0
                des_maxx = cx0 + desired_w / 2.0
                des_miny = cy0 - desired_h / 2.0
                des_maxy = cy0 + desired_h / 2.0

                # Map desired bbox to pixel coordinates in the fetched image
                def x_to_px(x):
                    return int(round((x - west) / (east - west) * (w_px - 1)))

                def y_to_px(y):
                    # origin='upper' in imshow: y pixel 0 corresponds to north
                    return int(round((north - y) / (north - south) * (h_px - 1)))

                lx = max(0, x_to_px(des_minx))
                rx = min(w_px, x_to_px(des_maxx) + 1)
                ty = max(0, y_to_px(des_maxy))
                by = min(h_px, y_to_px(des_miny) + 1)

                # ensure indices valid
                if rx > lx and by > ty:
                    cropped = img[ty:by, lx:rx].copy()
                    ax.imshow(cropped, extent=(des_minx, des_maxx, des_miny, des_maxy), origin="lower", interpolation="nearest")
                    final_minx, final_maxx, final_miny, final_maxy = des_minx, des_maxx, des_miny, des_maxy
                else:
                    # fallback to showing full image extent
                    ax.imshow(img, extent=(west, east, south, north), origin="lower", interpolation="nearest")
                    final_minx, final_maxx, final_miny, final_maxy = west, east, south, north
            except Exception:
                try:
                    ax.imshow(img, extent=(minx, maxx, miny, maxy), origin="lower", interpolation="nearest")
                    final_minx, final_maxx, final_miny, final_maxy = minx, maxx, miny, maxy
                except Exception:
                    logger.debug("ax.imshow failed for basemap image, falling back to add_basemap")
                    try:
                        cx.add_basemap(ax=ax, crs="EPSG:3857", source=provider, zoom=try_zoom)
                    except Exception as exc:
                        logger.warning("Failed to add basemap fallback: %s", exc)
        else:
            try:
                cx.add_basemap(ax=ax, crs="EPSG:3857", source=provider, zoom=try_zoom)
            except Exception as exc:
                logger.warning("Failed to add basemap fallback: %s", exc)

        # Ensure overlays are plotted in the same extent as the basemap image
        try:
            ax.set_xlim(final_minx, final_maxx)
            ax.set_ylim(final_miny, final_maxy)
        except Exception:
            pass

        try:
            gdf_3857.plot(ax=ax, facecolor="none", edgecolor=args.color, linewidth=args.linewidth)
        except Exception:
            logger.debug("Failed to plot route geometry on top of basemap")
        outname = f"{route[1].replace('/', '')} - {route[2].replace('/', '')} Map.png"
        if args.plot_stops:
            logger.info("Plotting stops for route %s -> %s", route[1], route[2])
            try:
                n = plot_stops_on_ax(ax, sid, trips, stops_index, stop_times_index, line_color=args.color, target_crs="EPSG:3857", label_rotation=args.label_rotation)
                logger.info("Plotted %d stops for route", n)
            except Exception as exc:
                logger.warning("Exception while plotting stops: %s", exc)

        try:
            fig_obj = ax.figure
            if not isinstance(fig_obj, Figure):
                fig_obj = plt.gcf()
            try:
                fig_obj.set_size_inches(figsize, forward=True)
            except Exception:
                try:
                    fig_obj.set_size_inches(figsize)
                except Exception:
                    pass
            try:
                fig_obj.subplots_adjust(left=0, right=1, top=1, bottom=0)
            except Exception:
                pass
            fig_obj.savefig(outname, dpi=target_dpi)
            logger.info("Saved map to %s", outname)
        except Exception as exc:
            logger.error("Failed to save map to %s: %s", outname, exc)
        plt.show()


if __name__ == "__main__":
    main()
