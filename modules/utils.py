import os
import copy
import random
import logging

from typing import Any, Dict, Iterable, List, Set, Tuple
from collections import defaultdict

import tqdm
import pandas as pd

from svglib.svglib import Drawing
from geopy.geocoders import Photon
from pypdf import PdfReader, PdfWriter
from shapely.geometry import shape, Point

from modules.datatypes import HierarchyStop, Shape, Stop, Routedata
from modules.db import update_location_cache, get_table_data, get_most_frequent_values, get_in_filtered_data_iter

if __package__ is None:
    PACKAGE = ""
else:
    PACKAGE = __package__
SCRIPTDIR = os.path.abspath(os.path.dirname(__file__).removesuffix(PACKAGE))
CACHEDIR = os.path.join(SCRIPTDIR, "__pycache__")

geolocator = Photon(user_agent="fahrplan.py")


def load_gtfs(folder: str, type: str) -> List[Dict]:
    """Load GTFS data."""
    logger = logging.getLogger(name=os.path.basename(SCRIPTDIR))
    data: List[Dict] = []
    if os.path.isdir(folder):
        data_path = os.path.join(folder, type + ".txt")
        logger.info("Loading GTFS from %s", data_path)
        if os.path.exists(data_path):
            logger.debug("Loading %s from %s", type, data_path)
            df = pd.read_csv(data_path, dtype=str)
            df = df.fillna("")
            data.extend(df.to_dict(orient="records"))
            logger.debug("Loaded %d %s from %s", len(data), type, data_path)
        else:
            logger.error("No %s.txt in %s", type, folder)
    return data


def build_shapedict(shape_ids: List[str]) -> Dict[str, List[Point]]:
    """Build a dictionary mapping shape_id to list of Point geometries."""
    shapedict: Dict[str, List] = defaultdict(list)

    shapes = get_in_filtered_data_iter("shapes", column="shape_id", values=shape_ids)

    for shapeline in shapes:
        sid = shapeline.shape_id
        shapedict[sid].append(Point(float(shapeline.shape_pt_lon), float(shapeline.shape_pt_lat), float(shapeline.shape_dist_traveled)))

    return dict(shapedict)


def build_list_index(list: Iterable, index: str) -> Dict[str, Any]:
    """Build an index from a list of namedtuples based on a specified key."""
    data: Dict[str, Any] = {}
    for item in list:
        data[getattr(item, index)] = item
    return data


def build_stop_times_index(trip_ids: List[str]) -> Dict[str, List]:
    """Index stop_times by trip_id for quick lookup."""
    logger = logging.getLogger(name=os.path.basename(SCRIPTDIR))
    idx: Dict[str, List] = {}
    for st in get_in_filtered_data_iter("stop_times", column="trip_id", values=trip_ids):
        tid = getattr(st, "trip_id")
        if not tid:
            continue
        idx.setdefault(tid, []).append(st)
    logger.debug("Built stop_times index with %d trips", len(idx))
    return idx


def prepare_linedraw_info(
    stop_times: Dict[str, List],
    trips: Iterable,
    stops: Dict[str, Any],
    line,
    direction,
    ourstop: List[str],
):
    """Prepare line drawing information for selected shapes."""
    shapes: Set[Shape] = set()
    stop_points: Set[Tuple] = set()
    end_stop_names: Set[str] = set()
    for trip in trips:
        if trip.route_id == line and "d" + trip.direction_id == direction:
            if trip.shape_id != "":
                shapes.add(Shape(trip.shape_id, trip.trip_id))
    shapedict = build_shapedict([shap.shapeid for shap in shapes])
    linedrawinfo = {"shapes": [], "points": [], "endstops": []}
    for shap in shapes:
        times = stop_times[shap.tripid]
        for timeidx, time in enumerate(times):
            if time.stop_id in ourstop:
                shape_dist_traveled = time.shape_dist_traveled
                geometry = shapedict[shap.shapeid]
                for geoidx, point in enumerate(geometry):
                    if point.z == float(shape_dist_traveled):
                        geo = geometry[geoidx:]
                        tim = times[timeidx:]
                        stop_points.update(
                            [
                                (
                                    Point(float(stops[stop.stop_id].stop_lon), float(stops[stop.stop_id].stop_lat)),
                                    Stop(stop.stop_id, get_stop_name(stop.stop_id)),
                                )
                                for stop in tim
                            ]
                        )
                        if len(geo) != 1:
                            linedrawinfo["shapes"].append({"geometry": shape({"type": "LineString", "coordinates": geo})})
                        endstop = stops[times[-1].stop_id]
                        end_stop_names.add(get_stop_name(endstop.stop_id))
    linedrawinfo["points"] = list(stop_points)
    linedrawinfo["endstops"] = list(end_stop_names)
    return linedrawinfo


def create_merged_pdf(pages: List[str], path: str):
    output = PdfWriter()

    for page in pages:
        pdf = PdfReader(page)
        output.add_page(pdf.pages[0])

    output.write(path)


def merge_dicts(a: Dict[str, Dict[str, Dict[str, List[Routedata]]]], b: Dict[str, Dict[str, Dict[str, List[Routedata]]]]):
    c = copy.deepcopy(a)
    for k, v in b.items():
        if k in a:
            for k2, v2 in v.items():
                if k2 in a[k]:
                    for k3, v3 in v2.items():
                        if k3 in a[k][k2]:
                            c[k][k2][k3].extend(v3)
                        else:
                            c[k][k2][k3] = v3
                else:
                    c[k][k2] = v2
        else:
            c[k] = v
    return c


def dict_set(lst: List[Routedata]) -> List[Routedata]:
    seen = []
    setlike = []
    for d in lst:
        signature = {"time": d.time, "line": d.line, "dire": d.dire}
        if signature not in seen:
            seen.append(signature)
            setlike.append(d)
    return setlike


def most_frequent(lst: List[str]):
    counts = {i: lst.count(i) for i in set(lst)}
    max_count = max(counts.values(), default=0)
    if max_count == 1:
        return lst[0]
    return max(set(lst), key=lst.count, default="")


def remove_suffix(text: str) -> str:
    sp = text.split()
    if len(sp) > 0:
        if len(sp[-1]) < 3:
            sp.pop()
    return " ".join(sp)


def merge(lst: List[str]):
    rl = []
    for i in lst:
        rl.append(remove_suffix(i))
    return rl


def scale(drawing: Drawing, scaling_factor: float):
    """
    Scale a reportlab.graphics.shapes.Drawing()
    object while maintaining the aspect ratio
    """
    scaling_x = scaling_factor
    scaling_y = scaling_factor

    drawing.width = drawing.minWidth() * scaling_x
    drawing.height = drawing.height * scaling_y
    drawing.scale(scaling_x, scaling_y)
    return drawing


def is_contrasting(color):
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness < 157


def is_vibrant(color):
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    max_channel = max(r, g, b)
    min_channel = min(r, g, b)
    saturation = (max_channel - min_channel) / max_channel if max_channel else 0
    return 0.9 > saturation > 0.2


def generate_contrasting_vibrant_color():
    while True:
        color = "#" + "".join(random.choice("0123456789abcdef") for _ in range(6))
        if is_contrasting(color) and is_vibrant(color):
            return color


def build_stop_hierarchy() -> Dict[str, HierarchyStop]:
    hierarchy: Dict[str, HierarchyStop] = {}

    all_stops = get_table_data("stops")

    children_by_parent: Dict[str, List] = {}
    parent_stops = []

    for stop in all_stops:
        parent_station = stop.parent_station if hasattr(stop, "parent_station") else stop[stop._fields.index("parent_station")] if "parent_station" in stop._fields else ""

        if parent_station == "":
            parent_stops.append(stop)
        else:
            if parent_station not in children_by_parent:
                children_by_parent[parent_station] = []
            children_by_parent[parent_station].append(stop)

    for parent in parent_stops:
        hierarchy[parent.stop_id] = HierarchyStop.from_dict(parent._asdict())
        children = children_by_parent.get(parent.stop_id, [])
        if children:
            hierarchy[parent.stop_id].children = [HierarchyStop.from_dict(child._asdict()) for child in children]

    return hierarchy


def get_place(coords: Tuple[float, float]):
    try:
        return geolocator.reverse(coords).raw["properties"]  # type: ignore
    except Exception:
        return {}


def query_stop_names(stop_hierarchy: Dict[str, HierarchyStop], loadingbars=True) -> Dict[str, HierarchyStop]:
    logger = logging.getLogger(name=os.path.basename(SCRIPTDIR))
    try:
        locationcache = get_table_data("location_cache")
        locationcache = {entry.stop_id: entry.name for entry in locationcache}
    except Exception:
        locationcache = {}

    # Load HST table once, handling both hst and stg column naming conventions
    globid_to_name = {}
    try:
        all_hst_data = get_table_data("hst")
        for row in all_hst_data:
            # Try hst columns first, fall back to stg columns
            globid = getattr(row, "hst_globid", None) or getattr(row, "stg_globid", None)
            name = getattr(row, "hst_name", None) or getattr(row, "stg_name", None)
            if globid and name:
                globid_to_name[globid] = name
    except Exception:
        pass
    try:
        if loadingbars:
            iterator = tqdm.tqdm(stop_hierarchy.values(), desc="Querying stop names", unit=" stops", ascii=True, dynamic_ncols=True)
        else:
            iterator = stop_hierarchy.values()
        for stop in iterator:
            if stop.stop_id in locationcache:
                stop.stop_name = locationcache[stop.stop_id]
                continue

            if stop.children is not None:
                child_ids = [child.stop_id for child in stop.children]
                child_names = [child.stop_name for child in stop.children]
            else:
                child_ids = []
                child_names = [""]

            hst_candidates = []
            for stop_id in [stop.stop_id] + child_ids:
                try:
                    globid = stop_id.split("_", 1)[1]
                    if globid in globid_to_name:
                        hst_candidates.append((globid, globid_to_name[globid]))
                except (IndexError, KeyError):
                    pass

            if hst_candidates:
                stop.stop_name = hst_candidates[0][1]
                locationcache[stop.stop_id] = stop.stop_name
            else:
                if stop.stop_name not in child_names and stop.stop_name not in child_names[0]:
                    if stop.stop_id not in locationcache:
                        logger.info("Stop not found in cache or HST: %s", stop.stop_id)
                        found = False
                        if stop.children is not None:
                            attempts = [(stop.stop_lat, stop.stop_lon)] + [(child.stop_lat, child.stop_lon) for child in stop.children]
                        else:
                            attempts = [(stop.stop_lat, stop.stop_lon)]
                        for attempt in attempts:
                            logger.info("Attempting to lookup coordinates: %s", attempt)
                            location_lookup = get_place(attempt)
                            if (
                                "name" in location_lookup
                                and location_lookup["name"] != ""
                                and location_lookup.get("osm_value") in ["bus_stop", "stop", "station", "train_station", "halt"]
                            ):
                                stop.stop_name = location_lookup["name"]
                                locationcache[stop.stop_id] = location_lookup["name"]
                                logger.info("Found stop name '%s' for stop ID %s", stop.stop_name, stop.stop_id)
                                found = True
                                break
                        if not found:
                            if child_names[0] != "":
                                stop.stop_name = child_names[0]
                                locationcache[stop.stop_id] = child_names[0]
                            else:
                                locationcache[stop.stop_id] = stop.stop_name
                elif stop.stop_name in child_names[0]:
                    locationcache[stop.stop_id] = child_names[0]
                else:
                    locationcache[stop.stop_id] = stop.stop_name
                stop.stop_name = locationcache[stop.stop_id]

            if stop.children is not None:
                for child in stop.children:
                    locationcache[child.stop_id] = locationcache[stop.stop_id]
    finally:
        update_location_cache(locationcache)

    return stop_hierarchy


def get_stop_name(stop_id: str) -> str:
    return get_table_data("location_cache", columns=["name"], filters={"stop_id": stop_id})[0]


def build_dest_list() -> Dict[str, Dict[str, str]]:
    destinations = {}
    for trip in get_table_data("trips", columns=["route_id", "direction_id"], distinct=True):
        if trip.route_id not in destinations:
            destinations[trip.route_id] = {}
        destinations[trip.route_id][f"d{trip.direction_id}"] = get_most_frequent_values(
            "trips", column="trip_headsign", filters={"route_id": trip.route_id, "direction_id": trip.direction_id}
        )[0].trip_headsign
    return destinations
