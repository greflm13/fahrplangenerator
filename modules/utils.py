import os
import copy
import json
import random

from typing import Dict, List, Set, Tuple
from collections import namedtuple

import tqdm
import pandas as pd

from svglib.svglib import Drawing
from geopy.geocoders import Photon
from pypdf import PdfReader, PdfWriter
from shapely.geometry import shape, Point

from modules.logger import logger

if __package__ is None:
    PACKAGE = ""
else:
    PACKAGE = __package__
SCRIPTDIR = os.path.abspath(os.path.dirname(__file__).removesuffix(PACKAGE))
CACHEDIR = os.path.join(SCRIPTDIR, "__pycache__")

geolocator = Photon(user_agent="fahrplan.py")

Shape = namedtuple("Shape", ["shapeid", "tripid"])
Stop = namedtuple("Stop", ["id", "name"])


def load_gtfs(folder: str, type: str) -> List[Dict]:
    """Load GTFS data."""
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


def build_shapedict(shapes: List[Dict]) -> Dict[str, List[Point]]:
    """Build a dictionary mapping shape_id to list of Point geometries."""
    shapedict: Dict[str, List] = {}
    for shapeline in tqdm.tqdm(shapes, desc="Building shapes", unit=" points", ascii=True, dynamic_ncols=True):
        sid = shapeline["shape_id"]
        if sid not in shapedict:
            logger.debug("Processing shape ID: %s", sid)
            shapedict[sid] = []
        shapedict[sid].append(Point(shapeline["shape_pt_lon"], shapeline["shape_pt_lat"], shapeline["shape_dist_traveled"]))
    return shapedict


def build_list_index(list: List[Dict], index: str) -> Dict[str, Dict]:
    """Build an index from a list of dictionaries based on a specified key."""
    data: Dict[str, Dict] = {}
    for item in list:
        data[item[index]] = item
    return data


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


def prepare_linedraw_info(
    shapedict: Dict[str, List[Point]],
    stop_times: Dict[str, List[Dict]],
    trips: List[Dict],
    routes: Dict[str, Dict],
    stops: Dict[str, Dict[str, str]],
    line,
    direction,
    ourstop: List[str],
):
    """Prepare line drawing information for selected shapes."""
    shapes: Set[Shape] = set()
    stop_points: Set[Tuple] = set()
    end_stop_ids: Set[str] = set()
    for trip in trips:
        if routes[trip["route_id"]]["route_short_name"] == line and "d" + trip["direction_id"] == direction:
            if trip["shape_id"] != "":
                shapes.add(Shape(trip["shape_id"], trip["trip_id"]))
    linedrawinfo = {"shapes": [], "points": [], "endstops": []}
    for shap in shapes:
        times = stop_times[shap.tripid]
        for timeidx, time in enumerate(times):
            if time["stop_id"] in ourstop:
                shape_dist_traveled = time["shape_dist_traveled"]
                geometry = shapedict[shap.shapeid]
                for geoidx, point in enumerate(geometry):
                    if point.z == float(shape_dist_traveled):
                        geo = geometry[geoidx:]
                        tim = times[timeidx:]
                        stop_points.update(
                            [
                                (
                                    Point(float(stops[stop["stop_id"]]["stop_lon"]), float(stops[stop["stop_id"]]["stop_lat"])),
                                    Stop(stop["stop_id"], get_stop_name(stop["stop_id"], stops)),
                                )
                                for stop in tim
                            ]
                        )
                        if len(geo) != 1:
                            linedrawinfo["shapes"].append({"geometry": shape({"type": "LineString", "coordinates": geo})})
                        endstop = stops[times[-1]["stop_id"]]
                        end_stop_ids.add(get_stop_name(endstop["stop_id"], stops))
    linedrawinfo["points"] = list(stop_points)
    linedrawinfo["endstops"] = list(end_stop_ids)
    return linedrawinfo


def create_merged_pdf(pages: list[str], path: str):
    output = PdfWriter()

    for page in pages:
        pdf = PdfReader(page)
        output.add_page(pdf.pages[0])

    output.write(path)


def merge_dicts(a: dict[str, dict[str, dict[str, list[dict[str, str]]]]], b: dict[str, dict[str, dict[str, list[dict[str, str]]]]]):
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


def dict_set(lst: list[dict[str, str]]):
    seen = []
    setlike = []
    for d in lst:
        signature = {"time": d["time"], "line": d["line"], "dire": d["dire"]}
        if signature not in seen:
            seen.append(signature)
            setlike.append(d)
    return setlike


def most_frequent(lst: list[str]):
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


def merge(lst: list[str]):
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


def find_correct_stop_name(stops):
    parent_stops = []
    stopss = {}
    for stop in tqdm.tqdm(parent_stops, desc="Fixing parent stop names", unit=" stops", ascii=True, dynamic_ncols=True):
        stop["stop_ids"] = [stop["stop_id"]]
        child_names = [s["stop_name"] for s in stops if s.get("parent_station", "") == stop["stop_id"]]
        if child_names:
            child_name = most_frequent(merge(child_names))
            if stop["stop_name"] not in child_name and child_name not in stop["stop_name"]:
                child_name = child_name + " " + stop["stop_name"]
            stop["stop_name"] = max(child_name, stop["stop_name"], key=len)
        if stopss.get(stop["stop_name"], False):
            stopss[stop["stop_name"]]["stop_ids"].append(stop["stop_id"])
        else:
            stopss[stop["stop_name"]] = stop


def build_stop_hierarchy(stops: List[Dict[str, str]]):
    hierarchy = {}
    for stop in stops:
        if stop["parent_station"] == "":
            if stop["stop_id"] in hierarchy.keys():
                hierarchy[stop["stop_id"]].update(stop)
            else:
                hierarchy[stop["stop_id"]] = stop
        else:
            if stop["parent_station"] not in hierarchy.keys():
                hierarchy[stop["parent_station"]] = {"children": [stop]}
            elif "children" not in hierarchy[stop["parent_station"]].keys():
                hierarchy[stop["parent_station"]].update({"children": [stop]})
            else:
                hierarchy[stop["parent_station"]]["children"].append(stop)
    return hierarchy


def get_place(coords: Tuple[float, float]):
    return geolocator.reverse(coords).raw["properties"]  # type: ignore


def query_stop_names(stop_hierarchy: Dict, hst_map=None):
    if hst_map is None:
        hst_map = {}
    loccache = os.path.join(CACHEDIR, "location_cache")
    os.makedirs(CACHEDIR, exist_ok=True)
    if not os.path.exists(loccache):
        with open(loccache, "x", encoding="utf-8") as f:
            f.write(json.dumps({}))
    with open(loccache, "r", encoding="utf-8") as f:
        locationcache = json.loads(f.read())
    for stop in tqdm.tqdm(stop_hierarchy.values(), desc="Querying stop names", unit=" stops", ascii=True, dynamic_ncols=True):
        if "children" in stop:
            child_ids = [child["stop_id"] for child in stop["children"]]
        else:
            child_ids = []
        if stop["stop_id"] not in locationcache.keys() and stop["stop_id"] not in hst_map.keys() and not any(child in hst_map for child in child_ids):
            if "children" in stop:
                child_names = [child["stop_name"] for child in stop["children"]]
            else:
                child_names = [""]
            if stop["stop_name"] not in child_names and stop["stop_name"] not in child_names[0]:
                if stop["stop_id"] not in locationcache:
                    found = False
                    if "children" in stop:
                        attempts = [(float(stop["stop_lat"]), float(stop["stop_lon"]))] + [(float(child["stop_lat"]), float(child["stop_lon"])) for child in stop["children"]]
                    else:
                        attempts = [(float(stop["stop_lat"]), float(stop["stop_lon"]))]
                    for attempt in attempts:
                        location_lookup = get_place(attempt)
                        if (
                            "name" in location_lookup.keys()
                            and location_lookup["name"] != ""
                            and location_lookup["osm_value"] in ["bus_stop", "stop", "station", "train_station", "halt"]
                        ):
                            stop["stop_name"] = location_lookup["name"]
                            locationcache[stop["stop_id"]] = location_lookup["name"]
                            found = True
                            break
                    if not found:
                        if child_names[0] != "":
                            stop["stop_name"] = child_names[0]
                            locationcache[stop["stop_id"]] = child_names[0]
                        else:
                            locationcache[stop["stop_id"]] = stop["stop_name"]
            elif stop["stop_name"] in child_names[0]:
                locationcache[stop["stop_id"]] = child_names[0]
            else:
                locationcache[stop["stop_id"]] = stop["stop_name"]
            stop["stop_name"] = locationcache[stop["stop_id"]]
            with open(loccache, "w", encoding="utf-8") as f:
                f.write(json.dumps(locationcache, indent=2))
        elif stop["stop_id"] in hst_map.keys():
            stop["stop_name"] = hst_map[stop["stop_id"]]
            locationcache[stop["stop_id"]] = hst_map[stop["stop_id"]]
        elif any(child in hst_map for child in child_ids):
            for child in child_ids:
                if child in hst_map:
                    stop["stop_name"] = hst_map[child]
                    locationcache[stop["stop_id"]] = hst_map[child]
                    break
        else:
            stop["stop_name"] = locationcache[stop["stop_id"]]
        if "children" in stop:
            for child in stop["children"]:
                locationcache[child["stop_id"]] = locationcache[stop["stop_id"]]

    with open(loccache, "w", encoding="utf-8") as f:
        f.write(json.dumps(locationcache, indent=2))

    return stop_hierarchy


def get_stop_name(stop_id: str, stops) -> str:
    loccache = os.path.join(CACHEDIR, "location_cache")
    with open(loccache, "r", encoding="utf-8") as f:
        locationcache = json.loads(f.read())
    if stops[stop_id]["parent_station"] != "":
        return locationcache[stops[stop_id]["parent_station"]]
    else:
        return locationcache[stop_id]


def load_hst_json(json: Dict):
    id_mapping = {}
    for steig in json["features"]:
        id_mapping[steig["properties"]["stg_globid"]] = steig["properties"]["stg_name"]
    return id_mapping
