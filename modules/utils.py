import os
import copy
import random

from typing import Dict, List, Set, Tuple
from collections import namedtuple

import pandas as pd

from svglib.svglib import Drawing
from pypdf import PdfReader, PdfWriter
from shapely.geometry import shape, Point

from modules.logger import logger

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
    for shapeline in shapes:
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
    for trip in trips:
        if routes[trip["route_id"]]["route_short_name"] == line and "d" + trip["direction_id"] == direction:
            shapes.add(Shape(trip["shape_id"], trip["trip_id"]))
    linedrawinfo = {"shapes": [], "points": [], "rows": []}
    for shap in shapes:
        times = stop_times[shap.tripid]
        for timeidx, time in enumerate(times):
            if time["stop_id"] in ourstop:
                shape_dist_traveled = time["shape_dist_traveled"]
                geometry = shapedict[shap.shapeid]
                for geoidx, point in enumerate(geometry):
                    if point.z == float(shape_dist_traveled):
                        geo = [Point(point.x, point.y) for point in geometry[geoidx:]]
                        tim = times[timeidx:]
                        stop_points.update(
                            [
                                (
                                    Point(float(stops[stop["stop_id"]]["stop_lon"]), float(stops[stop["stop_id"]]["stop_lat"])),
                                    Stop(stop["stop_id"], stops[stop["stop_id"]]["stop_name"]),
                                )
                                for stop in tim
                            ]
                        )
                        linedrawinfo["shapes"].append({"geometry": shape({"type": "LineString", "coordinates": geo})})
                        break
    linedrawinfo["points"] = list(stop_points)
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
