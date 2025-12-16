#!/usr/bin/env python

import os
import json
import argparse
import tempfile

import PIL.Image
import tqdm
import requests
import questionary
from rich_argparse import RichHelpFormatter

from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib import colors, pagesizes
from reportlab.lib.utils import ImageReader

import modules.utils as utils

from modules.map import draw_map
from modules.logger import logger
from modules.datatypes import HierarchyStop

PIL.Image.MAX_IMAGE_PIXELS = 9331200000
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

pdfmetrics.registerFont(TTFont("header", "FiraSans-Black.ttf"))
pdfmetrics.registerFont(TTFont("hour", "FiraSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("add", "FiraSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("foot", "FiraSans-Thin.ttf"))


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


def addtimes(pdf: Canvas, daytimes: dict[str, list[dict[str, str]]], day: str, posy: float, accent: colors.Color, dest: str, addstops: dict[str, int] = {"num": 1}):
    logger.info(f"Add {day}")

    pdg = pdf

    pre = pdg, posy, addstops.copy(), 0

    # Color Rectangle
    pdf.setFillColor(accent)
    pdf.setStrokeColor(colors.black)
    pdf.setLineWidth(1.2)
    pdf.rect(x=80, y=posy, width=1028, height=30, fill=1)

    # Hours Text
    pdf.setFont("hour", 17)
    pdf.setFillColor(colors.white)
    pdf.drawCentredString(x=165, y=posy + 8.5, text=day)

    spacing = 858 / len(daytimes.keys())
    posx = 250 - spacing / 2
    times = 0
    for k, v in daytimes.items():
        posx += spacing
        pdf.setFont("hour", 17)
        pdf.setFillColor(colors.white)
        pdf.drawCentredString(x=posx, y=posy + 8.5, text=k[1:])
        space = 0
        loctimes = 0
        for time in v:
            sp = time["stop"].split()
            if len(sp[-1]) < 2:
                sp.pop()
            stopn = " ".join(sp)
            if time["dest"] != stopn:
                loctimes += 1
                # Minutes Text
                pdf.setFont("hour", 17)
                pdf.setFillColor(colors.black)
                pdf.drawCentredString(x=posx, y=posy - 20 + space, text=time["time"][3:])
                if time["dest"] != dest:
                    if not addstops.get(time["dest"], False):
                        addstops[time["dest"]] = addstops["num"]
                        addstops["num"] += 1
                    pdf.setFont("add", 8)
                    pdf.setFontSize(8)
                    pdf.drawCentredString(x=posx + 12, y=posy - 22 + space, text=ALPHABET[addstops[time["dest"]] - 1])
                    pdf.setFont("hour", 17)
                space -= 25
        times = max(times, loctimes)
    posx = 250

    if times == 0:
        return pre

    # Lines
    pdf.line(x1=80, y1=posy + 30, x2=80, y2=posy - times * 25 - 3.5)
    pdf.line(x1=250, y1=posy + 30, x2=250, y2=posy - times * 25 - 3.5)
    pdf.line(x1=79.4, y1=posy - times * 25 - 3.5, x2=1108.6, y2=posy - times * 25 - 3.5)
    for _ in daytimes.keys():
        posx += spacing
        pdf.line(x1=posx, y1=posy + 30, x2=posx, y2=posy - times * 25 - 3.5)

    posy -= 60 + times * 25
    return pdf, posy, addstops, times


def create_page(
    line: str,
    dest: str,
    stop: HierarchyStop,
    path: str,
    montimes: dict[str, list[dict[str, str]]],
    sattimes: dict[str, list[dict[str, str]]],
    suntimes: dict[str, list[dict[str, str]]],
    color: str,
    logo: tempfile._TemporaryFileWrapper | str | None = "</srgn>",
):
    logger.info(f"Create page for {line} - {dest}")
    limit = 0
    times = 0
    for i in montimes.values():
        times = max(times, len(i))
    limit += times
    times = 0
    for i in sattimes.values():
        times = max(times, len(i))
    limit += times
    times = 0
    for i in suntimes.values():
        times = max(times, len(i))
    limit += times

    if limit > 20:
        pagesize = pagesizes.portrait(pagesizes.A4)
        movey = 840
    else:
        pagesize = pagesizes.landscape(pagesizes.A4)
        movey = 0
    pdf = Canvas(path, pagesize=pagesize)
    pdf.scale(pagesize[0] / 1188, pagesize[1] / (840 + movey))
    accent = colors.HexColor(color)

    # Header
    pdf.setFont("header", 48)
    pdf.setFillColor(accent)
    pdf.drawCentredString(x=1188 / 2, y=760 + movey, text=f"{line} - {dest}")

    posy = 690 + movey

    addstops = {"num": 1}

    times = 0

    if len(montimes) > 0:
        loctime = 0
        for v in montimes.values():
            for time in v:
                sp = time["stop"].split()
                if len(sp[-1]) < 2:
                    sp.pop()
                stopn = " ".join(sp)
                if time["dest"] != stopn:
                    loctime += 1
        if loctime > 0:
            pdf, posy, addstops, loctimes = addtimes(pdf, montimes, "Montag-Freitag", posy, accent, dest, addstops)
            times = max(times, loctimes)

    if len(sattimes) > 0:
        loctime = 0
        for v in sattimes.values():
            for time in v:
                sp = time["stop"].split()
                if len(sp[-1]) < 2:
                    sp.pop()
                stopn = " ".join(sp)
                if time["dest"] != stopn:
                    loctime += 1
        if loctime > 0:
            pdf, posy, addstops, loctimes = addtimes(pdf, sattimes, "Samstag", posy, accent, dest, addstops)
            times = max(times, loctimes)

    if len(suntimes) > 0:
        loctime = 0
        for v in suntimes.values():
            for time in v:
                sp = time["stop"].split()
                if len(sp[-1]) < 2:
                    sp.pop()
                stopn = " ".join(sp)
                if time["dest"] != stopn:
                    loctime += 1
        if loctime > 0:
            pdf, posy, addstops, loctimes = addtimes(pdf, suntimes, "Sonntag", posy, accent, dest, addstops)
            times = max(times, loctimes)

    if times == 0:
        return None

    addstops.pop("num")

    if len(addstops.keys()) > 0:
        pdf.setFont("add", 10)
        pdf.setFillColor(colors.black)
        y = posy + 34
        for k, v in addstops.items():
            pdf.drawString(x=80, y=y, text=f"{ALPHABET[v - 1]}: Bis {k}")
            y -= 10

    if isinstance(logo, tempfile._TemporaryFileWrapper):
        image = ImageReader(logo.name)
        pdf.drawImage(image, x=1041, y=15, width=80, height=30, preserveAspectRatio=True)
    elif isinstance(logo, str):
        pdf.setFont("logo", 20)
        pdf.setFillColor(colors.black)
        pdf.drawRightString(x=1108, y=23.5, text=logo)

    pdf.setFont("foot", 17)
    pdf.setFillColor(colors.black)
    pdf.drawString(x=80, y=23.5, text=stop.stop_name)

    pdf.save()
    logger.info("Done.")
    return path


def compute(ourstop: HierarchyStop, stop_times, trips: list[dict], calendar, routes: list[dict], stops: dict[str, dict], args, destinations: dict[str, dict[str, str]], shapedict):
    logger.info("computing our stops")
    ourstops = [ourstop.to_dict()]
    if ourstop.children is not None:
        ourstops.extend(ourstops[0]["children"])
        del ourstops[0]["children"]
    logger.info("computing our times")
    ourtimes = []
    stopids = [stop["stop_id"] for stop in ourstops]
    for time in tqdm.tqdm(stop_times, desc="Finding stop times", unit=" stop times", ascii=True, dynamic_ncols=True):
        if time["stop_id"] in stopids and time.get("pickup_type", "0") == "0":
            ourtimes.append(time)
    logger.info("computing our trips")
    ourtrips = []
    times = [time["trip_id"] for time in ourtimes]
    for trip in tqdm.tqdm(trips, desc="Finding trips", unit=" trips", ascii=True, dynamic_ncols=True):
        if trip["trip_id"] in times:
            ourtrips.append(trip)
    logger.info("computing our services")
    ourservs = []
    services = [trip["service_id"] for trip in ourtrips]
    for serv in tqdm.tqdm(calendar, desc="Finding services", unit=" services", ascii=True, dynamic_ncols=True):
        if serv["service_id"] in services:
            ourservs.append(serv)
    logger.info("computing our routes")
    ourroute = []
    routeids = [trip["route_id"] for trip in ourtrips]
    for rout in tqdm.tqdm(routes, desc="Finding routes", unit=" routes", ascii=True, dynamic_ncols=True):
        if rout["route_id"] in routeids:
            ourroute.append(rout)

    logger.info("playing variable shuffle")

    if ourstops == []:
        print(f'Stop "{ourstop}" not found!')
        return

    selected_stops = utils.build_list_index(ourstops, "stop_id")
    selected_stop_times = utils.build_list_index(ourtimes, "trip_id")
    stop_times = utils.build_stop_times_index(stop_times)
    selected_trips = utils.build_list_index(ourtrips, "trip_id")
    calendar = utils.build_list_index(ourservs, "service_id")
    selected_routes = utils.build_list_index(ourroute, "route_id")

    mon: list[dict[str, str]] = []
    sat: list[dict[str, str]] = []
    sun: list[dict[str, str]] = []

    for trip in tqdm.tqdm(ourtrips, desc="Sorting trips", unit=" trips", ascii=True, dynamic_ncols=True):
        if calendar[trip["service_id"]]["monday"] == "1":
            mon.append(
                {
                    "dest": trip["trip_headsign"],
                    "time": selected_stop_times[trip["trip_id"]]["arrival_time"][:-3],
                    "line": trip["route_id"],
                    "dire": selected_trips[trip["trip_id"]]["direction_id"],
                    "stop": selected_stops[selected_stop_times[trip["trip_id"]]["stop_id"]]["stop_name"],
                }
            )
        if calendar[trip["service_id"]]["saturday"] == "1":
            sat.append(
                {
                    "dest": trip["trip_headsign"],
                    "time": selected_stop_times[trip["trip_id"]]["arrival_time"][:-3],
                    "line": trip["route_id"],
                    "dire": selected_trips[trip["trip_id"]]["direction_id"],
                    "stop": selected_stops[selected_stop_times[trip["trip_id"]]["stop_id"]]["stop_name"],
                }
            )
        if calendar[trip["service_id"]]["sunday"] == "1":
            sun.append(
                {
                    "dest": trip["trip_headsign"],
                    "time": selected_stop_times[trip["trip_id"]]["arrival_time"][:-3],
                    "line": trip["route_id"],
                    "dire": selected_trips[trip["trip_id"]]["direction_id"],
                    "stop": selected_stops[selected_stop_times[trip["trip_id"]]["stop_id"]]["stop_name"],
                }
            )

    mon = sorted(utils.dict_set(mon), key=lambda x: x["time"])
    sat = sorted(utils.dict_set(sat), key=lambda x: x["time"])
    sun = sorted(utils.dict_set(sun), key=lambda x: x["time"])

    mondict: dict[str, dict[str, dict[str, list[dict[str, str]]]]] = {}
    satdict: dict[str, dict[str, dict[str, list[dict[str, str]]]]] = {}
    sundict: dict[str, dict[str, dict[str, list[dict[str, str]]]]] = {}

    for trip in tqdm.tqdm(mon, desc="Indexing Monday-Friday trips", unit=" trips", ascii=True, dynamic_ncols=True):
        if not mondict.get(trip["line"], False):
            mondict[trip["line"]] = {}
        if not mondict.get(trip["line"], {}).get(f"d{trip['dire']}", False):
            mondict[trip["line"]][f"d{trip['dire']}"] = {}
        for i in range(0, 24):
            if not mondict[trip["line"]][f"d{trip['dire']}"].get(f"t{i:02}", False):
                mondict[trip["line"]][f"d{trip['dire']}"][f"t{i:02}"] = []
        mondict[trip["line"]][f"d{trip['dire']}"].setdefault(f"t{trip['time'][:2]}", []).append(trip)

    for trip in tqdm.tqdm(sat, desc="Indexing Saturday trips", unit=" trips", ascii=True, dynamic_ncols=True):
        if not satdict.get(trip["line"], False):
            satdict[trip["line"]] = {}
        if not satdict.get(trip["line"], {}).get(f"d{trip['dire']}", False):
            satdict[trip["line"]][f"d{trip['dire']}"] = {}
        for i in range(0, 24):
            if not satdict[trip["line"]][f"d{trip['dire']}"].get(f"t{i:02}", False):
                satdict[trip["line"]][f"d{trip['dire']}"][f"t{i:02}"] = []
        satdict[trip["line"]][f"d{trip['dire']}"].setdefault(f"t{trip['time'][:2]}", []).append(trip)

    for trip in tqdm.tqdm(sun, desc="Indexing Sunday trips", unit=" trips", ascii=True, dynamic_ncols=True):
        if not sundict.get(trip["line"], False):
            sundict[trip["line"]] = {}
        if not sundict.get(trip["line"], {}).get(f"d{trip['dire']}", False):
            sundict[trip["line"]][f"d{trip['dire']}"] = {}
        for i in range(0, 24):
            if not sundict[trip["line"]][f"d{trip['dire']}"].get(f"t{i:02}", False):
                sundict[trip["line"]][f"d{trip['dire']}"][f"t{i:02}"] = []
        sundict[trip["line"]][f"d{trip['dire']}"].setdefault(f"t{trip['time'][:2]}", []).append(trip)

    pages: dict[str, dict[str, str | None]] = {}

    lines = utils.merge_dicts(utils.merge_dicts(mondict, satdict), sundict)

    if args.logo:
        try:
            logger.info("getting logo")
            res = requests.get("https://files.sorogon.eu/logo.png")
            tmpfile = tempfile.NamedTemporaryFile(suffix=".png")
            tmpfile.write(res.content)
            tmpfile.flush()
        except requests.exceptions.ConnectionError:
            logger.info("fallback to string logo")
            try:
                pdfmetrics.registerFont(TTFont("logo", "BarlowCondensed_Thin.ttf"))
            except Exception:
                try:
                    pdfmetrics.registerFont(TTFont("logo", "BarlowCondensed-Thin.ttf"))
                except Exception:
                    pdfmetrics.registerFont(TTFont("logo", "FiraSans-Thin.ttf"))
            tmpfile = "</srgn>"
    else:
        tmpfile = None

    try:
        tmpdir = tempfile.mkdtemp()
        for line, dires in tqdm.tqdm(lines.items(), desc="Creating pages", unit=" lines", ascii=True, dynamic_ncols=True):
            if args.color == "random":
                color = utils.generate_contrasting_vibrant_color()
            else:
                color = args.color
            if not pages.get(line, False):
                pages[line] = {}
            for k in dires.keys():
                dest = destinations[line][k]
                if dest != ourstop.stop_name:
                    page = pages.get(line, {}).get(k, {})
                    if not isinstance(page, str):
                        page = tempfile.mkstemp(suffix=".pdf", dir=tmpdir)[1]
                    pages[line][k] = create_page(
                        selected_routes[line]["route_short_name"],
                        dest,
                        ourstop,
                        page,
                        mondict.get(line, {}).get(k, {}),
                        satdict.get(line, {}).get(k, {}),
                        sundict.get(line, {}).get(k, {}),
                        color,
                        tmpfile,
                    )
                    if args.map:
                        mappage = tempfile.mkstemp(suffix=".pdf", dir=tmpdir)[1]
                        pages[line][k + "map"] = draw_map(
                            page=mappage,
                            stop=ourstop,
                            logo=tmpfile,
                            routes=utils.prepare_linedraw_info(shapedict, stop_times, ourtrips, selected_routes, stops, line, k, [stop["stop_id"] for stop in ourstops]),
                            color=color,
                            label_rotation=15,
                            tmpdir=tmpdir,
                            map_provider=args.map_provider,
                            dpi=args.map_dpi,
                        )

        pagelst: list[str] = []
        for line in tqdm.tqdm(pages.values(), desc="Collecting pages", unit=" lines", ascii=True, dynamic_ncols=True):
            for dire in line.values():
                if dire is not None:
                    pagelst.append(dire)

        utils.create_merged_pdf(pagelst, args.output)

        for line in pages.values():
            for dire in line.values():
                if dire is not None:
                    os.remove(dire)
    finally:
        for file in os.listdir(tmpdir):
            try:
                os.remove(file)
            except Exception:
                pass
        try:
            os.removedirs(tmpdir)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter)
    parser.add_argument("-i", "--input", help="Input folder(s)", action="extend", nargs="+", required=True, dest="input")
    parser.add_argument("-c", "--color", help="Timetable color", type=str, required=False, dest="color", default="random")
    parser.add_argument("-o", "--output", help="Output file", type=str, required=False, dest="output", default="fahrplan.pdf")
    parser.add_argument("-m", "--map", help="Generate maps", action="store_true", dest="map")
    parser.add_argument("-j", "--stop-name-json", help="Stop name mapping json", required=False, type=str, dest="mapping_json")
    parser.add_argument("--dpi", help="map dpi", type=int, dest="map_dpi")
    parser.add_argument("--no-logo", action="store_false", dest="logo")
    parser.add_argument(
        "--map-provider",
        type=str,
        choices=["BasemapAT", "OPNVKarte", "OSM", "OSMDE", "ORM", "OTM", "UN", "SAT"],
        default="BasemapAT",
        help="Map provider for the basemap (default: BasemapAT)",
        dest="map_provider",
    )
    args = parser.parse_args()

    stops, stop_times, trips, calendar, routes, shapes = [], [], [], [], [], []

    for folder in tqdm.tqdm(args.input, desc="Loading data", unit=" folders", ascii=True, dynamic_ncols=True):
        if os.path.isdir(folder):
            stops.extend(utils.load_gtfs(folder, "stops"))
            stop_times.extend(utils.load_gtfs(folder, "stop_times"))
            trips.extend(utils.load_gtfs(folder, "trips"))
            calendar.extend(utils.load_gtfs(folder, "calendar"))
            routes.extend(utils.load_gtfs(folder, "routes"))
            if args.map:
                shapes.extend(utils.load_gtfs(folder, "shapes"))
        else:
            logger.error("Input folder %s does not exist", folder)

    if args.map:
        shapedict = utils.build_shapedict(shapes)
    else:
        shapedict = None

    if args.mapping_json:
        with open(args.mapping_json, "r", encoding="utf-8") as f:
            hst_map = utils.load_hst_json(json.loads(f.read()))
    else:
        hst_map = {}

    stop_hierarchy = utils.build_stop_hierarchy(stops)
    stop_hierarchy = utils.query_stop_names(stop_hierarchy, hst_map)
    destinations = utils.build_dest_list(trips)
    stopss = {}
    for stop in stop_hierarchy.values():
        if stop.stop_name not in stopss:
            stopss[stop.stop_name] = [stop.stop_id]
        else:
            stopss[stop.stop_name].append(stop.stop_id)
    stops = utils.build_list_index(stops, "stop_id")

    logger.info("loaded data")

    choices = sorted({stop.stop_name for stop in stop_hierarchy.values()})
    while True:
        choice = questionary.autocomplete("Haltestelle/Bahnhof: ", choices=choices, match_middle=True, validate=lambda val: val in choices, style=custom_style).ask()
        ourstop: HierarchyStop
        if len(stopss[choice]) == 1:
            ourstop = stop_hierarchy[stopss[choice][0]]
        else:
            combined_children = []
            for stop_id in stopss[choice]:
                stop = stop_hierarchy[stop_id]
                if stop.children is not None:
                    combined_children.extend(stop.children)
            ourstop = stop_hierarchy[stopss[choice][0]]
            ourstop.children = combined_children

        compute(ourstop, stop_times, trips, calendar, routes, stops, args, destinations, shapedict)


if __name__ == "__main__":
    main()
