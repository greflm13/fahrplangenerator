#!/usr/bin/env python

import os
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
from reportlab.graphics import renderPDF
from reportlab.lib import colors, pagesizes
from svglib.svglib import svg2rlg, Drawing

import modules.utils as utils

from modules.logger import logger
from modules.map import draw_map

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
    stop: dict[str, str],
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
        drawing = svg2rlg(logo.name)
        if isinstance(drawing, Drawing):
            drawing = utils.scale(drawing, 0.5)
            renderPDF.draw(drawing, pdf, x=1041, y=15)
        else:
            logger.warning("Logo is not a drawing")
            pdf.setFont("logo", 20)
            pdf.setFillColor(colors.black)
            pdf.drawRightString(x=1108, y=23.5, text="</srgn>")
    elif isinstance(logo, str):
        pdf.setFont("logo", 20)
        pdf.setFillColor(colors.black)
        pdf.drawRightString(x=1108, y=23.5, text=logo)

    pdf.setFont("foot", 17)
    pdf.setFillColor(colors.black)
    pdf.drawString(x=80, y=23.5, text=stop["stop_name"])

    pdf.save()
    logger.info("Done.")
    return path


def main():
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter)
    parser.add_argument("-i", "--input", help="Input folder(s)", action="extend", nargs="+", required=True, dest="input")
    parser.add_argument("-c", "--color", help="Timetable color", type=str, required=False, dest="color", default="random")
    parser.add_argument("-o", "--output", help="Output file", type=str, required=False, dest="output", default="fahrplan.pdf")
    parser.add_argument("-m", "--map", help="Generate maps", action="store_true", dest="map")
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

    stopss = {}
    parent_stops = [stop for stop in stops if stop.get("parent_station", "") == ""]
    for stop in tqdm.tqdm(parent_stops, desc="Fixing parent stop names", unit=" stops", ascii=True, dynamic_ncols=True):
        stop["stop_ids"] = [stop["stop_id"]]
        child_names = [s["stop_name"] for s in stops if s.get("parent_station", "") == stop["stop_id"]]
        if child_names:
            child_name = utils.most_frequent(utils.merge(child_names))
            if stop["stop_name"] not in child_name and child_name not in stop["stop_name"]:
                child_name = child_name + " " + stop["stop_name"]
            stop["stop_name"] = max(child_name, stop["stop_name"], key=len)
        if stopss.get(stop["stop_name"], False):
            stopss[stop["stop_name"]]["stop_ids"].append(stop["stop_id"])
        else:
            stopss[stop["stop_name"]] = stop
    logger.info("loaded data")

    choices = sorted({stop["stop_name"] for stop in parent_stops})
    choice = questionary.autocomplete("Haltestelle/Bahnhof: ", choices=choices, match_middle=True, validate=lambda val: val in choices, style=custom_style).ask()
    ourstop = stopss[choice]

    logger.info("computing our stops")
    ourstops = []
    for stop in tqdm.tqdm(stops, desc="Finding stops", unit=" stops", ascii=True, dynamic_ncols=True):
        for stopid in ourstop.get("stop_ids", [ourstop["stop_id"]]):
            if stop.get("parent_station", "") == stopid and stop not in ourstops:
                ourstops.append(stop)
            elif stop["stop_id"] == stopid and stop not in ourstops:
                ourstops.append(stop)
    logger.info("computing our times")
    ourtimes = []
    for time in tqdm.tqdm(stop_times, desc="Finding stop times", unit=" stop times", ascii=True, dynamic_ncols=True):
        if time["stop_id"] in [stop["stop_id"] for stop in ourstops] and time.get("pickup_type", "0") == "0":
            ourtimes.append(time)
    logger.info("computing our trips")
    ourtrips = []
    for trip in tqdm.tqdm(trips, desc="Finding trips", unit=" trips", ascii=True, dynamic_ncols=True):
        if trip["trip_id"] in [time["trip_id"] for time in ourtimes]:
            ourtrips.append(trip)
    logger.info("computing our services")
    ourservs = []
    for serv in tqdm.tqdm(calendar, desc="Finding services", unit=" services", ascii=True, dynamic_ncols=True):
        if serv["service_id"] in [trip["service_id"] for trip in ourtrips]:
            ourservs.append(serv)
    logger.info("computing our routes")
    ourroute = []
    for rout in tqdm.tqdm(routes, desc="Finding routes", unit=" routes", ascii=True, dynamic_ncols=True):
        if rout["route_id"] in [trip["route_id"] for trip in ourtrips]:
            ourroute.append(rout)

    logger.info("playing variable shuffle")

    if ourstops == []:
        print(f'Stop "{ourstop}" not found!')
        return

    stops = utils.build_list_index(stops, "stop_id")
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
                    "line": selected_routes[trip["route_id"]]["route_short_name"],
                    "dire": selected_trips[trip["trip_id"]]["direction_id"],
                    "stop": selected_stops[selected_stop_times[trip["trip_id"]]["stop_id"]]["stop_name"],
                }
            )
        if calendar[trip["service_id"]]["saturday"] == "1":
            sat.append(
                {
                    "dest": trip["trip_headsign"],
                    "time": selected_stop_times[trip["trip_id"]]["arrival_time"][:-3],
                    "line": selected_routes[trip["route_id"]]["route_short_name"],
                    "dire": selected_trips[trip["trip_id"]]["direction_id"],
                    "stop": selected_stops[selected_stop_times[trip["trip_id"]]["stop_id"]]["stop_name"],
                }
            )
        if calendar[trip["service_id"]]["sunday"] == "1":
            sun.append(
                {
                    "dest": trip["trip_headsign"],
                    "time": selected_stop_times[trip["trip_id"]]["arrival_time"][:-3],
                    "line": selected_routes[trip["route_id"]]["route_short_name"],
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
            res = requests.get("https://files.sorogon.eu/logo-fixed.svg")
            tmpfile = tempfile.NamedTemporaryFile()
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
                destlist = []
                for time in mondict.get(line, satdict.get(line, sundict.get(line, {}))).get(k, satdict.get(line, {}).get(k, sundict.get(line, {}).get(k, {}))).values():
                    destlist.extend([t["dest"] for t in time])
                dest = utils.most_frequent(destlist)
                if dest != ourstop:
                    page = pages.get(line, {}).get(k, {})
                    if not isinstance(page, str):
                        page = tempfile.mkstemp(suffix=".pdf", dir=tmpdir)[1]
                    pages[line][k] = create_page(
                        line,
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


if __name__ == "__main__":
    main()
