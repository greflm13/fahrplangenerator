#!/usr/bin/env python

import os
import logging
import asyncio
import argparse
import tempfile
from typing import Any

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
import modules.db as db

from modules.map import draw_map, MAP_PROVIDERS
from modules.logger import setup_logger, rotate_log_file
from modules.datatypes import HierarchyStop, Routedata

# Constants for file paths and exclusions
if __package__ is None:
    PACKAGE = ""
else:
    PACKAGE = __package__
SCRIPTDIR = os.path.abspath(os.path.dirname(__file__).removesuffix(PACKAGE))
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


def addtimes(
    pdf: Canvas,
    daytimes: dict[str, list[Routedata]],
    day: str,
    posy: float,
    accent: colors.Color,
    dest: str,
    addstops: dict[str, int] = {"num": 1},
):
    logger = logging.getLogger(name=os.path.basename(SCRIPTDIR))
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
            sp = time.stop.split()
            if len(sp[-1]) < 2:
                sp.pop()
            stopn = " ".join(sp)
            if time.dest != stopn:
                loctimes += 1
                # Minutes Text
                pdf.setFont("hour", 17)
                pdf.setFillColor(colors.black)
                pdf.drawCentredString(x=posx, y=posy - 20 + space, text=time.time[3:])
                if time.dest != dest:
                    if not addstops.get(time.dest, False):
                        addstops[time.dest] = addstops["num"]
                        addstops["num"] += 1
                    pdf.setFont("add", 8)
                    pdf.setFontSize(8)
                    pdf.drawCentredString(x=posx + 12, y=posy - 22 + space, text=ALPHABET[addstops[time.dest] - 1])
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
    stop_name: str,
    path: str,
    montimes: dict[str, list[Routedata]],
    sattimes: dict[str, list[Routedata]],
    suntimes: dict[str, list[Routedata]],
    color: str,
    logo: tempfile._TemporaryFileWrapper | str | None = "</srgn>",
    logger=logging.getLogger(name=os.path.basename(SCRIPTDIR)),
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
                sp = time.stop.split()
                if len(sp[-1]) < 2:
                    sp.pop()
                stopn = " ".join(sp)
                if time.dest != stopn:
                    loctime += 1
        if loctime > 0:
            pdf, posy, addstops, loctimes = addtimes(pdf, montimes, "Montag-Freitag", posy, accent, dest, addstops)
            times = max(times, loctimes)

    if len(sattimes) > 0:
        loctime = 0
        for v in sattimes.values():
            for time in v:
                sp = time.stop.split()
                if len(sp[-1]) < 2:
                    sp.pop()
                stopn = " ".join(sp)
                if time.dest != stopn:
                    loctime += 1
        if loctime > 0:
            pdf, posy, addstops, loctimes = addtimes(pdf, sattimes, "Samstag", posy, accent, dest, addstops)
            times = max(times, loctimes)

    if len(suntimes) > 0:
        loctime = 0
        for v in suntimes.values():
            for time in v:
                sp = time.stop.split()
                if len(sp[-1]) < 2:
                    sp.pop()
                stopn = " ".join(sp)
                if time.dest != stopn:
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
    pdf.drawString(x=80, y=23.5, text=stop_name)

    pdf.save()
    logger.info("Done.")
    return path


async def compute(
    ourstop: list[HierarchyStop],
    stops: dict[str, Any],
    args,
    destinations: dict[str, dict[str, str]],
    loadingbars: bool = True,
    logger=logging.getLogger(name=os.path.basename(SCRIPTDIR)),
):
    logger.info("Computing stops")
    ourstops = [stop.to_dict() for stop in ourstop]
    logger.info("Computing times")
    stopids = [stop["stop_id"] for stop in ourstops]
    ourtimes = await db.get_in_filtered_data("stop_times", column="stop_id", values=stopids)
    logger.info("Computing trips")
    times = await db.get_in_filtered_data("stop_times", column="stop_id", values=stopids, columns=["trip_id"])
    ourtrips = await db.get_in_filtered_data("trips", column="trip_id", values=times)
    logger.info("Computing services")
    services = await db.get_in_filtered_data("trips", column="trip_id", values=times, columns=["service_id"])
    ourservs = await db.get_in_filtered_data("calendar", column="service_id", values=services)
    logger.info("Computing routes")
    routeids = await db.get_in_filtered_data("trips", column="trip_id", values=times, columns=["route_id"])
    ourroute = await db.get_in_filtered_data("routes", column="route_id", values=routeids)

    if ourstops == []:
        print(f'Stop "{ourstop}" not found!')
        logger.error('Stop "%s" not found!', ourstop)
        return

    selected_stop_times = utils.build_list_index(ourtimes, "trip_id")
    stop_times = await utils.build_stop_times_index([trip.trip_id for trip in ourtrips])
    calendar = utils.build_list_index(ourservs, "service_id")
    selected_routes = utils.build_list_index(ourroute, "route_id")

    monset: set[Routedata] = set()
    satset: set[Routedata] = set()
    sunset: set[Routedata] = set()

    if loadingbars:
        trip_iterator = tqdm.tqdm(ourtrips, desc="Sorting trips", unit=" trips", ascii=True, dynamic_ncols=True)
    else:
        trip_iterator = ourtrips
    for trip in trip_iterator:
        data = Routedata(
            dest=trip.trip_headsign,
            time=selected_stop_times[trip.trip_id].departure_time[:-3],
            line=trip.route_id,
            dire=f"d{trip.direction_id}",
            stop=ourstop[0].stop_name,
        )
        if calendar[trip.service_id].monday == "1":
            monset.add(data)
        if calendar[trip.service_id].saturday == "1":
            satset.add(data)
        if calendar[trip.service_id].sunday == "1":
            sunset.add(data)

    mon = sorted(monset, key=lambda x: x.time)
    sat = sorted(satset, key=lambda x: x.time)
    sun = sorted(sunset, key=lambda x: x.time)

    mondict: dict[str, dict[str, dict[str, list[Routedata]]]] = {}
    satdict: dict[str, dict[str, dict[str, list[Routedata]]]] = {}
    sundict: dict[str, dict[str, dict[str, list[Routedata]]]] = {}

    if loadingbars:
        mon_iterator = tqdm.tqdm(mon, desc="Indexing trips", unit=" trips", ascii=True, dynamic_ncols=True)
    else:
        mon_iterator = mon
    for trip in mon_iterator:
        if not mondict.get(trip.line, False):
            mondict[trip.line] = {}
        if not mondict.get(trip.line, {}).get(trip.dire, False):
            mondict[trip.line][trip.dire] = {}
        for i in range(0, 24):
            if not mondict[trip.line][trip.dire].get(f"t{i:02}", False):
                mondict[trip.line][trip.dire][f"t{i:02}"] = []
        mondict[trip.line][trip.dire].setdefault(f"t{trip.time[:2]}", []).append(trip)
    if loadingbars:
        sat_iterator = tqdm.tqdm(sat, desc="Indexing Saturday trips", unit=" trips", ascii=True, dynamic_ncols=True)
    else:
        sat_iterator = sat
    for trip in sat_iterator:
        if not satdict.get(trip.line, False):
            satdict[trip.line] = {}
        if not satdict.get(trip.line, {}).get(trip.dire, False):
            satdict[trip.line][trip.dire] = {}
        for i in range(0, 24):
            if not satdict[trip.line][trip.dire].get(f"t{i:02}", False):
                satdict[trip.line][trip.dire][f"t{i:02}"] = []
        satdict[trip.line][trip.dire].setdefault(f"t{trip.time[:2]}", []).append(trip)
    if loadingbars:
        sun_iterator = tqdm.tqdm(sun, desc="Indexing Sunday trips", unit=" trips", ascii=True, dynamic_ncols=True)
    else:
        sun_iterator = sun
    for trip in sun_iterator:
        if not sundict.get(trip.line, False):
            sundict[trip.line] = {}
        if not sundict.get(trip.line, {}).get(trip.dire, False):
            sundict[trip.line][trip.dire] = {}
        for i in range(0, 24):
            if not sundict[trip.line][trip.dire].get(f"t{i:02}", False):
                sundict[trip.line][trip.dire][f"t{i:02}"] = []
        sundict[trip.line][trip.dire].setdefault(f"t{trip.time[:2]}", []).append(trip)
    pages: dict[str, dict[str, str | None]] = {}

    lines = utils.merge_dicts(utils.merge_dicts(mondict, satdict), sundict)

    if args.logo:
        try:
            logger.info("Getting logo")
            res = requests.get("https://files.sorogon.eu/logo.png")
            tmpfile = tempfile.NamedTemporaryFile(suffix=".png")
            tmpfile.write(res.content)
            tmpfile.flush()
        except requests.exceptions.ConnectionError:
            logger.info("Fallback to string logo")
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
        if loadingbars:
            line_iterator = tqdm.tqdm(lines.items(), desc="Creating pages", unit=" lines", ascii=True, dynamic_ncols=True)
        else:
            line_iterator = lines.items()
        for line, dires in line_iterator:
            if args.color == "random":
                color = utils.generate_contrasting_vibrant_color()
            else:
                color = args.color
            if not pages.get(line, False):
                pages[line] = {}
            for k in dires.keys():
                dest = destinations[line][k]
                if dest != ourstop[0].stop_name:
                    page = pages.get(line, {}).get(k, {})
                    if not isinstance(page, str):
                        page = tempfile.mkstemp(suffix=".pdf", dir=tmpdir)[1]
                    logger.info("Creating page")
                    pages[line][k] = create_page(
                        selected_routes[line].route_short_name,
                        dest,
                        ourstop[0].stop_name,
                        page,
                        mondict.get(line, {}).get(k, {}),
                        satdict.get(line, {}).get(k, {}),
                        sundict.get(line, {}).get(k, {}),
                        color,
                        tmpfile,
                    )
                    if args.map:
                        logger.info("Drawing map")
                        mappage = tempfile.mkstemp(suffix=".pdf", dir=tmpdir)[1]
                        pages[line][k + "map"] = await draw_map(
                            page=mappage,
                            stop_name=ourstop[0].stop_name,
                            logo=tmpfile,
                            routes=await utils.prepare_linedraw_info(stop_times, ourtrips, stops, line, k, [stop["stop_id"] for stop in ourstops]),
                            color=color,
                            label_rotation=0,
                            tmpdir=tmpdir,
                            map_provider=args.map_provider,
                            dpi=args.map_dpi,
                        )

        pagelst: list[str] = []
        if loadingbars:
            line_iterator = tqdm.tqdm(pages.values(), desc="Collecting pages", unit=" lines", ascii=True, dynamic_ncols=True)
        else:
            line_iterator = pages.values()
        for line in line_iterator:
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


async def main():
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter)
    parser.add_argument("-i", "--input", help="Input folder(s)", action="extend", nargs="+", default=[], required=False, dest="input")
    parser.add_argument("-c", "--color", help="Timetable color", type=str, required=False, dest="color", default="random")
    parser.add_argument("-o", "--output", help="Output file", type=str, required=False, dest="output", default="fahrplan.pdf")
    parser.add_argument("-m", "--map", help="Generate maps", action="store_true", dest="map")
    parser.add_argument("-s", "--stop-name-csv", help="Stop name mapping csv folder", required=False, type=str, dest="mapping_csv")
    parser.add_argument("-r", "--reset-db", help="Reset local database", action="store_true", dest="reset_db")
    parser.add_argument("--dpi", help="map dpi", type=int, dest="map_dpi")
    parser.add_argument("--no-logo", action="store_false", dest="logo")
    parser.add_argument(
        "--map-provider",
        type=str,
        choices=MAP_PROVIDERS.keys(),
        default="BasemapAT",
        help="Map provider for the basemap (default: BasemapAT)",
        dest="map_provider",
    )
    args = parser.parse_args()

    rotate_log_file(compress=True)
    setup_logger()
    logger = logging.getLogger(name=os.path.basename(SCRIPTDIR))

    if args.reset_db:
        append = False
    else:
        append = True

    if args.mapping_csv:
        await db.load_hst_csv(args.mapping_csv, append=append)

    if len(args.input) > 0:
        for aid, folder in tqdm.tqdm(enumerate(args.input), total=len(args.input), desc="Loading data", unit="folder", ascii=True, dynamic_ncols=True):
            if os.path.isdir(folder):
                await db.load_gtfs(folder, aid, append=append)
                append = True
            else:
                logger.error("Input folder %s does not exist", folder)

    stops = await db.get_table_data("stops")

    stop_hierarchy = await utils.build_stop_hierarchy()
    stop_hierarchy = await utils.query_stop_names(stop_hierarchy)
    destinations = await utils.build_dest_list()
    stop_id_mapping = {}
    for stop in stop_hierarchy.values():
        if stop.stop_name not in stop_id_mapping:
            stop_id_mapping[stop.stop_name] = [stop.stop_id]
        else:
            stop_id_mapping[stop.stop_name].append(stop.stop_id)
    stops = utils.build_list_index(stops, "stop_id")

    logger.info("Loaded data")

    choices = sorted(stop_id_mapping.keys())
    while True:
        try:
            choice = await asyncio.to_thread(
                lambda: questionary.autocomplete("Haltestelle/Bahnhof: ", choices=choices, match_middle=True, validate=lambda val: val in choices, style=custom_style).ask()
            )
            # choice = questionary.autocomplete("Haltestelle/Bahnhof: ", choices=choices, match_middle=True, validate=lambda val: val in choices, style=custom_style).ask()
        except KeyboardInterrupt:
            print()
            break
        if choice is None:
            print()
            break
        if len(stop_id_mapping[choice]) == 1:
            ourstop = [stop_hierarchy[stop_id_mapping[choice][0]]]
            if ourstop[0].children is not None:
                ourstop.extend(ourstop[0].children)
                del ourstop[0].children
        else:
            ourstop = []
            for stop_id in stop_id_mapping[choice]:
                stop = stop_hierarchy[stop_id]
                if stop.children is not None:
                    ourstop.extend(stop.children)
                    del stop.children
                ourstop.append(stop)

        await compute(ourstop, stops, args, destinations)


if __name__ == "__main__":
    asyncio.run(main())
