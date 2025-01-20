import os
import csv
import argparse
import tempfile

import survey
from rich_argparse import RichHelpFormatter
from pyjarowinkler import distance
from pypdf import PdfReader, PdfWriter

from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib import colors, pagesizes


pdfmetrics.registerFont(TTFont("header", "Montserrat-Black.ttf"))
pdfmetrics.registerFont(TTFont("hour", "Montserrat-Bold.ttf"))


def create_merged_pdf(pages: list, path: str):
    output = PdfWriter()

    for page in pages:
        pdf = PdfReader(page)
        output.add_page(pdf.pages[0])

    output.write(path)


def dict_set(lst: list):
    seen = []
    setlike = []
    for d in lst:
        signature = {"time": d["time"], "line": d["line"], "dire": d["dire"]}
        if signature not in seen:
            seen.append(signature)
            setlike.append(d)
    return setlike


def most_frequent(l: list):
    return max(set(l), key=l.count)


def detect_shared_prefix(lst: list) -> tuple[bool, str]:
    if not lst:
        return False, ""

    prefix = lst[0]
    for item in lst[1:]:
        if not isinstance(item, str):
            item = item["name"]
        while not item.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return False, ""

    return True, prefix


def merge_similar(inputList):
    refinedInputList = []
    for v in inputList:
        if len(refinedInputList) > 0:
            last = refinedInputList.pop()
            if distance.get_jaro_distance(last, v, winkler=True, scaling=0.1) > 0.9:
                pre, prefix = detect_shared_prefix([last, v])
                if pre:
                    refinedInputList.append(prefix)
                else:
                    refinedInputList.append(last)
                    refinedInputList.append(v)
            else:
                refinedInputList.append(last)
                refinedInputList.append(v)
        else:
            refinedInputList.append(v)
    return refinedInputList


def addtimes(pdf: canvas.Canvas, daytimes: dict, day: str, posy: float, accent: str):
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
        pdf.setFillColor(colors.white)
        pdf.drawCentredString(x=posx, y=posy + 8.5, text=k[1:])
        space = 0
        times = max(times, len(v))
        for time in v:
            # Minutes Text
            pdf.setFillColor(colors.black)
            pdf.drawCentredString(x=posx, y=posy - 20 + space, text=time["time"][3:])
            space -= 25
    posx = 250

    # Lines
    pdf.line(x1=80, y1=posy + 30, x2=80, y2=posy - times * 25 - 3.5)
    pdf.line(x1=250, y1=posy + 30, x2=250, y2=posy - times * 25 - 3.5)
    pdf.line(x1=80, y1=posy - times * 25 - 3.5, x2=1108, y2=posy - times * 25 - 3.5)
    for _ in daytimes.keys():
        posx += spacing
        pdf.line(x1=posx, y1=posy + 30, x2=posx, y2=posy - times * 25 - 3.5)

    posy -= 60 + times * 25
    return pdf, posy


def create_page(line: str, dest: str, imgpath: str, montimes: dict, sattimes: dict, suntimes: dict, color: str):
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
    pdf = canvas.Canvas(imgpath, pagesize=pagesize)
    pdf.scale(pagesize[0] / 1188, pagesize[1] / (840 + movey))
    accent = colors.HexColor(color)

    # Header
    pdf.setFont("header", 48)
    pdf.setFillColor(accent)
    pdf.drawCentredString(x=1188 / 2, y=760 + movey, text=f"{line} - {dest}")

    posy = 690 + movey

    if len(montimes) > 0:
        pdf, posy = addtimes(pdf, montimes, "Montag-Freitag", posy, accent)

    if len(sattimes) > 0:
        pdf, posy = addtimes(pdf, sattimes, "Samstag", posy, accent)

    if len(suntimes) > 0:
        pdf, posy = addtimes(pdf, suntimes, "Sonntag", posy, accent)

    pdf.save()
    return imgpath


def main():
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter)
    parser.add_argument("-i", "--input", help="Input folder(s)", action="extend", nargs="+", required=True, dest="input")
    parser.add_argument("-c", "--color", help="Timetable color", type=str, required=False, dest="color", default="#3f3f3f")
    parser.add_argument("-s", "--stop", help="Stop to generate timetable for", type=str, required=False, dest="stop", default="")
    args = parser.parse_args()

    stops, stop_times, trips, calendar, routes = [], [], [], [], []

    for folder in args.input:
        with open(os.path.join(folder, "stops.txt"), mode="r", encoding="utf-8-sig") as f:
            stops.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])

        with open(os.path.join(folder, "stop_times.txt"), mode="r", encoding="utf-8-sig") as f:
            stop_times.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])

        with open(os.path.join(folder, "trips.txt"), mode="r", encoding="utf-8-sig") as f:
            trips.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])

        with open(os.path.join(folder, "calendar.txt"), mode="r", encoding="utf-8-sig") as f:
            calendar.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])

        with open(os.path.join(folder, "routes.txt"), mode="r", encoding="utf-8-sig") as f:
            routes.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])

    if args.stop == "":
        choices = merge_similar(sorted({stop["stop_name"] for stop in stops}))
        choice = survey.routines.select("Haltestelle/Bahnhof: ", options=choices)
        ourstop = choices[choice]
    else:
        ourstop = args.stop

    ourstops = [stop for stop in stops if stop["stop_name"].startswith(ourstop)]
    ourtimes = [time for time in stop_times if time["stop_id"] in [stop["stop_id"] for stop in ourstops]]
    ourtrips = [trip for trip in trips if trip["trip_id"] in [time["trip_id"] for time in ourtimes]]
    ourservs = [serv for serv in calendar if serv["service_id"] in [trip["service_id"] for trip in ourtrips]]
    ourroute = [rout for rout in routes if rout["route_id"] in [trip["route_id"] for trip in ourtrips]]

    if ourstops == []:
        print(f'Stop "{ourstop}" not found!')
        return

    stops = {}
    stop_times = {}
    trips = {}
    calendar = {}
    routes = {}

    for stop in ourstops:
        stops[stop["stop_id"]] = stop

    for time in ourtimes:
        stop_times[time["trip_id"]] = time

    for trip in ourtrips:
        trips[trip["trip_id"]] = trip

    for serv in ourservs:
        calendar[serv["service_id"]] = serv

    for rout in ourroute:
        routes[rout["route_id"]] = rout

    mon = []
    sat = []
    sun = []

    for trip in ourtrips:
        if calendar[trip["service_id"]]["monday"] == "1":
            mon.append(
                {
                    "dest": trip["trip_headsign"],
                    "time": stop_times[trip["trip_id"]]["arrival_time"][:-3],
                    "line": routes[trip["route_id"]]["route_short_name"],
                    "dire": trips[trip["trip_id"]]["direction_id"],
                }
            )
        if calendar[trip["service_id"]]["saturday"] == "1":
            sat.append(
                {
                    "dest": trip["trip_headsign"],
                    "time": stop_times[trip["trip_id"]]["arrival_time"][:-3],
                    "line": routes[trip["route_id"]]["route_short_name"],
                    "dire": trips[trip["trip_id"]]["direction_id"],
                }
            )
        if calendar[trip["service_id"]]["sunday"] == "1":
            sun.append(
                {
                    "dest": trip["trip_headsign"],
                    "time": stop_times[trip["trip_id"]]["arrival_time"][:-3],
                    "line": routes[trip["route_id"]]["route_short_name"],
                    "dire": trips[trip["trip_id"]]["direction_id"],
                }
            )

    mon = sorted(dict_set(mon), key=lambda x: x["time"])
    sat = sorted(dict_set(sat), key=lambda x: x["time"])
    sun = sorted(dict_set(sun), key=lambda x: x["time"])

    mondict = {}
    satdict = {}
    sundict = {}

    for trip in mon:
        if not mondict.get(trip["line"], False):
            mondict[trip["line"]] = {}
        if not mondict.get(trip["line"]).get(f"d{trip['dire']}", False):
            mondict[trip["line"]][f"d{trip['dire']}"] = {}
        for i in range(0, 24):
            if not mondict[trip["line"]][f"d{trip['dire']}"].get(f"t{i:02}", False):
                mondict[trip["line"]][f"d{trip['dire']}"][f"t{i:02}"] = []
        mondict[trip["line"]][f"d{trip['dire']}"].setdefault(f"t{trip['time'][:2]}", []).append(trip)

    for trip in sat:
        if not satdict.get(trip["line"], False):
            satdict[trip["line"]] = {}
        if not satdict.get(trip["line"]).get(f"d{trip['dire']}", False):
            satdict[trip["line"]][f"d{trip['dire']}"] = {}
        for i in range(0, 24):
            if not satdict[trip["line"]][f"d{trip['dire']}"].get(f"t{i:02}", False):
                satdict[trip["line"]][f"d{trip['dire']}"][f"t{i:02}"] = []
        satdict[trip["line"]][f"d{trip['dire']}"].setdefault(f"t{trip['time'][:2]}", []).append(trip)

    for trip in sun:
        if not sundict.get(trip["line"], False):
            sundict[trip["line"]] = {}
        if not sundict.get(trip["line"]).get(f"d{trip['dire']}", False):
            sundict[trip["line"]][f"d{trip['dire']}"] = {}
        for i in range(0, 24):
            if not sundict[trip["line"]][f"d{trip['dire']}"].get(f"t{i:02}", False):
                sundict[trip["line"]][f"d{trip['dire']}"][f"t{i:02}"] = []
        sundict[trip["line"]][f"d{trip['dire']}"].setdefault(f"t{trip['time'][:2]}", []).append(trip)

    sheets = {}

    lines = mondict | satdict | sundict

    for line, dires in lines.items():
        if not sheets.get(line, False):
            sheets[line] = {}
        for k, dire in dires.items():
            destlist = []
            for time in dire.values():
                destlist.extend([t["dest"] for t in time])
            img = sheets.get(line, {}).get(k, {})
            if img == {}:
                img = tempfile.mkstemp(suffix=".pdf")[1]
            sheets[line][k] = create_page(
                line,
                most_frequent(destlist),
                img,
                mondict.get(line, {}).get(k, {}),
                satdict.get(line, {}).get(k, {}),
                sundict.get(line, {}).get(k, {}),
                args.color,
            )

    sheetlst = []
    for line in sheets.values():
        for dire in line.values():
            sheetlst.append(dire)

    create_merged_pdf(sheetlst, "fahrplan.pdf")

    for line in sheets.values():
        for dire in line.values():
            os.remove(dire)


if __name__ == "__main__":
    main()
