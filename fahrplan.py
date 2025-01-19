import os
import csv
import argparse
import tempfile

import survey
from PIL import Image, ImageDraw, ImageFont
from rich_argparse import RichHelpFormatter
from pyjarowinkler import distance

Image.MAX_IMAGE_PIXELS = 933120000

header = ImageFont.truetype("Montserrat-Black.otf", 480)
hour = ImageFont.truetype("Montserrat-Bold.otf", 170)


def slg(l: list, i: int):
    try:
        return f"{l[i]['time'][3:]}"
    except IndexError:
        return "  "


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


def create_page(line: str, dest: str, imgpath: str, montimes: dict, sattimes: dict, suntimes: dict, color: str):
    img = Image.open(imgpath)
    draw = ImageDraw.Draw(img)

    box = draw.textbbox((0, 0), f"{line} - {dest}", font=header)
    draw.text((11880 / 2 - box[2] / 2, 397), f"{line} - {dest}", font=header, fill=color)

    starty = 1270

    if len(montimes) > 0:
        draw.rectangle([(797, starty), (797 + 10136, starty + 296)], fill=color, outline=(0, 0, 0), width=12)
        draw.text((945, starty + 40), f"Montag - Freitag", font=hour, fill=(255, 255, 255))
        spacing = 8350 / len(montimes.keys())
        posx = 2300
        times = 0
        for k, v in montimes.items():
            posx += spacing
            draw.text((posx, starty + 40), k[1:], font=hour, align="center", fill=(255, 255, 255))
            space = 0
            times = max(times, len(v))
            for time in v:
                draw.text((posx, starty + 330 + space), time["time"][3:], font=hour, align="center", fill=(0, 0, 0))
                space += 250
        posx = 2300
        draw.rectangle([(2560, starty), (2560 + 12, starty + 346 + times * 250)], fill=(0, 0, 0), width=0)
        draw.rectangle([(797, starty), (797 + 12, starty + 346 + times * 250)], fill=(0, 0, 0), width=0)
        draw.rectangle([(797, starty + 334 + times * 250), (797 + 10136, starty + 346 + times * 250)], fill=(0, 0, 0), width=0)
        for k in montimes.keys():
            posx += spacing
            draw.rectangle([(posx + 272, starty), (posx + 284, starty + 346 + times * 250)], fill=(0, 0, 0), width=0)

        starty += 600 + times * 250

    if len(sattimes) > 0:
        draw.rectangle([(797, starty), (797 + 10136, starty + 296)], fill=color, outline=(0, 0, 0), width=12)
        draw.text((1245, starty + 40), f"Samstag", font=hour, fill=(255, 255, 255))
        spacing = 8350 / len(sattimes.keys())
        posx = 2300
        times = 0
        for k, v in sattimes.items():
            posx += spacing
            draw.text((posx, starty + 40), k[1:], font=hour, align="center", fill=(255, 255, 255))
            space = 0
            times = max(times, len(v))
            for time in v:
                draw.text((posx, starty + 330 + space), time["time"][3:], font=hour, align="center", fill=(0, 0, 0))
                space += 250
        posx = 2300
        draw.rectangle([(2560, starty), (2560 + 12, starty + 346 + times * 250)], fill=(0, 0, 0), width=0)
        draw.rectangle([(797, starty), (797 + 12, starty + 346 + times * 250)], fill=(0, 0, 0), width=0)
        draw.rectangle([(797, starty + 334 + times * 250), (797 + 10136, starty + 346 + times * 250)], fill=(0, 0, 0), width=0)
        for k in sattimes.keys():
            posx += spacing
            draw.rectangle([(posx + 272, starty), (posx + 284, starty + 346 + times * 250)], fill=(0, 0, 0), width=0)

        starty += 600 + times * 250

    if len(suntimes) > 0:
        draw.rectangle([(797, starty), (797 + 10136, starty + 296)], fill=color, outline=(0, 0, 0), width=12)
        draw.text((1245, starty + 40), f"Sonntag", font=hour, fill=(255, 255, 255))
        spacing = 8350 / len(suntimes.keys())
        posx = 2300
        times = 0
        for k, v in suntimes.items():
            posx += spacing
            draw.text((posx, starty + 40), k[1:], font=hour, align="center", fill=(255, 255, 255))
            space = 0
            times = max(times, len(v))
            for time in v:
                draw.text((posx, starty + 330 + space), time["time"][3:], font=hour, align="center", fill=(0, 0, 0))
                space += 250
        posx = 2300
        draw.rectangle([(2560, starty), (2560 + 12, starty + 346 + times * 250)], fill=(0, 0, 0), width=0)
        draw.rectangle([(797, starty), (797 + 12, starty + 346 + times * 250)], fill=(0, 0, 0), width=0)
        draw.rectangle([(797, starty + 334 + times * 250), (797 + 10136, starty + 346 + times * 250)], fill=(0, 0, 0), width=0)
        for k in suntimes.keys():
            posx += spacing
            draw.rectangle([(posx + 272, starty), (posx + 284, starty + 346 + times * 250)], fill=(0, 0, 0), width=0)

    img.save(imgpath)
    img.close()
    return imgpath


def main():
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter)
    parser.add_argument("-i", "--input", help="Input folder", type=str, required=True, dest="input")
    parser.add_argument("-c", "--color", help="Timetable color", type=str, required=False, dest="color", default="#3f3f3f")
    parser.add_argument("-s", "--stop", help="Stop to generate timetable for", type=str, required=False, dest="stop", default="")
    args = parser.parse_args()

    with open(os.path.join(args.input, "stops.txt"), mode="r", encoding="utf-8-sig") as f:
        stops = [dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)]

    with open(os.path.join(args.input, "stop_times.txt"), mode="r", encoding="utf-8-sig") as f:
        stop_times = [dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)]

    with open(os.path.join(args.input, "trips.txt"), mode="r", encoding="utf-8-sig") as f:
        trips = [dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)]

    with open(os.path.join(args.input, "calendar.txt"), mode="r", encoding="utf-8-sig") as f:
        calendar = [dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)]

    with open(os.path.join(args.input, "routes.txt"), mode="r", encoding="utf-8-sig") as f:
        routes = [dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)]

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
                img = tempfile.mkstemp(suffix=".png")[1]
                file = Image.new("RGB", (11880, 8400), (255, 255, 255))
                file.save(img)
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
            sheetlst.append(Image.open(dire))

    first = sheetlst.pop(0)
    pagesizepxl = first.size
    pagesizein = (297 * 0.03937008, 210 * 0.03937008)
    dpi = (pagesizepxl[0] / pagesizein[0], pagesizepxl[1] / pagesizein[1])
    first.save(os.path.join(args.input, "fahrplan.pdf"), "PDF", save_all=True, append_images=sheetlst, dpi=dpi)

    for line in sheets.values():
        for dire in line.values():
            os.remove(dire)


if __name__ == "__main__":
    main()
