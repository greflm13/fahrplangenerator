import contextily as cx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import shape
import numpy as np
import math
import os
import csv
import requests
from io import BytesIO
from PIL import Image


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile, ytile, zoom):
    n = 2.0**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def get_image_cluster(lat_deg, lon_deg, delta_lat, delta_long, zoom):
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}
    smurl = r"http://a.tile.openstreetmap.org/{0}/{1}/{2}.png"
    xmin, ymax = deg2num(lat_deg, lon_deg, zoom)
    xmax, ymin = deg2num(lat_deg + delta_lat, lon_deg + delta_long, zoom)

    cluster = Image.new("RGB", ((xmax - xmin + 1) * 256 - 1, (ymax - ymin + 1) * 256 - 1))
    for xtile in range(xmin, xmax + 1):
        for ytile in range(ymin, ymax + 1):
            try:
                imgurl = smurl.format(zoom, xtile, ytile)
                print("Opening: " + imgurl)
                imgstr = requests.get(imgurl, headers=headers)
                tile = Image.open(BytesIO(imgstr.content))
                cluster.paste(tile, box=((xtile - xmin) * 256, (ytile - ymin) * 255))
            except:
                print("Couldn't download image")
                tile = None

    return cluster


if __name__ == "__main__":
    # a = get_image_cluster(46.81407495, 15.29708758, 0.02, 0.05, 13)
    # fig = plt.figure()
    # fig.patch.set_facecolor("white")
    # plt.imshow(np.asarray(a))
    # plt.show()
    shapes = []
    shapedict = {}
    with open(os.path.join("/home/user/Documents/20251012-2010-at-styria-2025/gkb-at", "shapes.txt"), mode="r", encoding="utf-8-sig") as f:
        shapes.extend([dict(row.items()) for row in csv.DictReader(f, skipinitialspace=True)])
    for shapeline in shapes:
        if shapeline["shape_id"] not in shapedict:
            shapedict[shapeline["shape_id"]] = []
        shapedict[shapeline["shape_id"]].append([shapeline["shape_pt_lon"], shapeline["shape_pt_lat"]])
    geojson = {"type": "LineString", "coordinates": shapedict["37-S61-j25-2.16.R"]}
    gdf = gpd.GeoDataFrame({"geometry": [shape(geojson)]}, crs="EPSG:4326")
    ax = gdf.plot(facecolor="none", edgecolor="red", linewidth=2)
    cx.add_basemap(ax=ax, crs="EPSG:4326", source=cx.providers.OpenStreetMap.DE)
    plt.show()
