import os
import io
import math
import logging
import requests
from typing import Tuple

import numpy as np
from PIL import Image
from pyproj import Transformer
from xyzservices import TileProvider

# Constants for file paths and exclusions
if __package__ is None:
    PACKAGE = ""
else:
    PACKAGE = __package__
SCRIPTDIR = os.path.abspath(os.path.dirname(__file__).removesuffix(PACKAGE))

TILE_SIZE = 256
USER_AGENT = "xyzservices-tile-renderer/1.1"
WEB_MERCATOR_HALF = 20037508.342789244
WEB_MERCATOR_EXTENT = WEB_MERCATOR_HALF * 2
DEFAULT_ZOOM_LIMITS = (0, 19)


WGS84_TO_MERCATOR = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

logger = logging.getLogger(name=os.path.basename(SCRIPTDIR))


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    lat_rad = math.radians(lat)
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_lonlat(x: int, y: int, zoom: int) -> Tuple[float, float]:
    n = 2**zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def auto_zoom(
    bounds_wgs84: Tuple[float, float, float, float],
    axis_pixel_size: Tuple[int, int],
    provider: TileProvider,
) -> int:
    xmin, xmax, ymin, ymax = bounds_wgs84
    width_px, height_px = axis_pixel_size

    x0, y0 = WGS84_TO_MERCATOR.transform(xmin, ymin)
    x1, y1 = WGS84_TO_MERCATOR.transform(xmax, ymax)

    width_m = abs(x1 - x0)
    height_m = abs(y1 - y0)

    required_res = max(width_m / width_px, height_m / height_px)

    zmin = provider.get("min_zoom", DEFAULT_ZOOM_LIMITS[0])
    zmax = provider.get("max_zoom", DEFAULT_ZOOM_LIMITS[1])

    for z in range(zmin, zmax + 1):
        tile_res = WEB_MERCATOR_EXTENT / (TILE_SIZE * 2**z)
        if tile_res <= required_res:
            return z

    return zmin


class TileCache:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def tile_path(self, provider: TileProvider, z: int, x: int, y: int) -> str:
        return os.path.join(
            self.root,
            provider.name.replace(".", "_"),
            str(z),
            str(x),
            f"{y}.png",
        )

    def load(self, provider: TileProvider, z: int, x: int, y: int) -> Image.Image | None:
        path = self.tile_path(provider, z, x, y)
        if os.path.exists(path):
            return Image.open(path).convert("RGBA")
        return None

    def save(self, provider: TileProvider, z: int, x: int, y: int, img: Image.Image) -> None:
        path = self.tile_path(provider, z, x, y)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path, format="PNG")


def fetch_tile(provider: TileProvider, z: int, x: int, y: int) -> Image.Image:
    url = provider.build_url(x=x, y=y, z=z)
    logger.info("Fetching Tile", extra={"provider": provider.get("name"), "tile": {"x": x, "y": y, "z": z}})
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")


def draw_attribution(ax, provider: TileProvider) -> None:
    text = provider.get("attribution", "")
    if not text:
        return

    ax.text(
        0.995,
        0.005,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color="black",
        zorder=10,
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="none",
            alpha=0.7,
        ),
    )


def render_basemap(
    ax,
    zoom: int | None,
    provider: TileProvider,
    cache_dir: str,
    show_attribution: bool = True,
) -> None:
    xmin, xmax = sorted(ax.get_xlim())
    ymin, ymax = sorted(ax.get_ylim())
    if zoom is None:
        fig = ax.figure
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width_px = int(bbox.width * fig.dpi)
        height_px = int(bbox.height * fig.dpi)

        zoom = auto_zoom(
            bounds_wgs84=(xmin, xmax, ymin, ymax),
            axis_pixel_size=(width_px, height_px),
            provider=provider,
        )

    cache = TileCache(cache_dir)

    x0, y1 = lonlat_to_tile(xmin, ymax, zoom)
    x1, y0 = lonlat_to_tile(xmax, ymin, zoom)

    width = (x1 - x0 + 1) * TILE_SIZE
    height = (y0 - y1 + 1) * TILE_SIZE

    canvas = Image.new("RGBA", (width, height))

    for xt in range(x0, x1 + 1):
        for yt in range(y0, y1 - 1, -1):
            tile = cache.load(provider, zoom, xt, yt)
            if tile is None:
                tile = fetch_tile(provider, zoom, xt, yt)
                cache.save(provider, zoom, xt, yt, tile)
            tile_size = tile.height

            px = (xt - x0) * tile_size
            py = (yt - y1) * tile_size
            canvas.paste(tile, (px, py))

    lon_min, lat_max = tile_to_lonlat(x0, y0, zoom)
    lon_max, lat_min = tile_to_lonlat(x1 + 1, y1 + 1, zoom)

    ax.imshow(
        np.asarray(canvas),
        extent=(lon_min, lon_max, lat_min, lat_max),
        origin="upper",
        zorder=0,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if show_attribution:
        draw_attribution(ax, provider)
