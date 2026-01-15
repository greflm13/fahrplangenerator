import os
import io
import math
import time
import logging
import requests

from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from pyproj import Transformer
from matplotlib.axes import Axes
from xyzservices import TileProvider

# Constants for file paths and exclusions
if __package__ is None:
    PACKAGE = ""
else:
    PACKAGE = __package__
SCRIPTDIR = os.path.abspath(os.path.dirname(__file__).removesuffix(PACKAGE))

_TILE_MEM_CACHE: dict[tuple, Image.Image] = {}
_HTTP_SESSION = requests.Session()

TILE_SIZE = 256
USER_AGENT = "xyzservices-tile-renderer/1.1"
WEB_MERCATOR_HALF = 20037508.342789244
WEB_MERCATOR_EXTENT = WEB_MERCATOR_HALF * 2
DEFAULT_ZOOM_LIMITS = (0, 19)

WGS84_TO_MERCATOR = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
MERCATOR_TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

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


def get_tile_scale(provider: TileProvider) -> int:
    for key in ("tileScale", "scale", "scales"):
        v = provider.get(key)
        if isinstance(v, int) and v > 0:
            return v
    return 1


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
        key = (provider.name, z, x, y)
        if key in _TILE_MEM_CACHE:
            return _TILE_MEM_CACHE[key]

        path = self.tile_path(provider, z, x, y)
        if os.path.exists(path):
            img = Image.open(path).convert("RGBA")
            _TILE_MEM_CACHE[key] = img
            return img
        return None

    def save(self, provider: TileProvider, z: int, x: int, y: int, img: Image.Image) -> None:
        path = self.tile_path(provider, z, x, y)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path, format="PNG")


def _load_or_fetch_tile(cache: TileCache, provider: TileProvider, z: int, x: int, y: int):
    key = (provider.name, z, x, y)

    if key in _TILE_MEM_CACHE:
        return key, _TILE_MEM_CACHE[key]

    tile = cache.load(provider, z, x, y)
    if tile is not None:
        _TILE_MEM_CACHE[key] = tile
        return key, tile

    tile = fetch_tile_with_retry(
        provider,
        z,
        x,
        y,
        timeout=5.0,
        retries=3,
        backoff=0.5,
    )

    try:
        cache.save(provider, z, x, y, tile)
    except Exception:
        pass

    _TILE_MEM_CACHE[key] = tile
    return key, tile


def fetch_tile_with_retry(provider: TileProvider, z: int, x: int, y: int, *, timeout: float = 5.0, retries: int = 3, backoff: float = 0.5) -> Image.Image:
    """
    Fetch a tile with timeout + retry.
    Returns a transparent tile if all retries fail.
    """

    last_err = None

    for attempt in range(1, retries + 1):
        try:
            return fetch_tile(provider, z, x, y, timeout=timeout)

        except Exception as e:
            last_err = e
            if attempt < retries:
                sleep = backoff * (2 ** (attempt - 1))
                time.sleep(sleep)

    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    return img


def fetch_tile(provider: TileProvider, z: int, x: int, y: int, timeout: float = 5.0) -> Image.Image:
    url = provider.build_url(x=x, y=y, z=z)
    logger.info("Fetching Tile", extra={"provider": provider.get("name"), "tile": {"x": x, "y": y, "z": z}})
    headers = {"User-Agent": USER_AGENT}
    r = _HTTP_SESSION.get(url, headers=headers, timeout=timeout)
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
    ax: Axes,
    extends: Tuple[float, float, float, float],
    zoom: int | None,
    provider: TileProvider,
    cache_dir: str,
    show_attribution: bool = True,
    max_workers=min(8, os.cpu_count() * 2),  # type: ignore
) -> None:
    xmin, xmax, ymin, ymax = extends

    lon_min, lat_min = MERCATOR_TO_WGS84.transform(xmin, ymin)
    lon_max, lat_max = MERCATOR_TO_WGS84.transform(xmax, ymax)

    if zoom is None:
        fig = ax.figure
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width_px = int(bbox.width * fig.dpi)
        height_px = int(bbox.height * fig.dpi)

        zoom = auto_zoom(
            bounds_wgs84=(lon_min, lon_max, lat_min, lat_max),
            axis_pixel_size=(width_px, height_px),
            provider=provider,
        )

    cache = TileCache(cache_dir)

    x0, y1 = lonlat_to_tile(lon_min, lat_max, zoom)
    x1, y0 = lonlat_to_tile(lon_max, lat_min, zoom)

    cols = x1 - x0 + 1
    rows = y0 - y1 + 1

    first_key = (provider.name, zoom, x0, y1)
    if first_key in _TILE_MEM_CACHE:
        sample_tile = _TILE_MEM_CACHE[first_key]
    else:
        sample_tile = cache.load(provider, zoom, x0, y1)
        if sample_tile is None:
            sample_tile = fetch_tile(provider, zoom, x0, y1)
            cache.save(provider, zoom, x0, y1, sample_tile)
        _TILE_MEM_CACHE[first_key] = sample_tile

    tile_px = sample_tile.width

    canvas = np.zeros((rows * tile_px, cols * tile_px, 4), dtype=np.uint8)

    def place(img: Image.Image, cx: int, cy: int):
        arr = np.asarray(img)
        y0 = cy * tile_px
        x0 = cx * tile_px
        canvas[y0 : y0 + tile_px, x0 : x0 + tile_px] = arr

    place(sample_tile, 0, 0)

    tasks = []
    for xt in range(x0, x1 + 1):
        for yt in range(y1, y0 + 1):
            if xt == x0 and yt == y1:
                continue
            tasks.append((xt, yt))

    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_load_or_fetch_tile, cache, provider, zoom, xt, yt): (xt, yt) for xt, yt in tasks}

        for fut in as_completed(futures):
            key, tile = fut.result()
            results[key] = tile

    for (prov, z, xt, yt), tile in results.items():
        cx = xt - x0
        cy = yt - y1
        place(tile, cx, cy)

    grid_px = 256
    res = WEB_MERCATOR_EXTENT / (grid_px * 2**zoom)

    left = -WEB_MERCATOR_HALF + x0 * grid_px * res
    right = -WEB_MERCATOR_HALF + (x1 + 1) * grid_px * res
    top = WEB_MERCATOR_HALF - y1 * grid_px * res
    bottom = WEB_MERCATOR_HALF - (y0 + 1) * grid_px * res

    ax.imshow(
        canvas,
        extent=(left, right, bottom, top),
        origin="upper",
        zorder=0,
    )

    if show_attribution:
        draw_attribution(ax, provider)
