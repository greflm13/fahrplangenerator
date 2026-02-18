import os
import io
import math
import asyncio
import logging

from typing import Tuple

import aiohttp
import numpy as np

from PIL import Image
from pyproj import Transformer
from matplotlib.axes import Axes
from xyzservices import TileProvider

# Constants for file paths and exclusions
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__)).removesuffix(__package__ if __package__ else "")

_TILE_MEM_CACHE: dict[tuple, Image.Image] = {}

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


def auto_zoom(bounds_wgs84: Tuple[float, float, float, float], axis_pixel_size: Tuple[int, int], provider: TileProvider, zoom_modifier=0) -> int:
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
            return min(z + zoom_modifier, zmax)

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

    async def load(self, provider: TileProvider, z: int, x: int, y: int) -> Image.Image | None:
        key = (provider.name, z, x, y)
        if key in _TILE_MEM_CACHE:
            return _TILE_MEM_CACHE[key]

        path = self.tile_path(provider, z, x, y)
        if os.path.exists(path):
            loop = asyncio.get_running_loop()
            img = await loop.run_in_executor(None, lambda: Image.open(path).convert("RGBA"))
            _TILE_MEM_CACHE[key] = img
            return img
        return None

    async def save(self, provider: TileProvider, z: int, x: int, y: int, img: Image.Image) -> None:
        path = self.tile_path(provider, z, x, y)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: img.save(path, format="PNG"))


async def _load_or_fetch_tile(cache: TileCache, provider: TileProvider, z: int, x: int, y: int, retries: int = 15, size: int = 256):
    key = (provider.name, z, x, y)

    if key in _TILE_MEM_CACHE:
        return key, _TILE_MEM_CACHE[key], 200

    tile = await cache.load(provider, z, x, y)
    if tile is not None:
        _TILE_MEM_CACHE[key] = tile
        return key, tile, 200

    tile, status = await fetch_tile_with_retry(provider, z, x, y, retries=retries, size=size)
    try:
        await cache.save(provider, z, x, y, tile)
    except Exception:
        pass

    _TILE_MEM_CACHE[key] = tile
    return key, tile, status


async def fetch_tile_with_retry(
    provider: TileProvider, z: int, x: int, y: int, *, timeout: float = 5.0, retries: int = 15, backoff: float = 0.5, size: int = 256
) -> tuple[Image.Image, int]:
    logger.info("Downloading Tile", extra={"provider": provider.get("name"), "tile": {"x": x, "y": y, "zoom": z}})
    for attempt in range(1, retries + 1):
        try:
            return await fetch_tile(provider, z, x, y, timeout=timeout), 200
        except Exception as e:
            if isinstance(e, aiohttp.ClientResponseError):
                if e.code == 404:
                    return Image.new("RGBA", (size, size), (0, 0, 0, 0)), 404
            if attempt < retries:
                logger.info(
                    "Download failed, retrying", extra={"attempt": attempt, "retries": retries, "delay": backoff * (2 ** (attempt - 1)), "tile": {"x": x, "y": y, "zoom": z}}
                )
                await asyncio.sleep(backoff * (2 ** (attempt - 1)))

    return Image.new("RGBA", (size, size), (0, 0, 0, 0)), 500


async def fetch_tile(provider: TileProvider, z: int, x: int, y: int, timeout: float = 5.0) -> Image.Image:
    url = provider.build_url(x=x, y=y, z=z)
    headers = {"User-Agent": USER_AGENT}

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(timeout)) as r:
            r.raise_for_status()
            data = await r.read()

    return Image.open(io.BytesIO(data)).convert("RGBA")


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
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
    )


async def render_basemap(
    ax: Axes,
    extends: Tuple[float, float, float, float],
    zoom: int | None,
    provider: TileProvider,
    cache_dir: str,
    show_attribution: bool = True,
    zoom_modifier=0,
    max_workers=min(8, os.cpu_count() * 2),  # type: ignore
) -> None:
    try:
        xmin, xmax, ymin, ymax = extends

        lon_min, lat_min = MERCATOR_TO_WGS84.transform(xmin, ymin)
        lon_max, lat_max = MERCATOR_TO_WGS84.transform(xmax, ymax)

        if zoom is None:
            fig = ax.figure
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width_px = int(bbox.width * fig.dpi)
            height_px = int(bbox.height * fig.dpi)

            zoom = auto_zoom(bounds_wgs84=(lon_min, lon_max, lat_min, lat_max), axis_pixel_size=(width_px, height_px), provider=provider, zoom_modifier=zoom_modifier)

        cache = TileCache(cache_dir)

        x0, y1 = lonlat_to_tile(lon_min, lat_max, zoom)
        x1, y0 = lonlat_to_tile(lon_max, lat_min, zoom)

        cols = x1 - x0 + 1
        rows = y0 - y1 + 1

        coords_to_try = [(xx, yy) for yy in range(y1, y0 + 1) for xx in range(x0, x1 + 1)]

        sample_tile = None
        status = None

        for xx, yy in coords_to_try:
            key = (provider.name, zoom, xx, yy)

            if key in _TILE_MEM_CACHE:
                sample_tile = _TILE_MEM_CACHE[key]
                status = 200
                break

            tile = await cache.load(provider, zoom, xx, yy)
            if tile is not None:
                sample_tile = tile
                status = 200
                _TILE_MEM_CACHE[key] = tile
                break

            _, tile, status = await _load_or_fetch_tile(cache, provider, zoom, xx, yy, retries=16)
            if status == 200:
                sample_tile = tile
                await cache.save(provider, zoom, xx, yy, tile)
                _TILE_MEM_CACHE[key] = tile
                break

        if sample_tile is None:
            raise RuntimeError("No tile returned HTTP 200")

        tile_px = sample_tile.width

        canvas = np.zeros((rows * tile_px, cols * tile_px, 4), dtype=np.uint8)

        def place(img: Image.Image, cx: int, cy: int, can):
            arr = np.asarray(img)
            y0 = cy * tile_px
            x0 = cx * tile_px
            can[y0 : y0 + tile_px, x0 : x0 + tile_px] = arr

        place(sample_tile, 0, 0, canvas)

        tasks = []
        for xt in range(x0, x1 + 1):
            for yt in range(y1, y0 + 1):
                if xt == x0 and yt == y1:
                    continue
                tasks.append((xt, yt))

        sem = asyncio.Semaphore(max_workers)

        async def guarded_load(xt, yt):
            async with sem:
                return await _load_or_fetch_tile(cache, provider, zoom, xt, yt, size=tile_px)

        results = await asyncio.gather(*(guarded_load(x, y) for x, y in tasks))

        for (_, _, xt, yt), tile, _ in results:
            cx = xt - x0
            cy = yt - y1
            place(tile, cx, cy, canvas)

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

    finally:
        canvas, results, sample_tile, tile, tasks, first_key = None, None, None, None, None, None
        del canvas, results, sample_tile, tile, tasks, first_key
