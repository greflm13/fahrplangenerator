import os
import math
import logging
import tempfile

from typing import List, Dict, Optional

import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

from matplotlib.axes import Axes
from shapely.geometry import Point
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib import colors, pagesizes
from reportlab.lib.utils import ImageReader
from matplotlib.patches import FancyArrowPatch
from xyzservices import TileProvider, providers

# Constants for file paths and exclusions
if __package__ is None:
    PACKAGE = ""
else:
    PACKAGE = __package__
SCRIPTDIR = os.path.abspath(os.path.dirname(__file__).removesuffix(PACKAGE))
MAP_PROVIDERS = {
    "BasemapAT": providers.BasemapAT.highdpi,
    "OPNVKarte": providers.OPNVKarte,
    "OSM": providers.OpenStreetMap.Mapnik,
    "OSMDE": providers.OpenStreetMap.DE,
    "ORM": providers.OpenRailwayMap,
    "OTM": providers.OpenTopoMap,
    "UN": providers.UN.ClearMap,
    "SAT": providers.Esri.WorldImagery,
}


def add_direction_arrows(ax: Axes, shapes: list, arrow_color: Optional[str] = None, min_size: int = 3, max_size: int = 60) -> None:
    logger = logging.getLogger(name=os.path.basename(SCRIPTDIR))
    try:
        for p in list(ax.patches):
            if getattr(p, "_is_direction_arrow", False):
                try:
                    ax.patches.remove(p)  # type: ignore
                except Exception:
                    pass

        try:
            x0, x1, y0, y1 = ax.axis()
        except Exception:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
        p0 = ax.transData.transform((x0, y0))
        px = ax.transData.transform((x1, y0))
        py = ax.transData.transform((x0, y1))
        px_span = abs(px[0] - p0[0])
        py_span = abs(py[1] - p0[1])
        diag = math.hypot(px_span, py_span)
        mutation_scale = int(max(min_size, min(max_size, max(5, diag * 0.015))))

        def _interp(pA, pB, t: float):
            return (pA[0] + (pB[0] - pA[0]) * t, pA[1] + (pB[1] - pA[1]) * t)

        def _add_arrows_from_points(points, color):
            seg_lens = []
            cum = [0.0]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            zs = [p[2] for p in points]
            for i in range(len(points) - 1):
                dx = xs[i + 1] - xs[i]
                dy = ys[i + 1] - ys[i]
                d = math.hypot(dx, dy)
                seg_lens.append(d)
                cum.append(cum[-1] + d)
            total = cum[-1]
            if total <= 0 or len(points) < 2:
                return

            def z_at(seg_idx, t):
                return zs[seg_idx] + t * (zs[seg_idx + 1] - zs[seg_idx])

            xmin, xmax, ymin, ymax = ax.axis()
            map_extends = max(xmax - xmin, ymax - ymin)
            line_extends = max(zs) - min(zs)

            n_arrows = int(line_extends / map_extends * 0.0001)

            for k in range(1, n_arrows + 1):
                target = total * k / (n_arrows + 1)
                seg_idx = 0
                while seg_idx < len(seg_lens) and cum[seg_idx + 1] < target:
                    seg_idx += 1
                seg_start = (xs[seg_idx], ys[seg_idx])
                seg_end = (xs[seg_idx + 1], ys[seg_idx + 1])
                seg_len = seg_lens[seg_idx]
                if seg_len <= 0:
                    continue
                local_target = target - cum[seg_idx]
                t = local_target / seg_len
                delta = min(0.25, 0.5 * (seg_len / total))
                t0 = max(0.0, t - delta / 2.0)
                t1 = min(1.0, t + delta / 2.0)
                pa = _interp(seg_start, seg_end, t0)
                pb = _interp(seg_start, seg_end, t1)
                z_pa = z_at(seg_idx, t0)
                z_pb = z_at(seg_idx, t1)
                if z_pb >= z_pa:
                    tip = pb
                    other = pa
                else:
                    tip = pa
                    other = pb

                try:
                    arrow = FancyArrowPatch(
                        posA=other,
                        posB=tip,
                        arrowstyle="-|>",
                        mutation_scale=mutation_scale,
                        linewidth=0,
                        color=color,
                        zorder=4,
                        transform=ax.transData,
                    )
                    setattr(arrow, "_is_direction_arrow", True)
                    ax.add_patch(arrow)
                except Exception:
                    continue

        for shp in shapes:
            geom = shp.get("geometry") if isinstance(shp, dict) else shp
            try:
                parts = [geom] if hasattr(geom, "coords") else list(geom.geoms)  # type: ignore
            except Exception:
                parts = []
            for part in parts:
                coords = list(part.coords)  # type: ignore
                _add_arrows_from_points(coords, arrow_color or (0, 0, 0))
    except Exception as exc:
        logger.debug("Failed to add direction arrows: %s", exc)


def draw_map(
    page: str,
    stop_name: str,
    logo,
    routes: Dict,
    color: str,
    label_rotation: int,
    label_fontsize: int = 6,
    zoom: int = -1,
    map_provider: str = "BasemapAT",
    dpi: int = 300,
    padding: int = 15,
    tmpdir: str = tempfile.gettempdir(),
) -> str | None:
    """Draw map for selected routes."""
    logger = logging.getLogger(name=os.path.basename(SCRIPTDIR))
    ax = None

    for route in routes["shapes"]:
        gdf = gpd.GeoDataFrame({"geometry": [route["geometry"]]}, crs="EPSG:4326")
        gxmin, gymin, gxmax, gymax = gdf.total_bounds
        gbox_w = gxmax - gxmin
        gbox_h = gymax - gymin
        gbox_aspect = gbox_w / gbox_h
        if gbox_aspect >= 1.0:
            figsize = (10 * math.sqrt(2), 10)
        else:
            figsize = (10, 10 * math.sqrt(2))
        if ax is None:
            ax = gdf.plot(facecolor="none", edgecolor=color, figsize=figsize, linewidth=2)
        else:
            gdf.plot(ax=ax, facecolor="none", edgecolor=color, figsize=figsize, linewidth=2)

    if ax is None:
        logger.error("No routes to plot on map")
        return None

    geometries = []
    for route in routes["shapes"]:
        if "geometry" in route:
            geometries.append(route["geometry"])
    add_direction_arrows(ax, geometries, arrow_color=color)

    try:
        n = plot_stops_on_ax(ax, routes["points"], line_color=color, endstops=routes["endstops"], label_fontsize=label_fontsize, label_rotation=label_rotation)
        logger.info("Plotted %d stops for route", n)
    except Exception as exc:
        logger.warning("Exception while plotting stops: %s", exc)

    ax.set_axis_off()
    xmin, xmax, ymin, ymax = ax.axis()
    bbox_w = xmax - xmin
    bbox_h = ymax - ymin
    bbox_aspect = bbox_w / bbox_h
    projection_aspect = ax.get_aspect()
    if isinstance(projection_aspect, str):
        projection_aspect = 1.0

    if bbox_aspect >= 1.0:
        target_aspect = math.sqrt(2) * projection_aspect
        aspect_diff = bbox_aspect - target_aspect
        logger.debug("Wide bounding box: bbox_aspect=%.3f, target_aspect=%.3f, aspect_diff=%.3f", bbox_aspect, target_aspect, aspect_diff)
        pagesize = pagesizes.landscape(pagesizes.A4)
        movey = 0
    else:
        target_aspect = 1.0 / math.sqrt(2) * projection_aspect
        aspect_diff = bbox_aspect - target_aspect
        logger.debug("Tall bounding box: bbox_aspect=%.3f, target_aspect=%.3f, aspect_diff=%.3f", bbox_aspect, target_aspect, aspect_diff)
        pagesize = pagesizes.portrait(pagesizes.A4)
        movey = 840
    if aspect_diff < 0:
        # Need to increase width
        target_xsize = bbox_h * target_aspect
        increase_x = (target_xsize - bbox_w) / 2.0
        xmin = xmin - increase_x
        xmax = xmax + increase_x
        logger.debug("Increased width by %.3f to %.3f", increase_x * 2.0, target_xsize)
    else:
        # Need to increase height
        target_ysize = bbox_w / target_aspect
        increase_y = (target_ysize - bbox_h) / 2.0
        ymin = ymin - increase_y
        ymax = ymax + increase_y
        logger.debug("Increased height by %.3f to %.3f", increase_y * 2.0, target_ysize)

    xpadding = (xmax - xmin) * (padding / 100.0)
    ypadding = (ymax - ymin) * (padding / 100.0)
    ax.set(xlim=(xmin - xpadding, xmax + xpadding), ylim=(ymin - ypadding, ymax + ypadding))
    logger.debug("Final axis limits: x=(%.6f, %.6f), y=(%.6f, %.6f)", xmin - xpadding, xmax + xpadding, ymin - ypadding, ymax + ypadding)

    zoom_param = zoom if zoom >= 0 else "auto"
    done = False
    while not done:
        try:
            logger.debug("Adding basemap with zoom=%s and provider=%s", zoom_param, map_provider)
            cx.add_basemap(ax=ax, crs="EPSG:4326", source=get_provider_source(map_provider), zoom=str(zoom_param), reset_extent=True)
            done = True
        except Exception as exc:
            logger.warning("Failed to add basemap: %s", exc)
            if isinstance(zoom_param, int) and zoom_param > 0:
                zoom_param -= 1
                logger.info("Retrying with lower zoom level: %s", zoom_param)
            else:
                logger.error("Cannot add basemap, giving up.")
                done = True

    try:
        pdf = Canvas(page, pagesize=pagesize)
        pdf.scale(pagesize[0] / 1188, pagesize[1] / (840 + movey))
        _, b = tempfile.mkstemp(suffix=".png", dir=tmpdir)
        plt.savefig(b, dpi=dpi, bbox_inches="tight", pad_inches=0, pil_kwargs={"compress_level": 0})
        plt.close()
        image = ImageReader(b)
        pdf.drawImage(image, x=50, y=60, width=1088, height=730 + movey, preserveAspectRatio=True)

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
        os.remove(b)
        logger.info("Saved map to %s", page)
    except Exception as exc:
        logger.error("Failed to save map to %s: %s", page, exc)
    return page


def plot_stops_on_ax(
    ax: Axes,
    stops: List,
    line_color: str,
    endstops: List[str],
    marker_size: int = 30,
    label_fontsize: int = 7,
    label_rotation: int = 0,
) -> int:
    """Plot stops for a given shape_id onto the provided axes and annotate stop names.

    The stop marker color is derived from the line color but made visually distinct by
    darkening it.

    Returns the number of plotted stops.
    """
    logger = logging.getLogger(name=os.path.basename(SCRIPTDIR))
    try:
        rgb = mcolors.to_rgb(line_color)
        stop_rgb = tuple(max(0.0, c * 0.5) for c in rgb)
        stop_color = stop_rgb
    except Exception:
        stop_color = (1.0, 0.0, 0.0)

    stop_rows = [row[1]._asdict() for row in stops]
    stops_points = [row[0] for row in stops]

    gdf_stops = gpd.GeoDataFrame(stop_rows, geometry=stops_points, crs="EPSG:4326")
    try:
        gdf_stops.plot(ax=ax, color=stop_color, markersize=marker_size, zorder=5)
    except Exception as exc:
        logger.warning("Failed to plot stops GeoDataFrame: %s", exc)
        return 0

    already_drawn = set()

    for row in gdf_stops.itertuples(index=False):
        pt = row.geometry
        name = row.name

        if name not in already_drawn:
            already_drawn.add(name)
        else:
            continue

        if name in endstops:
            fontweight = "bold"
        else:
            fontweight = "normal"

        if not isinstance(name, str) or not isinstance(pt, Point):
            continue
        try:
            txt = ax.annotate(
                name,
                xy=(pt.x, pt.y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=label_fontsize,
                color="black",
                weight=fontweight,
                ha="left",
                va="center",
                zorder=6,
                rotation=label_rotation,
                rotation_mode="anchor",
                bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "black", "linewidth": 0.2, "alpha": 2 / 3},
                clip_on=False,
            )
            try:
                txt.set_path_effects([pe.withStroke(linewidth=0.5, foreground="white")])
            except Exception:
                pass
        except Exception:
            continue

    return len(gdf_stops)


def get_provider_source(provider_name: str) -> TileProvider:
    """Get the contextily provider source based on the provider name."""
    return MAP_PROVIDERS.get(provider_name, providers.BasemapAT.highdpi)
