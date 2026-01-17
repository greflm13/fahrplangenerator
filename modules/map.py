import os
import math
import logging
import tempfile

from typing import List, Dict, Optional

import matplotlib
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

from matplotlib.axes import Axes
from shapely.geometry import Point
from adjustText import adjust_text
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib import colors, pagesizes
from reportlab.lib.utils import ImageReader
from matplotlib.patches import FancyArrowPatch
from xyzservices import TileProvider, providers

from modules.xyz_basemap import render_basemap

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
    "SAT": providers.Esri.WorldImagery,
}

matplotlib.use("Agg")


def points_to_data_offset(ax, dx_pt, dy_pt):
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0, 0))
    x1, y1 = inv.transform((dx_pt, dy_pt))
    return x1 - x0, y1 - y0


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
            if len(points) < 2:
                return
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            seg_lens = []
            cum = [0.0]
            for i in range(len(points) - 1):
                dx = xs[i + 1] - xs[i]
                dy = ys[i + 1] - ys[i]
                d = math.hypot(dx, dy)
                seg_lens.append(d)
                cum.append(cum[-1] + d)
            total = cum[-1]
            if total <= 0:
                return

            xmin, xmax, ymin, ymax = ax.axis()
            p0 = ax.transData.transform((xmin, ymin))
            p1 = ax.transData.transform((xmax, ymax))
            map_diag_px = math.hypot(p1[0] - p0[0], p1[1] - p0[1])

            n_arrows = max(3, int(map_diag_px / 150))

            for k in range(1, n_arrows + 1):
                target = total * k / (n_arrows + 1)
                seg_idx = 0
                while seg_idx < len(seg_lens) and cum[seg_idx + 1] < target:
                    seg_idx += 1
                if seg_idx >= len(seg_lens):
                    continue
                seg_start = (xs[seg_idx], ys[seg_idx])
                seg_end = (xs[seg_idx + 1], ys[seg_idx + 1])
                seg_len = seg_lens[seg_idx]
                if seg_len <= 0:
                    continue
                pa0 = (xs[seg_idx], ys[seg_idx])
                pb0 = (xs[seg_idx + 1], ys[seg_idx + 1])
                spa = ax.transData.transform(pa0)
                spb = ax.transData.transform(pb0)
                seg_len_px = math.hypot(spb[0] - spa[0], spb[1] - spa[1])
                if seg_len_px < 5:
                    continue
                local_target = target - cum[seg_idx]
                t = local_target / seg_len
                delta = min(0.25, 0.5 * (seg_len / total))
                t0 = max(0.0, t - delta / 2.0)
                t1 = min(1.0, t + delta / 2.0)
                pa = _interp(seg_start, seg_end, t0)
                pb = _interp(seg_start, seg_end, t1)
                tip, other = pb, pa

                try:
                    arrow = FancyArrowPatch(
                        posA=other,
                        posB=tip,
                        arrowstyle="-|>",
                        mutation_scale=mutation_scale,
                        linewidth=0,
                        color=color,
                        zorder=5,
                        transform=ax.transData,
                    )
                    setattr(arrow, "_is_direction_arrow", True)
                    ax.add_patch(arrow)
                except Exception as exc:
                    logger.debug("Arrow failed: %s", exc)

        for shp in shapes:
            geom = shp.get("geometry") if isinstance(shp, dict) else shp
            try:
                parts = [geom] if hasattr(geom, "coords") else list(geom.geoms)  # type: ignore
            except Exception:
                parts = []
            for part in parts:
                try:
                    coords = list(part.coords)  # type: ignore
                    _add_arrows_from_points(coords, arrow_color or (0, 0, 0))
                except Exception as exc:
                    logger.debug("Arrow failed on geometry: %s", exc)

    except Exception as exc:
        logger.debug("Failed to add direction arrows: %s", exc)


async def draw_map(
    page: str,
    stop_name: str,
    logo,
    routes: Dict,
    color: str,
    label_rotation: int,
    label_fontsize: int = 6,
    zoom: int = -1,
    map_provider: str = "BasemapAT",
    dpi: int = 600,
    padding: int = 15,
    tmpdir: str = tempfile.gettempdir(),
    logger=logging.getLogger(name=os.path.basename(SCRIPTDIR)),
) -> str | None:
    ax = None

    projected_geoms = []

    for route in routes["shapes"]:
        geom = route["geometry"]
        gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326").to_crs("EPSG:3857")
        projected_geoms.append(gdf.geometry.iloc[0])

        if ax is None:
            ax = gdf.plot(facecolor="none", edgecolor=color, figsize=(10, 10), linewidth=2)
        else:
            gdf.plot(ax=ax, facecolor="none", edgecolor=color, linewidth=2)

    if ax is None:
        logger.info("No routes to plot on map")
        return None

    bounds = [g.bounds for g in projected_geoms]
    xmin = min(b[0] for b in bounds)
    ymin = min(b[1] for b in bounds)
    xmax = max(b[2] for b in bounds)
    ymax = max(b[3] for b in bounds)

    xpad = (xmax - xmin) * (padding / 100.0)
    ypad = (ymax - ymin) * (padding / 100.0)
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    map_w = xmax - xmin
    map_h = ymax - ymin
    map_aspect = map_w / map_h

    A4_PORTRAIT = 210 / 297
    A4_LANDSCAPE = 297 / 210

    if map_aspect >= 1:
        pagesize = pagesizes.landscape(pagesizes.A4)
        target_aspect = A4_LANDSCAPE
    else:
        pagesize = pagesizes.portrait(pagesizes.A4)
        target_aspect = A4_PORTRAIT

    if map_aspect < target_aspect:
        target_w = map_h * target_aspect
        pad = (target_w - map_w) / 2
        xmin -= pad
        xmax += pad
    else:
        target_h = map_w / target_aspect
        pad = (target_h - map_h) / 2
        ymin -= pad
        ymax += pad

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    try:
        add_direction_arrows(ax, projected_geoms, arrow_color=color)
        logger.info("Plotted arrows on route")
    except Exception as exc:
        logger.warning("Exception while plotting arrows: %s", exc)

    try:
        n = plot_stops_on_ax(ax, routes["points"], line_color=color, endstops=routes["endstops"], label_fontsize=label_fontsize, label_rotation=label_rotation)
        logger.info("Plotted %d stops for route", n)
    except Exception as exc:
        logger.warning("Exception while plotting stops: %s", exc)

    zoom_param = zoom if zoom >= 0 else None
    try:
        await render_basemap(ax=ax, extends=(xmin, xmax, ymin, ymax), zoom=zoom_param, provider=get_provider_source(map_provider), cache_dir=os.path.join(SCRIPTDIR, "__pycache__"))
    except Exception as exc:
        logger.warning("Failed to add basemap: %s", exc)

    try:
        pdf = Canvas(page, pagesize=pagesize)

        page_w, page_h = pagesize

        margin_x = 50
        margin_y = 60

        draw_w = page_w - 2 * margin_x
        draw_h = page_h - 2 * margin_y

        _, tmp_png = tempfile.mkstemp(suffix=".png", dir=tmpdir)
        plt.savefig(tmp_png, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close()

        image = ImageReader(tmp_png)

        pdf.drawImage(
            image,
            x=margin_x,
            y=margin_y,
            width=draw_w,
            height=draw_h,
            preserveAspectRatio=True,
            anchor="c",
        )

        if isinstance(logo, tempfile._TemporaryFileWrapper):
            image = ImageReader(logo.name)
            pdf.drawImage(image, x=page_w - 130, y=20, width=80, height=30, preserveAspectRatio=True)
        elif isinstance(logo, str):
            pdf.setFont("logo", 20)
            pdf.setFillColor(colors.black)
            pdf.drawRightString(x=page_w - 20, y=23.5, text=logo)

        pdf.setFont("foot", 17)
        pdf.setFillColor(colors.black)
        pdf.drawString(x=margin_x, y=23.5, text=stop_name)

        pdf.save()
        os.remove(tmp_png)

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
        stop_color = (0.0, 0.0, 0.0)

    stop_rows = [row[1]._asdict() for row in stops]
    stops_points = [row[0] for row in stops]

    gdf_stops = gpd.GeoDataFrame(stop_rows, geometry=stops_points, crs="EPSG:4326").to_crs("EPSG:3857")
    try:
        gdf_stops.plot(ax=ax, color=stop_color, markersize=marker_size, zorder=5)
    except Exception as exc:
        logger.warning("Failed to plot stops GeoDataFrame: %s", exc)
        return 0

    already_drawn = set()
    texts = []

    for row in gdf_stops.itertuples(index=False):
        pt = row.geometry
        name = row.name

        if name not in already_drawn:
            already_drawn.add(name)
        else:
            continue

        fontweight = "bold" if name in endstops else "normal"

        if not isinstance(name, str) or not isinstance(pt, Point):
            continue
        try:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()

            dx, dy = 4, 4
            ha = "left"
            va = "center"

            if pt.x > x0 + 0.7 * (x1 - x0):
                dx = -4
                ha = "right"

            if pt.y > y0 + 0.7 * (y1 - y0):
                dy = -4
                va = "top"

            if pt.y < y0 + 0.3 * (y1 - y0):
                dy = 4
                va = "bottom"

            ddx, ddy = points_to_data_offset(ax, dx, dy)
            txt = ax.annotate(
                name,
                xy=(pt.x, pt.y),
                xytext=(pt.x + ddx, pt.y + ddy),
                textcoords="data",
                fontsize=label_fontsize,
                color="black",
                weight=fontweight,
                ha=ha,
                va=va,
                zorder=10,
                rotation=label_rotation,
                rotation_mode="anchor",
                bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "black", "linewidth": 0.2, "alpha": 2 / 3},
                clip_on=False,
            )
            txt.set_clip_path(ax.patch)
            try:
                txt.set_path_effects([pe.withStroke(linewidth=0.5, foreground="white")])
            except Exception:
                pass

            texts.append(txt)

        except Exception:
            continue

    if texts:
        try:
            adjust_text(
                texts,
                ax=ax,
                expand_text=(1, 1),
                expand_points=(1, 1),
                arrowprops=dict(
                    arrowstyle="-",
                    lw=1,
                    shrinkA=6,
                    shrinkB=6,
                    color=line_color,
                ),
                add_objects=[gdf_stops.geometry],
                only_move={"text": "xy"},
            )
        except Exception as exc:
            logger.warning("adjustText failed: %s", exc)

    return len(gdf_stops)


def get_provider_source(provider_name: str) -> TileProvider:
    """Get the contextily provider source based on the provider name."""
    return MAP_PROVIDERS.get(provider_name, providers.BasemapAT.highdpi)
