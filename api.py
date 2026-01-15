#!/usr/bin/env python

import os
import re
import base64
import logging
import tempfile

from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

import modules.utils as utils
import modules.db as db

from fahrplan import compute
from modules.map import MAP_PROVIDERS
from modules.logger import rotate_log_file, setup_logger

logger = logging.getLogger("uvicorn.error")

STOP_HIERARCHY = None
STOP_ID_MAPPING = None
STOPS = None
DESTINATIONS = None
TMPDIR = "/tmp"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load GTFS data at startup."""
    global STOP_HIERARCHY, STOP_ID_MAPPING, STOPS, DESTINATIONS, TMPDIR
    try:
        stops = await db.get_table_data("stops")
        stop_hierarchy = await utils.build_stop_hierarchy()
        stop_hierarchy = await utils.query_stop_names(stop_hierarchy, loadingbars=False)
        destinations = await utils.build_dest_list()

        stop_id_mapping = {}
        for stop in stop_hierarchy.values():
            if stop.stop_name not in stop_id_mapping:
                stop_id_mapping[stop.stop_name] = [stop.stop_id]
            else:
                stop_id_mapping[stop.stop_name].append(stop.stop_id)

        stops = utils.build_list_index(stops, "stop_id")

        STOP_HIERARCHY = stop_hierarchy
        STOP_ID_MAPPING = stop_id_mapping
        STOPS = stops
        DESTINATIONS = destinations
        TMPDIR = tempfile.mkdtemp(prefix="fahrplan_api_")

        logger.info("Loaded GTFS data")
        logger.info("Temporary directory created at %s", TMPDIR)
    except Exception as e:
        logger.error("Error loading GTFS data at startup: %s", str(e))

    yield

    if os.path.exists(TMPDIR):
        for filename in os.listdir(TMPDIR):
            file_path = os.path.join(TMPDIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.error("Error removing temporary file %s: %s", file_path, str(e))
        try:
            os.rmdir(TMPDIR)
            logger.info("Cleaned up temporary directory")
        except Exception as e:
            logger.error("Error removing temporary directory %s: %s", TMPDIR, str(e))


app = FastAPI(title="Fahrplan Generator API", description="Generate transit timetables", version="1.0.0", lifespan=lifespan, root_path="/api")


class FahrplanRequest(BaseModel):
    """Request model for timetable generation"""

    station_name: str = Field(..., description="Name of the station/stop")
    generate_map: bool = Field(default=False, description="Generate maps for routes")
    color: list = Field(default=["random"], description="Timetable color (or 'random')")
    map_provider: str = Field(default="BasemapAT", description="Map provider (BasemapAT, OPNVKarte, OSM, OSMDE, ORM, OTM, UN, SAT)")
    map_dpi: Optional[int] = Field(default=None, description="Map DPI resolution")


class RootResponse(BaseModel):
    """Response model for root endpoint"""

    status: str = Field(..., description="Status of the API")
    message: str = Field(..., description="Message about the API")
    endpoints: dict[str, str] = Field(..., description="Available API endpoints")


class StationsRequest(BaseModel):
    """Request model for available stations"""

    query: Optional[str] = Field(default=None, description="Search query for station names")


class StationsResponse(BaseModel):
    """Response model for available stations"""

    total: int = Field(..., description="Total number of stations")
    stations: list[str] = Field(..., description="List of station names")


class MapProvidersResponse(BaseModel):
    """Response model for available map providers"""

    map_providers: list[str] = Field(..., description="List of map providers")


class Info(BaseModel):
    """Response model for API information"""

    api_version: str = Field(..., description="API version")
    title: str = Field(..., description="API title")
    description: str = Field(..., description="API description")
    available_map_providers: list[str] = Field(..., description="List of available map providers")
    endpoints: dict[str, str] = Field(..., description="Available API endpoints")


class Args:
    """Simple class to mimic argparse Namespace"""

    def __init__(self, generate_map: bool = False, color: str = "random", map_provider: str = "BasemapAT", map_dpi: Optional[int] = None):
        self.input = []
        self.color = color
        self.output = "fahrplan.pdf"
        self.map = generate_map
        self.mapping_csv = None
        self.reset_db = False
        self.map_dpi = map_dpi
        self.logo = True
        self.map_provider = map_provider


@app.get("/", response_model=RootResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Fahrplan Generator API",
        "endpoints": {
            "POST /generate": "Generate a timetable",
            "GET /stations": "Get available stations",
            "GET /map-providers": "Get available map providers",
            "GET /info": "Get API information",
        },
    }


@app.get("/download", response_class=FileResponse)
async def download_timetable(dl: str):
    """Download a previously generated timetable PDF using a unique token."""
    try:
        if not dl:
            raise HTTPException(status_code=400, detail="No download token provided")

        file = dl.split(":")
        filepath = base64.b64decode(file[0]).decode("utf-8")
        filename = base64.b64decode(file[1]).decode("utf-8")
        if not filepath:
            raise HTTPException(status_code=400, detail="Invalid download token")

        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Requested timetable not found")

        logger.info("Timetable downloaded: %s", filepath)
        return FileResponse(path=filepath, filename=filename, media_type="application/pdf")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error downloading timetable: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error downloading timetable: {str(e)}")


@app.post("/generate")
async def generate_timetable(request: Annotated[FahrplanRequest, Form()]):
    """Generate a transit timetable PDF for the given station."""
    try:
        if request.color[0] != "random":
            request.color[0] = request.color[1]
        args = Args(generate_map=request.generate_map, color=request.color[0], map_provider=request.map_provider, map_dpi=request.map_dpi)

        global STOP_HIERARCHY, STOP_ID_MAPPING, STOPS, DESTINATIONS
        if STOPS is None or STOP_HIERARCHY is None or DESTINATIONS is None or STOP_ID_MAPPING is None:
            raise HTTPException(status_code=400, detail="No GTFS data loaded. Please load GTFS data files first or provide input_folders.")

        stops = STOPS
        stop_hierarchy = STOP_HIERARCHY
        destinations = DESTINATIONS
        stop_id_mapping = STOP_ID_MAPPING

        if request.station_name not in stop_id_mapping:
            raise HTTPException(status_code=400, detail=f"Station '{request.station_name}' not found")

        if len(stop_id_mapping[request.station_name]) == 1:
            ourstop = [stop_hierarchy[stop_id_mapping[request.station_name][0]]]
            if ourstop[0].children is not None:
                ourstop.extend(ourstop[0].children)
                del ourstop[0].children
        else:
            ourstop = []
            for stop_id in stop_id_mapping[request.station_name]:
                stop = stop_hierarchy[stop_id]
                if stop.children is not None:
                    ourstop.extend(stop.children)
                    del stop.children
                ourstop.append(stop)

        logger.info("Generating timetable for %s", request.station_name)

        safe_station = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", request.station_name).strip()
        safe_station = safe_station.replace(os.path.sep, "_")
        if not safe_station:
            safe_station = "fahrplan"
        outfile = os.path.join(TMPDIR, f"{safe_station}.pdf")

        _, args.output = tempfile.mkstemp(suffix=".pdf", prefix=f"{safe_station}_", dir=TMPDIR)

        await compute(ourstop, stops, args, destinations, False, logger)

        if not os.path.exists(args.output):
            raise HTTPException(status_code=500, detail="Failed to generate PDF file")

        logger.info("Timetable generated: %s", args.output)
        filepath = base64.b64encode(bytes(args.output, "utf-8"))
        filename = base64.b64encode(bytes(os.path.basename(outfile), "utf-8"))
        dl = filepath.decode("utf-8") + ":" + filename.decode("utf-8")
        return JSONResponse(content={"message": "PDF generated", "download": dl})

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error generating timetable: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error generating timetable: {str(e)}")


@app.get("/stations", response_model=StationsResponse)
async def get_available_stations(request: Annotated[StationsRequest, Query()]):
    """
    Get a list of all available stations in the database.
    """
    global STOP_ID_MAPPING
    if STOP_ID_MAPPING is None:
        raise HTTPException(status_code=400, detail="No GTFS data loaded. Please load GTFS data files first or provide input_folders.")
    try:
        stations = STOP_ID_MAPPING

        stations = sorted(STOP_ID_MAPPING.keys())

        if request.query:
            query_lower = request.query.lower()
            stations = [station for station in stations if query_lower in station.lower()]

        return {"total": len(stations), "stations": stations}
    except Exception as e:
        logger.error("Error fetching stations: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error fetching stations: {str(e)}")


@app.get("/map-providers", response_model=MapProvidersResponse)
async def get_map_providers():
    """Get a list of available map providers."""
    return {"map_providers": list(MAP_PROVIDERS.keys())}


@app.get("/info", response_model=Info)
async def get_info():
    """Get API information and available options"""
    return {
        "api_version": "1.0.0",
        "title": "Fahrplan Generator API",
        "description": "Generate transit timetables with optional maps",
        "available_map_providers": ["BasemapAT", "OPNVKarte", "OSM", "OSMDE", "ORM", "OTM", "UN", "SAT"],
        "endpoints": {
            "GET /": "Health check",
            "POST /generate": "Generate a timetable",
            "GET /stations": "Get available stations",
            "GET /map-providers": "Get available map providers",
            "GET /info": "Get API information",
        },
    }


if __name__ == "__main__":
    import uvicorn

    rotate_log_file()
    setup_logger()
    uvicorn.run(app, host="0.0.0.0", port=8000)
