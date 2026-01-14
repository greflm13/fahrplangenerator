#!/usr/bin/env python

import os
import re
import logging
import tempfile

# import urllib.parse
from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException, Form, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

import modules.utils as utils
import modules.db as db

from fahrplan import compute
from modules.map import MAP_PROVIDERS

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
        stops = db.get_table_data("stops")
        stop_hierarchy = utils.build_stop_hierarchy()
        stop_hierarchy = utils.query_stop_names(stop_hierarchy, loadingbars=False)
        destinations = utils.build_dest_list()

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
        TMPDIR = tempfile.mkdtemp()

        logger.info("Loaded GTFS data")
        logger.info(f"Temporary directory created at {TMPDIR}")
    except Exception as e:
        logger.error(f"Error loading GTFS data at startup: {str(e)}")

    yield

    if os.path.exists(TMPDIR):
        for filename in os.listdir(TMPDIR):
            file_path = os.path.join(TMPDIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Error removing temporary file {file_path}: {str(e)}")
        try:
            os.rmdir(TMPDIR)
            logger.info("Cleaned up temporary directory")
        except Exception as e:
            logger.error(f"Error removing temporary directory {TMPDIR}: {str(e)}")


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


@app.post("/generate", response_class=FileResponse)
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

        ourstop = stop_hierarchy[stop_id_mapping[request.station_name][0]]
        if len(stop_id_mapping[request.station_name]) > 1:
            combined_children = []
            for stop_id in stop_id_mapping[request.station_name]:
                stop = stop_hierarchy[stop_id]
                if stop.children is not None:
                    combined_children.extend(stop.children)
            ourstop.children = combined_children

        logger.info(f"Generating timetable for {request.station_name}")

        safe_station = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", request.station_name).strip()
        safe_station = safe_station.replace(os.path.sep, "_")
        if not safe_station:
            safe_station = "fahrplan"
        args.output = os.path.join(TMPDIR, f"{safe_station}.pdf")

        compute(ourstop, stops, args, destinations, False)

        if not os.path.exists(args.output):
            raise HTTPException(status_code=500, detail="Failed to generate PDF file")

        logger.info(f"Timetable generated: {args.output}")
        return FileResponse(path=args.output, filename=os.path.basename(args.output), media_type="application/pdf")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating timetable: {str(e)}")
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
        logger.error(f"Error fetching stations: {str(e)}")
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

    uvicorn.run(app, host="0.0.0.0", port=8000)
