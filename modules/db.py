import os
import csv

from collections import namedtuple

import aiosqlite


_namedtuple_cache = {}

DB_PATH = "gtfs.db"

PRIMARY_KEYS = {
    "agency": "agency_id",
    "calendar_dates": ("service_id", "date"),
    "calendar": "service_id",
    "feed_info": "feed_publisher_name",
    "levels": "level_id",
    "location_cache": "stop_id",
    "pathways": "pathway_id",
    "routes": "route_id",
    "shapes": ("shape_id", "shape_pt_sequence"),
    "hst": "FID",
    "stop_times": ("trip_id", "stop_sequence"),
    "stops": "stop_id",
    "transfers": ("from_stop_id", "to_stop_id"),
    "trips": "trip_id",
}

INDICES = {
    "calendar": ["service_id"],
    "routes": ["route_id"],
    "shapes": ["shape_id"],
    "hst": ["stg_globid", "hst_globid"],
    "stop_times": ["stop_id", "trip_id"],
    "stops": ["stop_id", "stop_name"],
    "trips": ["route_id", "service_id"],
}


def namedtuple_factory(cursor, row):
    """Create namedtuple rows with caching to avoid recreating class for each row."""
    fields = tuple(column[0] for column in cursor.description)

    if len(fields) == 1:
        return row[0]

    if fields not in _namedtuple_cache:
        _namedtuple_cache[fields] = namedtuple("Row", fields)

    cls = _namedtuple_cache[fields]
    return cls._make(row)


async def _fetch_all(sql: str, params: tuple = ()):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
            return [namedtuple_factory(cur, r) for r in rows]


async def _fetch_one(sql: str, params: tuple = ()):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(sql, params) as cur:
            async for row in cur:
                yield namedtuple_factory(cur, row)


async def load_gtfs(folder: str, agency_id: int, append=True) -> None:
    """Load GTFS data."""
    async with aiosqlite.connect(DB_PATH) as db:
        if os.path.isdir(folder):
            files = os.listdir(folder)
            for file in files:
                typ, ext = os.path.splitext(os.path.basename(file))
                if ext == ".txt":
                    data_path = os.path.join(folder, file)
                    with open(data_path, "r", encoding="utf-8-sig") as f:
                        csvdata = list(csv.reader(f, skipinitialspace=True))
                    header = csvdata[0]
                    id_col_indices = {i for i, col in enumerate(header) if col.endswith("_id") or col == "parent_station" or col == "feed_publisher_name"}
                    data = csvdata[1:]
                    for row in data:
                        for i in id_col_indices:
                            if row[i] != "":
                                row[i] = f"{agency_id}_{row[i]}"
                    primary_key = PRIMARY_KEYS.get(typ)
                    amount = ",".join(["?"] * len(header))
                    if not append:
                        await db.execute(f"DROP TABLE IF EXISTS {typ}")
                    if primary_key:
                        pk_clause = ", ".join(primary_key) if isinstance(primary_key, tuple) else primary_key
                        await db.execute(f"CREATE TABLE IF NOT EXISTS {typ}({','.join(header)}, PRIMARY KEY ({pk_clause}))")
                    else:
                        await db.execute(f"CREATE TABLE IF NOT EXISTS {typ}({','.join(header)})")
                    table_cols = [col.name for col in await _fetch_all(f"PRAGMA table_info({typ})")]
                    for col in header:
                        if col not in table_cols:
                            await db.execute(f"ALTER TABLE {typ} ADD COLUMN {col}")
                    db.executemany(f"INSERT OR REPLACE INTO {typ} ({','.join(header)}) VALUES ({amount})", data)
                    for index_col in INDICES.get(typ, []):
                        await db.execute(f"CREATE INDEX IF NOT EXISTS idx_{typ}_{index_col} ON {typ} ({index_col})")
                    await db.commit()


async def load_hst_csv(file: str, append=True) -> None:
    """Load HST CSV data."""
    async with aiosqlite.connect(DB_PATH) as db:
        data_path = os.path.join(file)
        typ = "hst"
        for file in os.listdir(data_path):
            if not file.endswith(".csv"):
                continue
            with open(os.path.join(data_path, file), "r", encoding="utf-8-sig") as f:
                csvdata = list(csv.reader(f, skipinitialspace=True))
            header = csvdata[0]
            data = csvdata[1:]
            primary_key = PRIMARY_KEYS.get(typ)
            amount = ",".join(["?"] * len(header))
            if not append:
                await db.execute(f"DROP TABLE IF EXISTS {typ}")
                append = True
            if primary_key:
                pk_clause = ", ".join(primary_key) if isinstance(primary_key, tuple) else primary_key
                await db.execute(f"CREATE TABLE IF NOT EXISTS {typ}({','.join(header)}, PRIMARY KEY ({pk_clause}))")
            else:
                await db.execute(f"CREATE TABLE IF NOT EXISTS {typ}({','.join(header)})")
            table_cols = [col.name for col in await _fetch_all(f"PRAGMA table_info({typ})")]
            for col in header:
                if col not in table_cols:
                    await db.execute(f"ALTER TABLE {typ} ADD COLUMN {col}")
            await db.executemany(f"INSERT OR REPLACE INTO {typ} ({','.join(header)}) VALUES ({amount})", data)
        for index_col in INDICES.get(typ, []):
            await db.execute(f"CREATE INDEX IF NOT EXISTS idx_{typ}_{index_col} ON {typ} ({index_col})")
        await db.commit()


async def update_location_cache(mappings: dict) -> None:
    """Update location cache in the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("CREATE TABLE IF NOT EXISTS location_cache(stop_id, name, PRIMARY KEY(stop_id))")
        await db.executemany("INSERT OR REPLACE INTO location_cache (stop_id, name) VALUES (?, ?)", [(stop_id, name) for stop_id, name in mappings.items()])
        await db.commit()


async def get_table_data(table: str, columns: list | None = None, filters: dict | None = None, distinct: bool = False):
    """Get specific columns from a table with multiple filters."""
    cols = ", ".join(columns) if columns is not None else "*"
    conditions = " AND ".join([f"{col} = ?" for col in filters.keys()]) if filters is not None else ""
    values = tuple(filters.values()) if filters is not None else ()
    dist = "DISTINCT " if distinct else ""
    if conditions == "":
        return await _fetch_all(f"SELECT {dist}{cols} FROM {table}")
    else:
        return await _fetch_all(f"SELECT {dist}{cols} FROM {table} WHERE {conditions}", values)


async def get_table_data_iter(table: str, columns: list | None = None, filters: dict | None = None, distinct: bool = False):
    """Get specific columns from a table as a generator (memory efficient for large results)."""
    cols = ", ".join(columns) if columns is not None else "*"
    conditions = " AND ".join([f"{col} = ?" for col in filters.keys()]) if filters is not None else ""
    values = tuple(filters.values()) if filters is not None else ()
    dist = "DISTINCT " if distinct else ""
    if conditions == "":
        return _fetch_one(f"SELECT {dist}{cols} FROM {table}")
    else:
        return _fetch_one(f"SELECT {dist}{cols} FROM {table} WHERE {conditions}", values)


async def get_in_filtered_data(table: str, column: str, values: list, columns: list | None = None, distinct: bool = False):
    """Get data from a table where a column's value is in a list."""
    cols = ", ".join(columns) if columns is not None else "*"
    placeholders = ",".join(["?"] * len(values))
    dist = "DISTINCT " if distinct else ""
    return await _fetch_all(f"SELECT {dist}{cols} FROM {table} WHERE {column} IN ({placeholders})", tuple(values))


async def get_in_filtered_data_iter(table: str, column: str, values: list, columns: list | None = None, distinct: bool = False):
    """Get data from a table where a column's value is in a list as a generator (memory efficient)."""
    cols = ", ".join(columns) if columns is not None else "*"
    placeholders = ",".join(["?"] * len(values))
    dist = "DISTINCT " if distinct else ""
    return _fetch_one(f"SELECT {dist}{cols} FROM {table} WHERE {column} IN ({placeholders})", tuple(values))


async def get_most_frequent_values(table: str, column: str, filters: dict | None = None, limit: int = 1):
    """Get the most frequent values in a specified column of a table."""
    if filters is not None:
        conditions = " AND ".join([f"{col} = ?" for col in filters.keys()])
        values = tuple(filters.values())
        return await _fetch_all(f"SELECT {column}, COUNT(*) as count FROM {table} WHERE {conditions} GROUP BY {column} ORDER BY count DESC LIMIT ?", (*values, limit))
    else:
        return await _fetch_all(f"SELECT {column}, COUNT(*) as count FROM {table} GROUP BY {column} ORDER BY count DESC LIMIT ?", (limit,))
