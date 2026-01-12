import os
import csv
import json
import sqlite3
from collections import namedtuple


_namedtuple_cache = {}


def namedtuple_factory(cursor, row):
    """Create namedtuple rows with caching to avoid recreating class for each row."""
    fields = tuple(column[0] for column in cursor.description)

    if len(fields) == 1:
        return row[0]

    if fields not in _namedtuple_cache:
        _namedtuple_cache[fields] = namedtuple("Row", fields)

    cls = _namedtuple_cache[fields]
    return cls._make(row)


con = sqlite3.connect("gtfs.db")
con.row_factory = namedtuple_factory
cur = con.cursor()

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
    "stg": "stg_id",
    "stop_times": ("trip_id", "stop_sequence"),
    "stops": "stop_id",
    "transfers": ("from_stop_id", "to_stop_id"),
    "trips": "trip_id",
}

INDICES = {
    "calendar": ["service_id"],
    "routes": ["route_id"],
    "shapes": ["shape_id"],
    "stg": ["stg_globid"],
    "stop_times": ["stop_id", "trip_id"],
    "stops": ["stop_id", "stop_name"],
    "trips": ["route_id", "service_id"],
}


def load_gtfs(folder: str, agency_id: int, append=True) -> None:
    """Load GTFS data."""
    if os.path.isdir(folder):
        files = os.listdir(folder)
        for file in files:
            typ, ext = os.path.splitext(os.path.basename(file))
            if ext == ".txt":
                data_path = os.path.join(folder, file)
                with open(data_path, "r", encoding="utf-8-sig") as f:
                    csvdata = list(csv.reader(f, skipinitialspace=True))
                header = csvdata[0]
                id_col_indices = {i for i, col in enumerate(header) if col.endswith("_id") or col == "parent_station"}
                data = csvdata[1:]
                for row in data:
                    for i in id_col_indices:
                        if row[i] != "":
                            row[i] = f"{agency_id}_{row[i]}"
                primary_key = PRIMARY_KEYS.get(typ)
                amount = ",".join(["?"] * len(header))
                if not append:
                    con.execute(f"DROP TABLE IF EXISTS {typ}")
                if primary_key:
                    pk_clause = ", ".join(primary_key) if isinstance(primary_key, tuple) else primary_key
                    con.execute(f"CREATE TABLE IF NOT EXISTS {typ}({','.join(header)}, PRIMARY KEY ({pk_clause}))")
                else:
                    con.execute(f"CREATE TABLE IF NOT EXISTS {typ}({','.join(header)})")
                table_cols = [col.name for col in con.execute(f"PRAGMA table_info({typ})").fetchall()]
                for col in header:
                    if col not in table_cols:
                        con.execute(f"ALTER TABLE {typ} ADD COLUMN {col}")
                con.executemany(f"INSERT OR REPLACE INTO {typ} ({','.join(header)}) VALUES ({amount})", data)
                for index_col in INDICES.get(typ, []):
                    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{typ}_{index_col} ON {typ} ({index_col})")
                con.commit()


def load_stg_json(file: str, append=True) -> None:
    """Load STG JSON data."""
    data_path = os.path.join(file)
    with open(data_path, "r", encoding="utf-8-sig") as f:
        jsondata = [feature["properties"] for feature in json.load(f)["features"]]
        header = jsondata[0].keys()
        primary_key = PRIMARY_KEYS.get("stg")
        amount = ",".join(["?"] * len(header))
        if not append:
            con.execute("DROP TABLE IF EXISTS stg")
        if primary_key:
            pk_clause = ", ".join(primary_key) if isinstance(primary_key, tuple) else primary_key
            con.execute(f"CREATE TABLE IF NOT EXISTS stg({','.join(header)}, PRIMARY KEY ({pk_clause}))")
        else:
            con.execute(f"CREATE TABLE IF NOT EXISTS stg({','.join(header)})")
        values = [tuple(record.values()) for record in jsondata]
        con.executemany(f"INSERT OR REPLACE INTO stg VALUES ({amount})", values)
        con.commit()


def update_location_cache(mappings: dict) -> None:
    """Update location cache in the database."""
    con.execute("CREATE TABLE IF NOT EXISTS location_cache(stop_id, name, PRIMARY KEY(stop_id))")
    con.executemany("INSERT OR REPLACE INTO location_cache (stop_id, name) VALUES (?, ?)", [(stop_id, name) for stop_id, name in mappings.items()])
    con.commit()


def get_table_data(table: str, columns: list | None = None, filters: dict | None = None, distinct: bool = False):
    """Get specific columns from a table with multiple filters."""
    cols = ", ".join(columns) if columns is not None else "*"
    conditions = " AND ".join([f"{col} = ?" for col in filters.keys()]) if filters is not None else ""
    values = tuple(filters.values()) if filters is not None else ()
    dist = "DISTINCT " if distinct else ""
    if conditions == "":
        cur.execute(f"SELECT {dist}{cols} FROM {table}")
    else:
        cur.execute(f"SELECT {dist}{cols} FROM {table} WHERE {conditions}", values)
    return cur.fetchall()


def get_table_data_iter(table: str, columns: list | None = None, filters: dict | None = None, distinct: bool = False):
    """Get specific columns from a table as a generator (memory efficient for large results)."""
    cols = ", ".join(columns) if columns is not None else "*"
    conditions = " AND ".join([f"{col} = ?" for col in filters.keys()]) if filters is not None else ""
    values = tuple(filters.values()) if filters is not None else ()
    dist = "DISTINCT " if distinct else ""
    if conditions == "":
        cur.execute(f"SELECT {dist}{cols} FROM {table}")
    else:
        cur.execute(f"SELECT {dist}{cols} FROM {table} WHERE {conditions}", values)

    while True:
        row = cur.fetchone()
        if row is None:
            break
        yield row


def get_in_filtered_data(table: str, column: str, values: list, columns: list | None = None, distinct: bool = False):
    """Get data from a table where a column's value is in a list."""
    cols = ", ".join(columns) if columns is not None else "*"
    placeholders = ",".join(["?"] * len(values))
    dist = "DISTINCT " if distinct else ""
    cur.execute(f"SELECT {dist}{cols} FROM {table} WHERE {column} IN ({placeholders})", tuple(values))
    return cur.fetchall()


def get_in_filtered_data_iter(table: str, column: str, values: list, columns: list | None = None, distinct: bool = False):
    """Get data from a table where a column's value is in a list as a generator (memory efficient)."""
    cols = ", ".join(columns) if columns is not None else "*"
    placeholders = ",".join(["?"] * len(values))
    dist = "DISTINCT " if distinct else ""
    cur.execute(f"SELECT {dist}{cols} FROM {table} WHERE {column} IN ({placeholders})", tuple(values))

    while True:
        row = cur.fetchone()
        if row is None:
            break
        yield row


def get_most_frequent_values(table: str, column: str, filters: dict | None = None, limit: int = 1):
    """Get the most frequent values in a specified column of a table."""
    if filters is not None:
        conditions = " AND ".join([f"{col} = ?" for col in filters.keys()])
        values = tuple(filters.values())
        cur.execute(f"SELECT {column}, COUNT(*) as count FROM {table} WHERE {conditions} GROUP BY {column} ORDER BY count DESC LIMIT ?", (*values, limit))
    else:
        cur.execute(f"SELECT {column}, COUNT(*) as count FROM {table} GROUP BY {column} ORDER BY count DESC LIMIT ?", (limit,))
    return cur.fetchall()
