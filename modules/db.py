import os
import csv
import sqlite3
from collections import namedtuple


def namedtuple_factory(cursor, row):
    fields = [column[0] for column in cursor.description]
    cls = namedtuple("Row", fields)
    return cls._make(row)


con = sqlite3.connect("gtfs.db")
con.row_factory = namedtuple_factory
cur = con.cursor()


def load_gtfs(folder: str, append=True) -> None:
    """Load GTFS data."""
    if os.path.isdir(folder):
        files = os.listdir(folder)
        for file in files:
            typ, ext = os.path.splitext(os.path.basename(file))
            if ext == ".txt":
                data_path = os.path.join(folder, file)
                with open(data_path, "r", encoding="utf-8-sig") as f:
                    csvdata = csv.reader(f, skipinitialspace=True)
                    csvlines = []
                    for row in csvdata:
                        csvlines.append(tuple(row))
                    header = csvlines[0]
                    data = csvlines[1:]
                    amount = ",".join(["?"] * len(header))
                    if not append:
                        con.execute(f"DROP TABLE IF EXISTS {typ}")
                    con.execute(f"CREATE TABLE IF NOT EXISTS {typ}({','.join(header)})")
                    con.executemany(f"INSERT INTO {typ} VALUES ({amount})", data)
                    con.commit()


load_gtfs("/home/user/Documents/20251231-0159_gtfs_evu_2026")


def get_table_data(table: str):
    """Get all data from a table."""
    cur.execute(f"SELECT * FROM {table}")
    return cur.fetchall()


def get_filtered_data(table: str, column: str, value):
    """Get filtered data from a table."""
    cur.execute(f"SELECT * FROM {table} WHERE {column} = ?", (value,))
    return cur.fetchall()


stops = get_filtered_data("stops", "stop_name", "Weststeiermark Bahnhof")
for stop in stops:
    print(stop)
