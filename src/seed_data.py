"""Generate synthetic demo data for AI-partments and load it into DuckDB.

Run with:

    python -m src.seed_data

This populates ``db/renteriq.duckdb`` with 12 NYC/NJ neighborhoods plus
plausible apartments, amenities, parks, facilities, transit stops, crime
events, and pre-computed neighborhood scores so Mock Demo Mode works
without any real data files.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DB_PATH, SAMPLE_DIR, ensure_directories
from src.database import initialize_database, replace_table


# Deterministic seed so demos are reproducible.
RANDOM_SEED = 42


# --- Reference data -----------------------------------------------------------

NEIGHBORHOODS: list[dict] = [
    {
        "name": "Astoria",
        "borough": "Queens",
        "city": "New York",
        "state": "NY",
        "nta_code": "QN0103",
        "center_lat": 40.7720,
        "center_lng": -73.9300,
        "rent_band": (2200, 3400),
        "safety": 75,
        "transit": 85,
        "green": 70,
        "facilities": 70,
        "housing": 65,
        "population": 78000,
        "median_income": 78000,
    },
    {
        "name": "Sunnyside",
        "borough": "Queens",
        "city": "New York",
        "state": "NY",
        "nta_code": "QN0202",
        "center_lat": 40.7430,
        "center_lng": -73.9200,
        "rent_band": (1900, 3000),
        "safety": 80,
        "transit": 78,
        "green": 60,
        "facilities": 65,
        "housing": 60,
        "population": 45000,
        "median_income": 72000,
    },
    {
        "name": "Harlem",
        "borough": "Manhattan",
        "city": "New York",
        "state": "NY",
        "nta_code": "MN1101",
        "center_lat": 40.8116,
        "center_lng": -73.9465,
        "rent_band": (2100, 3500),
        "safety": 60,
        "transit": 88,
        "green": 75,
        "facilities": 80,
        "housing": 78,
        "population": 116000,
        "median_income": 49000,
    },
    {
        "name": "Washington Heights",
        "borough": "Manhattan",
        "city": "New York",
        "state": "NY",
        "nta_code": "MN1201",
        "center_lat": 40.8417,
        "center_lng": -73.9393,
        "rent_band": (1800, 2900),
        "safety": 65,
        "transit": 80,
        "green": 78,
        "facilities": 70,
        "housing": 72,
        "population": 153000,
        "median_income": 47000,
    },
    {
        "name": "Williamsburg",
        "borough": "Brooklyn",
        "city": "New York",
        "state": "NY",
        "nta_code": "BK0902",
        "center_lat": 40.7081,
        "center_lng": -73.9571,
        "rent_band": (2700, 4500),
        "safety": 70,
        "transit": 80,
        "green": 65,
        "facilities": 72,
        "housing": 68,
        "population": 78000,
        "median_income": 86000,
    },
    {
        "name": "Bushwick",
        "borough": "Brooklyn",
        "city": "New York",
        "state": "NY",
        "nta_code": "BK0701",
        "center_lat": 40.6944,
        "center_lng": -73.9213,
        "rent_band": (1900, 3200),
        "safety": 58,
        "transit": 72,
        "green": 50,
        "facilities": 60,
        "housing": 65,
        "population": 130000,
        "median_income": 50000,
    },
    {
        "name": "Park Slope",
        "borough": "Brooklyn",
        "city": "New York",
        "state": "NY",
        "nta_code": "BK1101",
        "center_lat": 40.6710,
        "center_lng": -73.9814,
        "rent_band": (2800, 4500),
        "safety": 85,
        "transit": 78,
        "green": 88,
        "facilities": 80,
        "housing": 60,
        "population": 65000,
        "median_income": 130000,
    },
    {
        "name": "Jersey City Heights",
        "borough": "Hudson",
        "city": "Jersey City",
        "state": "NJ",
        "nta_code": "NJJC01",
        "center_lat": 40.7470,
        "center_lng": -74.0480,
        "rent_band": (1800, 2800),
        "safety": 72,
        "transit": 65,
        "green": 65,
        "facilities": 60,
        "housing": 55,
        "population": 51000,
        "median_income": 75000,
    },
    {
        "name": "Journal Square",
        "borough": "Hudson",
        "city": "Jersey City",
        "state": "NJ",
        "nta_code": "NJJC02",
        "center_lat": 40.7320,
        "center_lng": -74.0630,
        "rent_band": (1700, 2600),
        "safety": 68,
        "transit": 88,
        "green": 55,
        "facilities": 62,
        "housing": 60,
        "population": 35000,
        "median_income": 67000,
    },
    {
        "name": "Hoboken",
        "borough": "Hudson",
        "city": "Hoboken",
        "state": "NJ",
        "nta_code": "NJHK01",
        "center_lat": 40.7440,
        "center_lng": -74.0320,
        "rent_band": (2500, 4200),
        "safety": 86,
        "transit": 90,
        "green": 70,
        "facilities": 75,
        "housing": 55,
        "population": 53000,
        "median_income": 145000,
    },
    {
        "name": "Mott Haven",
        "borough": "Bronx",
        "city": "New York",
        "state": "NY",
        "nta_code": "BX0101",
        "center_lat": 40.8087,
        "center_lng": -73.9234,
        "rent_band": (1500, 2500),
        "safety": 55,
        "transit": 78,
        "green": 60,
        "facilities": 65,
        "housing": 75,
        "population": 91000,
        "median_income": 30000,
    },
    {
        "name": "Flushing",
        "borough": "Queens",
        "city": "New York",
        "state": "NY",
        "nta_code": "QN2201",
        "center_lat": 40.7654,
        "center_lng": -73.8318,
        "rent_band": (1800, 3000),
        "safety": 78,
        "transit": 75,
        "green": 80,
        "facilities": 78,
        "housing": 70,
        "population": 220000,
        "median_income": 60000,
    },
]


WORSHIP_NAMES = [
    ("Masjid An-Noor", "mosque"),
    ("Islamic Center", "mosque"),
    ("Trinity Church", "church"),
    ("St. Mary Cathedral", "church"),
    ("Beth Shalom Synagogue", "synagogue"),
    ("Sri Ganesh Temple", "temple"),
    ("Buddhist Meditation Hall", "temple"),
    ("Community Worship House", "worship"),
]

GROCERY_NAMES = ["Trader Joe's", "Whole Foods", "Key Food", "C-Town", "Associated Supermarket"]
SCHOOL_NAMES = ["P.S. 12", "P.S. 47", "I.S. 88", "Magnet Charter", "Community Prep"]
HEALTHCARE_NAMES = ["NYC Health Clinic", "Urgent Care Plus", "Family Medical Center"]
LIBRARY_NAMES = ["Public Library Branch", "Community Library"]
COMMUNITY_NAMES = ["Senior Activity Center", "Youth Community Center", "Cultural Hub"]

PARK_NAMES = ["Riverside Park", "Astoria Park", "Prospect Park", "Sunset Park", "Marcus Garvey Park", "Fort Tryon Park"]

OFFENSE_TYPES = [
    ("Petit Larceny", "property"),
    ("Grand Larceny", "property"),
    ("Burglary", "property"),
    ("Auto Theft", "property"),
    ("Assault 3", "violent"),
    ("Robbery", "violent"),
    ("Felony Assault", "violent"),
    ("Harassment", "other"),
    ("Criminal Mischief", "other"),
]

STREETS = [
    "Broadway",
    "Steinway Street",
    "Roosevelt Ave",
    "Bedford Ave",
    "Atlantic Ave",
    "Union Tpke",
    "Northern Blvd",
    "Linden Blvd",
    "Knickerbocker Ave",
    "Eastern Pkwy",
    "Grand St",
    "Lex Ave",
    "Amsterdam Ave",
    "Queens Blvd",
]


# --- Helpers ------------------------------------------------------------------


def _jitter(center: float, miles: float = 0.6) -> float:
    """Return a lat/lng coordinate near ``center`` (rough miles offset)."""
    delta = miles / 69.0
    return center + random.uniform(-delta, delta)


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 50.0
    return max(0.0, min(100.0, 100 * (value - lo) / (hi - lo)))


# --- Seed builders ------------------------------------------------------------


def build_neighborhoods() -> pd.DataFrame:
    rows: list[dict] = []
    for idx, nb in enumerate(NEIGHBORHOODS, start=1):
        rows.append(
            {
                "neighborhood_id": idx,
                "nta_code": nb["nta_code"],
                "name": nb["name"],
                "borough": nb["borough"],
                "city": nb["city"],
                "state": nb["state"],
                "center_lat": nb["center_lat"],
                "center_lng": nb["center_lng"],
            }
        )
    return pd.DataFrame(rows)


def build_apartments(neighborhoods: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    apt_id = 1
    today = datetime.today()
    for nb_meta, (_, nb_row) in zip(NEIGHBORHOODS, neighborhoods.iterrows()):
        n_listings = random.randint(6, 10)
        rent_lo, rent_hi = nb_meta["rent_band"]
        for _ in range(n_listings):
            bedrooms = random.choices([0, 1, 2, 3], weights=[2, 5, 4, 1])[0]
            bedroom_bonus = bedrooms * 350
            rent = int(random.uniform(rent_lo, rent_hi) + bedroom_bonus)
            sqft = 350 + bedrooms * 250 + random.randint(-60, 80)
            available_in = random.randint(0, 60)
            available_date = (today + timedelta(days=available_in)).date().isoformat()
            rows.append(
                {
                    "apartment_id": apt_id,
                    "neighborhood_id": int(nb_row["neighborhood_id"]),
                    "address": f"{random.randint(50, 950)} {random.choice(STREETS)}",
                    "rent": rent,
                    "bedrooms": bedrooms,
                    "bathrooms": float(random.choice([1.0, 1.0, 1.5, 2.0])),
                    "sqft": int(max(280, sqft)),
                    "available_date": available_date,
                    "lat": _jitter(nb_meta["center_lat"], 0.5),
                    "lng": _jitter(nb_meta["center_lng"], 0.5),
                    "source": "demo",
                    "listing_url": f"https://example.com/listing/{apt_id}",
                }
            )
            apt_id += 1
    return pd.DataFrame(rows)


def build_amenities(neighborhoods: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    amenity_id = 1
    for nb_meta, (_, nb_row) in zip(NEIGHBORHOODS, neighborhoods.iterrows()):
        nb_id = int(nb_row["neighborhood_id"])
        # Parks
        for _ in range(random.randint(2, 5)):
            rows.append(
                {
                    "amenity_id": amenity_id,
                    "neighborhood_id": nb_id,
                    "name": random.choice(PARK_NAMES),
                    "type": "park",
                    "lat": _jitter(nb_meta["center_lat"], 0.5),
                    "lng": _jitter(nb_meta["center_lng"], 0.5),
                    "distance_miles": round(random.uniform(0.05, 0.6), 2),
                    "source_file": "demo",
                }
            )
            amenity_id += 1
        # Worship
        for name, w_type in random.sample(WORSHIP_NAMES, k=random.randint(2, 4)):
            rows.append(
                {
                    "amenity_id": amenity_id,
                    "neighborhood_id": nb_id,
                    "name": name,
                    "type": w_type,
                    "lat": _jitter(nb_meta["center_lat"], 0.4),
                    "lng": _jitter(nb_meta["center_lng"], 0.4),
                    "distance_miles": round(random.uniform(0.05, 0.5), 2),
                    "source_file": "demo",
                }
            )
            amenity_id += 1
        # Grocery
        for name in random.sample(GROCERY_NAMES, k=random.randint(2, 3)):
            rows.append(
                {
                    "amenity_id": amenity_id,
                    "neighborhood_id": nb_id,
                    "name": name,
                    "type": "grocery",
                    "lat": _jitter(nb_meta["center_lat"], 0.3),
                    "lng": _jitter(nb_meta["center_lng"], 0.3),
                    "distance_miles": round(random.uniform(0.05, 0.4), 2),
                    "source_file": "demo",
                }
            )
            amenity_id += 1
        # Schools / healthcare / library / community
        for name in random.sample(SCHOOL_NAMES, k=random.randint(1, 3)):
            rows.append(
                {
                    "amenity_id": amenity_id,
                    "neighborhood_id": nb_id,
                    "name": name,
                    "type": "school",
                    "lat": _jitter(nb_meta["center_lat"], 0.4),
                    "lng": _jitter(nb_meta["center_lng"], 0.4),
                    "distance_miles": round(random.uniform(0.1, 0.6), 2),
                    "source_file": "demo",
                }
            )
            amenity_id += 1
        for name in random.sample(HEALTHCARE_NAMES, k=random.randint(1, 2)):
            rows.append(
                {
                    "amenity_id": amenity_id,
                    "neighborhood_id": nb_id,
                    "name": name,
                    "type": "healthcare",
                    "lat": _jitter(nb_meta["center_lat"], 0.4),
                    "lng": _jitter(nb_meta["center_lng"], 0.4),
                    "distance_miles": round(random.uniform(0.1, 0.6), 2),
                    "source_file": "demo",
                }
            )
            amenity_id += 1
        for name in random.sample(LIBRARY_NAMES, k=1):
            rows.append(
                {
                    "amenity_id": amenity_id,
                    "neighborhood_id": nb_id,
                    "name": name,
                    "type": "library",
                    "lat": _jitter(nb_meta["center_lat"], 0.3),
                    "lng": _jitter(nb_meta["center_lng"], 0.3),
                    "distance_miles": round(random.uniform(0.1, 0.5), 2),
                    "source_file": "demo",
                }
            )
            amenity_id += 1
        for name in random.sample(COMMUNITY_NAMES, k=random.randint(1, 2)):
            rows.append(
                {
                    "amenity_id": amenity_id,
                    "neighborhood_id": nb_id,
                    "name": name,
                    "type": "community_center",
                    "lat": _jitter(nb_meta["center_lat"], 0.4),
                    "lng": _jitter(nb_meta["center_lng"], 0.4),
                    "distance_miles": round(random.uniform(0.1, 0.6), 2),
                    "source_file": "demo",
                }
            )
            amenity_id += 1
        # Subway / bus markers (so the map shows transit dots).
        for _ in range(random.randint(1, 3)):
            rows.append(
                {
                    "amenity_id": amenity_id,
                    "neighborhood_id": nb_id,
                    "name": f"{nb_meta['name']} Station",
                    "type": "subway",
                    "lat": _jitter(nb_meta["center_lat"], 0.3),
                    "lng": _jitter(nb_meta["center_lng"], 0.3),
                    "distance_miles": round(random.uniform(0.05, 0.4), 2),
                    "source_file": "demo",
                }
            )
            amenity_id += 1
        for _ in range(random.randint(3, 6)):
            rows.append(
                {
                    "amenity_id": amenity_id,
                    "neighborhood_id": nb_id,
                    "name": "Bus Stop",
                    "type": "bus",
                    "lat": _jitter(nb_meta["center_lat"], 0.4),
                    "lng": _jitter(nb_meta["center_lng"], 0.4),
                    "distance_miles": round(random.uniform(0.05, 0.5), 2),
                    "source_file": "demo",
                }
            )
            amenity_id += 1
    return pd.DataFrame(rows)


def build_crime_events(neighborhoods: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    event_id = 1
    today = datetime.today()
    for nb_meta, (_, nb_row) in zip(NEIGHBORHOODS, neighborhoods.iterrows()):
        # Lower safety -> more events.
        safety = nb_meta["safety"]
        n_events = int(round((100 - safety) * 1.2)) + random.randint(5, 15)
        for _ in range(n_events):
            offense, severity = random.choice(OFFENSE_TYPES)
            days_ago = random.randint(0, 365)
            rows.append(
                {
                    "event_id": event_id,
                    "neighborhood_id": int(nb_row["neighborhood_id"]),
                    "offense_type": offense,
                    "severity": severity,
                    "date": (today - timedelta(days=days_ago)).date().isoformat(),
                    "lat": _jitter(nb_meta["center_lat"], 0.5),
                    "lng": _jitter(nb_meta["center_lng"], 0.5),
                }
            )
            event_id += 1
    return pd.DataFrame(rows)


def build_parks(neighborhoods: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    park_id = 1
    for nb_meta, (_, nb_row) in zip(NEIGHBORHOODS, neighborhoods.iterrows()):
        n_parks = max(1, int(round(nb_meta["green"] / 25)))
        for _ in range(n_parks):
            rows.append(
                {
                    "park_id": f"PARK{park_id:04d}",
                    "neighborhood_id": int(nb_row["neighborhood_id"]),
                    "name": random.choice(PARK_NAMES),
                    "borough": nb_meta["borough"],
                    "address": f"{random.randint(1, 999)} {random.choice(STREETS)}",
                    "lat": _jitter(nb_meta["center_lat"], 0.4),
                    "lng": _jitter(nb_meta["center_lng"], 0.4),
                    "acres": round(random.uniform(0.5, 80), 1),
                    "park_type": random.choice(["Neighborhood Park", "Playground", "Garden", "Plaza"]),
                    "source_file": "demo",
                }
            )
            park_id += 1
    return pd.DataFrame(rows)


def build_facilities(neighborhoods: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    fac_id = 1
    for nb_meta, (_, nb_row) in zip(NEIGHBORHOODS, neighborhoods.iterrows()):
        nb_id = int(nb_row["neighborhood_id"])
        score = nb_meta["facilities"]
        n_each = max(1, int(round(score / 25)))
        for category, label in [
            ("school", "Department of Education"),
            ("healthcare", "Department of Health"),
            ("library", "NYC Public Library"),
            ("community_center", "Department of Youth"),
        ]:
            for _ in range(n_each):
                rows.append(
                    {
                        "facility_id": f"FAC{fac_id:05d}",
                        "neighborhood_id": nb_id,
                        "name": f"{nb_meta['name']} {category.replace('_', ' ').title()}",
                        "facility_type": category,
                        "category": category,
                        "borough": nb_meta["borough"],
                        "address": f"{random.randint(1, 999)} {random.choice(STREETS)}",
                        "lat": _jitter(nb_meta["center_lat"], 0.4),
                        "lng": _jitter(nb_meta["center_lng"], 0.4),
                        "agency": label,
                        "source_file": "demo",
                    }
                )
                fac_id += 1
    return pd.DataFrame(rows)


def build_hpd_buildings(neighborhoods: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    bid = 1
    for nb_meta, (_, nb_row) in zip(NEIGHBORHOODS, neighborhoods.iterrows()):
        nb_id = int(nb_row["neighborhood_id"])
        score = nb_meta["housing"]
        n_buildings = max(2, int(round(score / 8)))
        for _ in range(n_buildings):
            rows.append(
                {
                    "building_id": f"HPD{bid:06d}",
                    "neighborhood_id": nb_id,
                    "borough": nb_meta["borough"],
                    "address": f"{random.randint(50, 950)} {random.choice(STREETS)}",
                    "zip_code": str(random.randint(10001, 11697)),
                    "lat": _jitter(nb_meta["center_lat"], 0.5),
                    "lng": _jitter(nb_meta["center_lng"], 0.5),
                    "residential_units": random.randint(6, 90),
                    "program_type": random.choice(
                        ["Mitchell-Lama", "8A Loan", "PLP", "TPT", "HDFC Coop"]
                    ),
                    "status": random.choice(["Active", "Active", "Closed Out"]),
                    "source_file": "demo",
                }
            )
            bid += 1
    return pd.DataFrame(rows)


def build_transit(neighborhoods: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bus_rows: list[dict] = []
    sub_rows: list[dict] = []
    bus_id = 1
    sub_id = 1
    for nb_meta, (_, nb_row) in zip(NEIGHBORHOODS, neighborhoods.iterrows()):
        nb_id = int(nb_row["neighborhood_id"])
        transit = nb_meta["transit"]
        n_bus = max(3, int(round(transit / 10)))
        n_subway = max(1, int(round(transit / 20)))
        for _ in range(n_bus):
            bus_rows.append(
                {
                    "stop_id": f"BUS{bus_id:05d}",
                    "stop_name": f"{nb_meta['name']} Bus Stop",
                    "lat": _jitter(nb_meta["center_lat"], 0.4),
                    "lng": _jitter(nb_meta["center_lng"], 0.4),
                    "borough": nb_meta["borough"],
                    "route_count": random.randint(1, 5),
                    "route_record_count": random.randint(2, 12),
                    "neighborhood_id": nb_id,
                    "source_file": "demo",
                }
            )
            bus_id += 1
        for _ in range(n_subway):
            sub_rows.append(
                {
                    "station_id": f"SUB{sub_id:04d}",
                    "station_name": f"{nb_meta['name']} Station",
                    "line": random.choice(["N/W", "7", "L", "A/C", "1", "2/3", "F"]),
                    "routes": random.choice(["N,W", "7", "L", "A,C", "1", "2,3", "F"]),
                    "lat": _jitter(nb_meta["center_lat"], 0.3),
                    "lng": _jitter(nb_meta["center_lng"], 0.3),
                    "borough": nb_meta["borough"],
                    "neighborhood_id": nb_id,
                    "source_file": "demo",
                }
            )
            sub_id += 1
    return pd.DataFrame(bus_rows), pd.DataFrame(sub_rows)


def build_neighborhood_stats(
    neighborhoods: pd.DataFrame,
    apartments: pd.DataFrame,
    amenities: pd.DataFrame,
    crimes: pd.DataFrame,
    parks: pd.DataFrame,
    facilities: pd.DataFrame,
    hpd: pd.DataFrame,
    bus: pd.DataFrame,
    subway: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    apt_grp = apartments.groupby("neighborhood_id")
    amen_grp = amenities.groupby(["neighborhood_id", "type"]).size().unstack(fill_value=0)
    crime_grp = crimes.groupby("neighborhood_id").size()
    park_grp = parks.groupby("neighborhood_id")
    fac_grp = facilities.groupby(["neighborhood_id", "facility_type"]).size().unstack(fill_value=0)
    hpd_grp = hpd.groupby("neighborhood_id")
    bus_grp = bus.groupby("neighborhood_id").size()
    sub_grp = subway.groupby("neighborhood_id").size()

    for nb_meta, (_, nb_row) in zip(NEIGHBORHOODS, neighborhoods.iterrows()):
        nb_id = int(nb_row["neighborhood_id"])
        # Apartments
        if nb_id in apt_grp.groups:
            a = apt_grp.get_group(nb_id)
            listing_count = int(len(a))
            avg_rent = float(a["rent"].mean())
            median_rent = float(a["rent"].median())
            min_rent = int(a["rent"].min())
            max_rent = int(a["rent"].max())
            studio = int((a["bedrooms"] == 0).sum())
            one_bed = int((a["bedrooms"] == 1).sum())
            two_bed = int((a["bedrooms"] == 2).sum())
            avg_sqft = float(a["sqft"].mean())
        else:
            listing_count = avg_rent = median_rent = min_rent = max_rent = 0
            studio = one_bed = two_bed = 0
            avg_sqft = 0.0
        # Amenities
        amen_row = amen_grp.loc[nb_id] if nb_id in amen_grp.index else None
        worship_count = int(
            sum(
                amen_row.get(t, 0)
                for t in ("mosque", "church", "synagogue", "temple", "worship")
            )
            if amen_row is not None
            else 0
        )
        grocery_count = int(amen_row.get("grocery", 0) if amen_row is not None else 0)
        # Crimes
        crime_event_count = int(crime_grp.get(nb_id, 0))
        crime_per_1000 = (
            crime_event_count / nb_meta["population"] * 1000
            if nb_meta["population"]
            else 0.0
        )
        # Parks
        if nb_id in park_grp.groups:
            p = park_grp.get_group(nb_id)
            park_count = int(len(p))
            total_park_acres = float(p["acres"].sum())
        else:
            park_count = 0
            total_park_acres = 0.0
        nearest_park_distance = round(random.uniform(0.05, 0.5), 2) if park_count else 1.5
        # Facilities
        fac_row = fac_grp.loc[nb_id] if nb_id in fac_grp.index else None
        school_count = int(fac_row.get("school", 0) if fac_row is not None else 0)
        healthcare_count = int(fac_row.get("healthcare", 0) if fac_row is not None else 0)
        library_count = int(fac_row.get("library", 0) if fac_row is not None else 0)
        community_count = int(fac_row.get("community_center", 0) if fac_row is not None else 0)
        facility_count = school_count + healthcare_count + library_count + community_count
        # HPD
        if nb_id in hpd_grp.groups:
            h = hpd_grp.get_group(nb_id)
            hpd_building_count = int(len(h))
            hpd_unit_count = int(h["residential_units"].sum())
        else:
            hpd_building_count = 0
            hpd_unit_count = 0
        # Transit counts
        bus_stop_count = int(bus_grp.get(nb_id, 0))
        subway_station_count = int(sub_grp.get(nb_id, 0))

        # Score derivations from demo "ground truth"
        safety_score = float(nb_meta["safety"])
        crime_score = float(100 - safety_score)
        transit_score = float(nb_meta["transit"])
        green_space_score = float(nb_meta["green"])
        public_service_score = float(nb_meta["facilities"])
        housing_supply_score = float(nb_meta["housing"])
        affordable_housing_signal = float(
            min(100, housing_supply_score * 0.7 + 30 - max(0, (median_rent - 2200) / 30))
        )
        bus_access_score = max(40.0, min(100.0, transit_score - 5 + random.uniform(-5, 5)))
        subway_access_score = max(40.0, min(100.0, transit_score + random.uniform(-3, 5)))
        nearest_bus_stop_distance = round(max(0.05, 0.6 - transit_score / 200), 2)
        nearest_subway_distance = round(max(0.1, 0.9 - transit_score / 150), 2)

        # Affordability: lower median rent vs band -> higher score
        rent_band = nb_meta["rent_band"]
        affordability_score = 100 - _normalize(
            (median_rent or sum(rent_band) / 2),
            1500,
            4500,
        )
        amenity_score = (
            0.30 * transit_score
            + 0.25 * green_space_score
            + 0.20 * public_service_score
            + 0.15 * _normalize(worship_count, 0, 6)
            + 0.10 * _normalize(grocery_count, 0, 4)
        )
        availability_score = (
            0.60 * _normalize(listing_count, 0, 10)
            + 0.25 * housing_supply_score
            + 0.15 * affordable_housing_signal
        )
        renter_fit_score = (
            0.30 * affordability_score
            + 0.25 * safety_score
            + 0.25 * amenity_score
            + 0.20 * availability_score
        )
        value_score = (
            0.50 * affordability_score
            + 0.30 * safety_score
            + 0.20 * amenity_score
        )
        safety_affordability_score = 0.5 * affordability_score + 0.5 * safety_score

        rows.append(
            {
                "neighborhood_id": nb_id,
                "population": nb_meta["population"],
                "median_income": nb_meta["median_income"],
                "median_rent": int(median_rent or sum(rent_band) / 2),
                "percent_students": round(random.uniform(5, 18), 1),
                "percent_families": round(random.uniform(20, 55), 1),
                "hpd_building_count": hpd_building_count,
                "hpd_unit_count": hpd_unit_count,
                "housing_supply_score": round(housing_supply_score, 1),
                "affordable_housing_signal": round(affordable_housing_signal, 1),
                "bus_stop_count": bus_stop_count,
                "subway_station_count": subway_station_count,
                "nearest_bus_stop_distance": nearest_bus_stop_distance,
                "nearest_subway_distance": nearest_subway_distance,
                "bus_access_score": round(bus_access_score, 1),
                "subway_access_score": round(subway_access_score, 1),
                "transit_score": round(transit_score, 1),
                "park_count": park_count,
                "total_park_acres": round(total_park_acres, 1),
                "nearest_park_distance": nearest_park_distance,
                "green_space_score": round(green_space_score, 1),
                "facility_count": facility_count,
                "school_count": school_count,
                "healthcare_count": healthcare_count,
                "library_count": library_count,
                "community_resource_count": community_count,
                "public_service_score": round(public_service_score, 1),
                "worship_count": worship_count,
                "grocery_count": grocery_count,
                "listing_count": listing_count,
                "avg_rent": round(avg_rent, 1) if listing_count else 0.0,
                "median_listing_rent": round(median_rent, 1) if listing_count else 0.0,
                "min_rent": min_rent,
                "max_rent": max_rent,
                "studio_count": studio,
                "one_bed_count": one_bed,
                "two_bed_count": two_bed,
                "avg_sqft": round(avg_sqft, 1) if listing_count else 0.0,
                "crime_score": round(crime_score, 1),
                "safety_score": round(safety_score, 1),
                "affordability_score": round(affordability_score, 1),
                "amenity_score": round(amenity_score, 1),
                "availability_score": round(availability_score, 1),
                "renter_fit_score": round(renter_fit_score, 1),
                "value_score": round(value_score, 1),
                "safety_affordability_score": round(safety_affordability_score, 1),
            }
        )
    return pd.DataFrame(rows)


# --- Public entry point -------------------------------------------------------


def seed_demo_data(db_path: Optional[Path] = None) -> dict[str, int]:
    """Generate every synthetic table and load it into DuckDB.

    Returns a {table: row_count} dict for the seed report.
    """
    random.seed(RANDOM_SEED)
    ensure_directories()
    initialize_database(db_path)

    neighborhoods = build_neighborhoods()
    apartments = build_apartments(neighborhoods)
    amenities = build_amenities(neighborhoods)
    crimes = build_crime_events(neighborhoods)
    parks = build_parks(neighborhoods)
    facilities = build_facilities(neighborhoods)
    hpd = build_hpd_buildings(neighborhoods)
    bus, subway = build_transit(neighborhoods)
    stats = build_neighborhood_stats(
        neighborhoods, apartments, amenities, crimes, parks, facilities, hpd, bus, subway
    )

    # Persist sample CSVs alongside the DuckDB load so judges can inspect them.
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    neighborhoods.to_csv(SAMPLE_DIR / "neighborhoods.csv", index=False)
    stats.to_csv(SAMPLE_DIR / "neighborhood_stats.csv", index=False)
    apartments.to_csv(SAMPLE_DIR / "apartments.csv", index=False)
    amenities.to_csv(SAMPLE_DIR / "amenities.csv", index=False)
    crimes.to_csv(SAMPLE_DIR / "crime_events.csv", index=False)

    counts: dict[str, int] = {}
    counts["neighborhoods"] = replace_table("neighborhoods", neighborhoods, db_path)
    counts["neighborhood_stats"] = replace_table("neighborhood_stats", stats, db_path)
    counts["apartments"] = replace_table("apartments", apartments, db_path)
    counts["amenities"] = replace_table("amenities", amenities, db_path)
    counts["crime_events"] = replace_table("crime_events", crimes, db_path)
    counts["parks"] = replace_table("parks", parks, db_path)
    counts["facilities"] = replace_table("facilities", facilities, db_path)
    counts["hpd_buildings"] = replace_table("hpd_buildings", hpd, db_path)
    counts["transit_bus_stops"] = replace_table("transit_bus_stops", bus, db_path)
    counts["transit_subway_stations"] = replace_table("transit_subway_stations", subway, db_path)
    return counts


def main() -> None:
    print(f"Seeding demo data into {DB_PATH} ...")
    counts = seed_demo_data()
    for table, n in counts.items():
        print(f"  {table:28s} {n:6d} rows")
    print("Done.")


if __name__ == "__main__":
    main()
