"""Smoke test the SQL safety validator."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.safety_agent import safety_agent


AMENITY_PROXIMITY_SQL = """
SELECT n.name AS neighborhood, n.borough, a.address, a.rent
FROM apartments a
JOIN neighborhoods n ON a.neighborhood_id = n.neighborhood_id
JOIN neighborhood_stats ns ON n.neighborhood_id = ns.neighborhood_id
WHERE a.rent <= 3000
  AND EXISTS (
    SELECT 1 FROM amenities am
    WHERE am.neighborhood_id = n.neighborhood_id
      AND am.type = 'mosque'
      AND am.distance_miles <= 0.5
  )
ORDER BY ns.renter_fit_score DESC
LIMIT 50
""".strip()


TRANSIT_HAVERSINE_SQL = """
SELECT n.name AS neighborhood, a.address, a.rent,
       (SELECT MIN(2 * 3958.8 * ASIN(SQRT(
           POWER(SIN(RADIANS((s.lat - a.lat) / 2)), 2) +
           COS(RADIANS(a.lat)) * COS(RADIANS(s.lat)) *
           POWER(SIN(RADIANS((s.lng - a.lng) / 2)), 2))))
        FROM transit_subway_stations s
        WHERE s.lat IS NOT NULL AND s.lng IS NOT NULL) AS distance_to_subway_miles
FROM apartments a
JOIN neighborhoods n ON a.neighborhood_id = n.neighborhood_id
JOIN neighborhood_stats ns ON n.neighborhood_id = ns.neighborhood_id
WHERE a.rent <= 2800
  AND (SELECT MIN(2 * 3958.8 * ASIN(SQRT(
           POWER(SIN(RADIANS((s.lat - a.lat) / 2)), 2) +
           COS(RADIANS(a.lat)) * COS(RADIANS(s.lat)) *
           POWER(SIN(RADIANS((s.lng - a.lng) / 2)), 2))))
        FROM transit_subway_stations s
        WHERE s.lat IS NOT NULL AND s.lng IS NOT NULL) <= 0.25
ORDER BY ns.renter_fit_score DESC
LIMIT 50
""".strip()


TESTS = [
    ("SELECT * FROM neighborhoods LIMIT 5", True),
    ("select n.* from neighborhoods n", True),
    ("WITH x AS (SELECT 1 AS v) SELECT v FROM x", True),
    (AMENITY_PROXIMITY_SQL, True),
    (TRANSIT_HAVERSINE_SQL, True),
    ("DROP TABLE neighborhoods", False),
    ("SELECT * FROM users", False),
    ("SELECT * FROM neighborhoods; DELETE FROM apartments", False),
    ("INSERT INTO apartments VALUES (1)", False),
    ("SELECT * FROM apartments; ATTACH 'evil.db' AS e", False),
    ("UPDATE apartments SET rent = 0", False),
    ("PRAGMA database_list", False),
]


def main() -> int:
    ok = 0
    for sql, expected_approved in TESTS:
        res = safety_agent(sql)
        actual = bool(res["approved"])
        flag = "OK" if actual == expected_approved else "FAIL"
        if actual == expected_approved:
            ok += 1
        print(f"  [{flag}] expected={expected_approved} actual={actual} reason={res['reason']!r:40s} | {sql[:60]}")
    print(f"\n{ok}/{len(TESTS)} safety tests passed")
    return 0 if ok == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
