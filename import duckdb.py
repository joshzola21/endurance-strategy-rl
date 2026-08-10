import duckdb

LAPS = "data/raw/laps.csv"
SESSIONS = (634, 682, 944, 1000)   # Daytona 25/26, Le Mans 25/26

con = duckdb.connect()
src = (f"read_csv_auto('{LAPS}', quote='\"', escape='\"', "
       "types={'car': 'VARCHAR'})")

# 1. Do leading zeros survive in the file at all?
print(con.execute(f"""
    SELECT COUNT(*) AS rows_with_leading_zero,
           COUNT(DISTINCT car) AS distinct_such_numbers
    FROM {src} WHERE car LIKE '0%'
""").df())

# 2. Which numbers collide once parsed as an integer, in the target races?
print(con.execute(f"""
    SELECT session_id,
           CAST(car AS BIGINT) AS collapses_to,
           STRING_AGG(DISTINCT car, ' + ') AS raw_values,
           STRING_AGG(DISTINCT class, ' + ') AS classes
    FROM {src}
    WHERE session = 'race' AND session_id IN {SESSIONS}
    GROUP BY session_id, collapses_to
    HAVING COUNT(DISTINCT car) > 1
    ORDER BY session_id, collapses_to
""").df())