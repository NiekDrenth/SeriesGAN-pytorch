#!/usr/bin/env python3.13
import wrds
import psycopg2
import pandas as pd

db = wrds.Connection(wrds_username='niekdrenth')

# Use psycopg2 because the wrds package doesnt work correctly
conn = psycopg2.connect(dbname='wrds', user='niekdrenth', host='wrds-pgdata.wharton.upenn.edu', port=9737)

# print(db.list_libraries().sort())
# print((db.list_libraries()))

# print("tables in taqmsec dataset:")
# print(db.list_tables(library='taqmsec'))
"""
        date,
        sym_root,
        best_bid,
        best_ask
"""
sql = """SELECT
    date,
    TO_CHAR(time_m, 'HH24:MI') AS time_m,
    best_bid,
    ROUND(hr_avg_prc, 2) AS hr_avg_pr,
    best_bid - ROUND(hr_avg_prc, 2) AS diff
FROM (
    SELECT
        *,
        RANK() OVER (
            PARTITION BY
                sym_root,
                date,
                EXTRACT(HOUR FROM time_m),
                FLOOR(EXTRACT(MINUTE FROM time_m) / 10)
            ORDER BY
                time_m,
                time_m_nano
        ) AS minute_rank,
        AVG(best_bid) OVER (
            PARTITION BY
                sym_root,
                date,
                EXTRACT(HOUR FROM time_m)
        ) AS hr_avg_prc
    FROM
        taqm_2022.complete_nbbo_2022
    WHERE
        sym_root = 'AAPL'
) a
WHERE
    minute_rank = 1
ORDER BY
    date;"""
conn.autocommit = True
cursor = conn.cursor()
cursor.execute(sql)

results = cursor.fetchall()


df = pd.DataFrame(results)
df.to_csv("output.csv", index=False)
conn.commit()
conn.close()
