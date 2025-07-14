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
# print(db.list_tables(library='taqm_2024'))
# sql = """SELECT
#     date,
#     TO_CHAR(time_m, 'HH24:MI') AS time_m,
#     best_bid,
#     ROUND(hr_avg_prc, 2) AS hr_avg_pr,
#     best_bid - ROUND(hr_avg_prc, 2) AS diff
# FROM (
#     SELECT
#         *,
#         RANK() OVER (
#             PARTITION BY
#                 sym_root,
#                 date,
#                 EXTRACT(HOUR FROM time_m),
#                 FLOOR(EXTRACT(MINUTE FROM time_m) / 10)
#             ORDER BY
#                 time_m,
#                 time_m_nano
#         ) AS minute_rank,
#         AVG(best_bid) OVER (
#             PARTITION BY
#                 sym_root,
#                 date,
#                 EXTRACT(HOUR FROM time_m)
#         ) AS hr_avg_prc
#     FROM
#         taqm_2022.complete_nbbo_2022
#     WHERE
#         sym_root = 'AAPL'
# ) a
# WHERE
#     minute_rank = 1
# ORDER BY
#     date;"""

# query = """
#     SELECT *
#     FROM taqm_2022.complete_nbbo_2022
#     LIMIT 0;
# """

# df = pd.read_sql(query, conn)

# print(df.columns.tolist())


sql = """WITH base_trades AS (
    SELECT
        sym_root,
        date,
        time_m,
        time_m_nano,
        price,
        size,
        LAST_VALUE(price) OVER (
            PARTITION BY sym_root, date,
                         EXTRACT(HOUR FROM time_m),
                         FLOOR(EXTRACT(MINUTE FROM time_m) / 10)
            ORDER BY time_m, time_m_nano
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS last_price,
        EXTRACT(HOUR FROM time_m) AS hour,
        FLOOR(EXTRACT(MINUTE FROM time_m) / 10) AS ten_min_block
    FROM taqm_2022.ctm_2022
    WHERE sym_root = 'AAPL'
      AND time_m::time >= TIME '09:30:00'
      AND time_m::time <= TIME '16:00:00'
),

trade_blocks AS (
    SELECT
        sym_root,
        date,
        hour,
        ten_min_block,
        MIN(price) AS low_price,
        MAX(price) AS high_price,
        SUM(size) AS total_volume,
        MAX(last_price) AS last_price
    FROM base_trades
    GROUP BY
        sym_root,
        date,
        hour,
        ten_min_block
),

nbbo_blocks AS (
    SELECT
        sym_root,
        date,
        hour,
        ten_min_block,
        best_bid,
        best_ask
    FROM (
        SELECT
            sym_root,
            date,
            time_m,
            time_m_nano,
            best_bid,
            best_ask,
            EXTRACT(HOUR FROM time_m) AS hour,
            FLOOR(EXTRACT(MINUTE FROM time_m) / 10) AS ten_min_block,
            RANK() OVER (
                PARTITION BY sym_root, date,
                             EXTRACT(HOUR FROM time_m),
                             FLOOR(EXTRACT(MINUTE FROM time_m) / 10)
                ORDER BY time_m DESC, time_m_nano DESC
            ) AS quote_rank
        FROM taqm_2022.complete_nbbo_2022
        WHERE sym_root = 'AAPL'
          AND time_m::time >= TIME '09:30:00'
          AND time_m::time <= TIME '16:00:00'
    ) sub
    WHERE quote_rank = 1
)

SELECT
    t.date,
    t.hour,
    t.ten_min_block,
    t.low_price,
    t.high_price,
    t.total_volume,
    t.last_price,
    n.best_bid,
    n.best_ask
FROM trade_blocks t
LEFT JOIN nbbo_blocks n
    ON t.sym_root = n.sym_root
   AND t.date = n.date
   AND t.hour = n.hour
   AND t.ten_min_block = n.ten_min_block
ORDER BY
    t.date,
    t.hour,
    t.ten_min_block;

"""
conn.autocommit = True
cursor = conn.cursor()
cursor.execute(sql)

results = cursor.fetchall()


df = pd.DataFrame(results)
df.to_csv("output.csv", index=False)
conn.commit()
conn.close()
