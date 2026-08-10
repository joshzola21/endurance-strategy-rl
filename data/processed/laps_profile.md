# `laps.csv` profile

- file: `/Users/joshzola/Documents/motorsport/endurance racing/endurance strategy rl/data/raw/laps.csv`
- size: 582 MB
- rows: 1,658,803

## Schema

| column | type |
|---|---|
| `series_code` | VARCHAR |
| `series` | VARCHAR |
| `start_date` | TIMESTAMP |
| `year` | BIGINT |
| `event` | VARCHAR |
| `race_label` | VARCHAR |
| `session` | VARCHAR |
| `session_id` | BIGINT |
| `session_time` | DOUBLE |
| `clock_time` | DOUBLE |
| `session_time_lap_number` | BIGINT |
| `car` | BIGINT |
| `class` | VARCHAR |
| `driver_name` | VARCHAR |
| `driver_id` | VARCHAR |
| `lap` | BIGINT |
| `lap_time` | DOUBLE |
| `lap_time_s1` | DOUBLE |
| `lap_time_s2` | DOUBLE |
| `lap_time_s3` | DOUBLE |
| `s1_meters` | DOUBLE |
| `s2_meters` | DOUBLE |
| `s3_meters` | DOUBLE |
| `has_microsectors` | BOOLEAN |
| `microsectors_json` | VARCHAR |
| `lap_time_driver_rank` | BIGINT |
| `lap_time_driver_quartile` | BIGINT |
| `bpillar_quartile` | BIGINT |
| `pit_time` | DOUBLE |
| `flags` | VARCHAR |
| `stint_start` | BIGINT |
| `stint_number` | BIGINT |
| `stint_lap` | BIGINT |
| `license` | VARCHAR |
| `license_rank` | BIGINT |
| `driver_country` | VARCHAR |
| `team_name` | VARCHAR |
| `chassis` | VARCHAR |
| `homologation` | VARCHAR |
| `manufacturer` | VARCHAR |
| `air_temp_f` | VARCHAR |
| `track_temp_f` | VARCHAR |
| `humidity_percent` | VARCHAR |
| `pressure_inhg` | VARCHAR |
| `wind_speed_mph` | VARCHAR |
| `wind_direction_degrees` | VARCHAR |
| `raining` | VARCHAR |
| `est_tire_age` | BIGINT |
| `class_normalized` | VARCHAR |
| `class_category` | VARCHAR |

## Is there an edition discriminator?

Columns whose name suggests one: `start_date`, `year`, `session_id`, `stint_start`

- `start_date`: 1012 distinct, range Timestamp('2021-01-28 11:05:00') to Timestamp('2026-07-12 14:05:00')
- `year`: 6 distinct, range np.int64(2021) to np.int64(2026)
- `session_id`: 1013 distinct, range np.int64(1) to np.int64(1013)
- `stint_start`: 2 distinct, range np.int64(0) to np.int64(1)

Event strings containing a four-digit year: **0 of 38** sampled

## Values

### `series_code` — 4 distinct

| value | rows |
|---|---|
| `imsa` | 804,453 |
| `wec` | 393,391 |
| `elms` | 291,868 |
| `alms` | 169,091 |

### `event` — 38 distinct

| value | rows |
|---|---|
| `Daytona` | 264,428 |
| `Sebring` | 143,177 |
| `Road Atlanta` | 120,056 |
| `Le Mans` | 100,664 |
| `Imola` | 77,933 |
| `Portimao` | 77,321 |
| `Spa` | 75,503 |
| `Yas Marina` | 66,806 |
| `Dubai` | 62,192 |
| `Watkins Glen` | 61,281 |
| `Paul Ricard` | 47,241 |
| `Bahrain` | 44,055 |
| `Indianapolis` | 42,372 |
| `Barcelona` | 41,397 |
| `Laguna Seca` | 40,228 |
| `Sepang` | 40,093 |
| `Fuji` | 36,621 |
| `Losail` | 36,097 |
| `Monza` | 31,049 |
| `Mosport` | 27,449 |
| `Road America` | 26,294 |
| `Interlagos` | 25,737 |
| `Long Beach` | 24,154 |
| `COTA` | 16,782 |
| `VIR` | 15,825 |
| `Detroit` | 13,962 |
| `Lime Rock` | 12,734 |
| `Spielberg` | 12,503 |
| `Mugello` | 12,342 |
| `Mid-Ohio` | 12,081 |
| `Silverstone` | 11,426 |
| `Watkins Glen 6 Hours` | 9,109 |
| `Aragon` | 8,341 |
| `Canadian Tire Motorsport Park` | 7,045 |
| `Bahrain 8 Hours` | 5,456 |
| `Bahrain 6 Hours` | 3,842 |
| `Belle Isle` | 2,797 |
| `Watkins Glen 240` | 2,410 |

### `session` — 5 distinct

| value | rows |
|---|---|
| `race` | 982,565 |
| `practice` | 462,651 |
| `test` | 160,595 |
| `qualifying` | 42,350 |
| `warmup` | 10,642 |

### `class` — 11 distinct

| value | rows |
|---|---|
| `LMP2` | 405,981 |
| `GTD` | 290,705 |
| `LMP3` | 184,152 |
| `HYPERCAR` | 169,366 |
| `LMGT3` | 154,791 |
| `GTDPRO` | 144,511 |
| `GTP` | 122,330 |
| `GT` | 87,520 |
| `LMP2 Pro/Am` | 45,059 |
| `DPi` | 40,610 |
| `GTLM` | 13,778 |

### `flags` — 5 distinct

| value | rows |
|---|---|
| `GF` | 1,436,199 |
| `FCY` | 98,002 |
| `nan` | 69,472 |
| `SF` | 23,641 |
| `FF` | 17,762 |
| `RF` | 13,727 |

## What the current scoping selects

`build_race_config` scopes with an ILIKE pattern on `event`. This is what those patterns pick up.

| pattern | rows | distinct cars | distinct events |
|---|---|---|---|
| `%daytona%` | 221,631 | 91 | 1 |
| `%le mans%` | 51,859 | 84 | 1 |
| `%sebring%` | 100,172 | 83 | 1 |
| `%spa%` | 35,287 | 79 | 1 |

A single Daytona 24 grid is roughly 60 cars and a single Le Mans roughly 62. Substantially more than that is the pooling.

## Does lap and stint numbering reset per event?

Per event, across its cars, the largest values seen (fifteen highest):

```
               event  cars  max_lap  max_stint_number  max_stint_lap
             Daytona    91      807                73            131
        Road Atlanta    75      443                10            154
             Le Mans    95      387                17             60
             Sebring    83      353                13             97
              Losail    42      318                 9            110
            Portimao    74      300                 9             91
             Bahrain    51      249                 9             88
     Bahrain 8 Hours    14      247                 8             94
        Indianapolis    74      243                 8            137
          Interlagos    42      242                 8             91
                Fuji    49      232                 7            132
               Imola    75      213                 9            115
               Monza    57      204                 8             83
        Watkins Glen    80      201                 7            120
Watkins Glen 6 Hours    37      200                 7             75
```

A 24-hour race is roughly 800 laps at Daytona and 380 at Le Mans. A max lap far above that means the numbering runs across editions, which is what produced 199-lap 'green stints'.

## The pit column

```
     n   mean     sd  lo   p05    p50   p95       hi
151797 248.07 472.09 1.0 22.57 101.76 878.5 14944.22
```

The calibrated dials report a standard deviation two to five times the mean. If that shows here too, the column is capturing something other than service time — red flags, penalties, or non-race sessions. The 5th percentile is the number that would become `pit_transit_frac` if it turns out to be a stop with no service in it.

## Sample rows

```
series_code    series          start_date  year event race_label  session  session_id  session_time  clock_time  session_time_lap_number  car class driver_name       driver_id  lap  lap_time  lap_time_s1  lap_time_s2  lap_time_s3  s1_meters  s2_meters  s3_meters  has_microsectors microsectors_json  lap_time_driver_rank  lap_time_driver_quartile  bpillar_quartile  pit_time flags  stint_start  stint_number  stint_lap license  license_rank driver_country          team_name           chassis homologation manufacturer air_temp_f track_temp_f humidity_percent pressure_inhg wind_speed_mph wind_direction_degrees raining  est_tire_age class_normalized class_category
       alms alms-2022 2022-02-11 13:50:00  2022 Dubai       None practice           1       161.716   49961.716                        1   12    GT  Ben Barker benjamin barker    1   161.716       78.997       47.379       35.340        NaN        NaN        NaN             False              None                     9                         3              <NA>    38.468    GF            1             1          0    Gold             4            GBR Dinamic Motorsport Porsche 911 GT3 R          GT3      Porsche       None         None             None          None           None                   None    None          <NA>              GT3            GT3
       alms alms-2022 2022-02-11 13:50:00  2022 Dubai       None practice           1       279.741   50079.741                        2   12    GT  Ben Barker benjamin barker    2   118.025       39.266       46.499       32.260        NaN        NaN        NaN             False              None                     1                         1              <NA>       NaN    GF            0             1          1    Gold             4            GBR Dinamic Motorsport Porsche 911 GT3 R          GT3      Porsche       None         None             None          None           None                   None    None          <NA>              GT3            GT3
       alms alms-2022 2022-02-11 13:50:00  2022 Dubai       None practice           1       398.309   50198.309                        3   12    GT  Ben Barker benjamin barker    3   118.568       39.479       46.558       32.531        NaN        NaN        NaN             False              None                     2                         1              <NA>       NaN    GF            0             1          2    Gold             4            GBR Dinamic Motorsport Porsche 911 GT3 R          GT3      Porsche       None         None             None          None           None                   None    None          <NA>              GT3            GT3
       alms alms-2022 2022-02-11 13:50:00  2022 Dubai       None practice           1       521.161   50321.161                        4   12    GT  Ben Barker benjamin barker    4   122.852       39.622       46.882       36.348        NaN        NaN        NaN             False              None                     5                         2              <NA>       NaN    GF            0             1          3    Gold             4            GBR Dinamic Motorsport Porsche 911 GT3 R          GT3      Porsche       None         None             None          None           None                   None    None          <NA>              GT3            GT3
       alms alms-2022 2022-02-11 13:50:00  2022 Dubai       None practice           1      1001.183   50801.183                        7   12    GT  Ben Barker benjamin barker    5   480.022      400.472       46.937       32.613        NaN        NaN        NaN             False              None                    11                         4              <NA>   362.776    GF            0             1          4    Gold             4            GBR Dinamic Motorsport Porsche 911 GT3 R          GT3      Porsche       None         None             None          None           None                   None    None          <NA>              GT3            GT3
```