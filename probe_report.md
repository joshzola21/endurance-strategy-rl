# Stage 00 scope probe

- file: `data/raw/laps.csv`
- race-session lap records: 982,565


## A. Is `session_id` a clean edition key?

```
 n_session_ids  spans_events  spans_years  spans_series  spans_session_types
          1013             0            0             0                    0
```

Anything other than zero in the `spans_` columns means `session_id` is not a per-session key and the scope key must be composite.

### Race sessions per (series, event, year) where there is more than one

```
series_code      event  year  race_sessions
       alms      Dubai  2022              2
       alms      Dubai  2023              2
       alms      Dubai  2025              2
       alms      Dubai  2026              2
       alms     Sepang  2024              2
       alms     Sepang  2025              2
       alms     Sepang  2026              2
       alms Yas Marina  2022              2
       alms Yas Marina  2023              2
       alms Yas Marina  2024              2
       alms Yas Marina  2025              2
       alms Yas Marina  2026              2
```

Empty means `(series_code, event, year)` identifies a race on its own and either key works. Non-empty means it does not, and `session_id` is the only safe scope - **or** those rows are the same race ingested twice, which section B decides.


## B. Duplicated ingestion

### Repeated (car, lap) inside one race session

```
 race_sessions  sessions_with_dupes  total_duplicate_car_lap_rows
           150                   59                         21799
```

### Identical lap records under more than one session id

```
series_code      event  year  shared_lap_records  sessions
       alms Yas Marina  2025                   4         2
       alms Yas Marina  2022                   4         2
       alms Yas Marina  2024                   2         2
       alms      Dubai  2022                   2         2
       alms      Dubai  2026                   2         2
       alms      Dubai  2023                   2         2
        wec    Bahrain  2025                   2         1
       imsa    Sebring  2025                   2         1
       alms Yas Marina  2023                   2         2
```

This is the section that decides whether scoping alone can fix the duration. If it is empty, it cannot, and the 216 hours has another cause.


## C. The editions, and the duration reconciliation

### imsa / Daytona

```
 session_id  year race_label  cars  max_lap  laps_recorded  elapsed_h  max_car_sum_h
        388  2021       None    48      807          33029      23.97          48.04
        467  2022       None    60      761          37597      23.97          48.02
        524  2023       None    59      783          39736      23.97          48.08
        579  2024       None    56      791          36628      23.97          47.97
        634  2025       None    58      781          37885      23.97          48.05
        682  2026       None    58      705          36756      23.97          48.09
```

Editions in file: **6**. Sum of per-edition max car totals: **288.3 h**.
`imsa.json` records 216.2 h and `wec.json` 120.4 h. If the sum above does not reach those, pooling is not the whole story and `calibrate_duration` is inflating on its own account.

`max_lap` against a single running - roughly 800 at Daytona, 380 at Le Mans - says whether `lap` resets per session.

### wec / Le Mans

```
 session_id  year race_label  cars  max_lap  laps_recorded  elapsed_h  max_car_sum_h
        766  2022       None    32      380          11514      23.93          24.09
        944  2025       None    60      387          20182      23.94          48.12
       1000  2026       None    60      381          20163      23.94          48.17
```

Editions in file: **3**. Sum of per-edition max car totals: **120.4 h**.
`imsa.json` records 216.2 h and `wec.json` 120.4 h. If the sum above does not reach those, pooling is not the whole story and `calibrate_duration` is inflating on its own account.

`max_lap` against a single running - roughly 800 at Daytona, 380 at Le Mans - says whether `lap` resets per session.


## I. Cars per class per edition

### imsa / Daytona

```
 year  DPi  GTD  GTDPRO  GTLM  GTP  LMP2  LMP3  TOTAL
 2021    7   19       0     6    0    10     7     49
 2022    7   22      13     0    0    10     9     61
 2023    0   23      10     0    9    10     9     61
 2024    0   23      13     0   10    13     0     59
 2025    0   21      15     0   12    12     0     60
 2026    0   21      15     0   11    13     0     60
```
### wec / Le Mans

```
 year  HYPERCAR  LMGT3  LMP2  TOTAL
 2022         5      0    27     32
 2025        20     24    17     61
 2026        17     25    19     61
```

This is the table the edition decision should be made on: a grid of three classes is a thinner demonstration than one of five, and recency trades against that.


## D. What `stint_number` counts


### imsa / Daytona 2025

```
 class  mean_distinct_stint_numbers  mean_pit_records  mean_distinct_drivers  stops_per_stint
   GTD                        11.38             25.29                   4.14             2.24
GTDPRO                        10.20             24.60                   3.67             2.35
   GTP                        10.92             25.42                   3.50             2.23
  LMP2                        12.25             34.25                   3.92             2.82
```

`stops_per_stint` near 1 means `stint_number` counts fuel stints, which is what `calibrate_stints` assumes. Near 3 means it counts driver stints, and the stint dial is wrong independently of scoping.

Where the counter steps:

```
 stint_steps  driver_changes  both  stint_steps_after_a_pit_lap
        2420            3559  2420                           86
```

`both` close to `stint_steps` means the counter follows the driver. `stint_steps_after_a_pit_lap` close to `stint_steps` means it follows the stop. Whichever it tracks is what the dial is actually measuring.


### imsa / Daytona 2026

```
 class  mean_distinct_stint_numbers  mean_pit_records  mean_distinct_drivers  stops_per_stint
   GTD                        11.33             21.67                   3.67             1.86
GTDPRO                        10.87             23.13                   3.47             2.06
   GTP                        11.45             29.55                   3.55             2.61
  LMP2                        12.62             30.38                   3.92             2.37
```

`stops_per_stint` near 1 means `stint_number` counts fuel stints, which is what `calibrate_stints` assumes. Near 3 means it counts driver stints, and the stint dial is wrong independently of scoping.

Where the counter steps:

```
 stint_steps  driver_changes  both  stint_steps_after_a_pit_lap
        2166            3231  2166                           68
```

`both` close to `stint_steps` means the counter follows the driver. `stint_steps_after_a_pit_lap` close to `stint_steps` means it follows the stop. Whichever it tracks is what the dial is actually measuring.


### wec / Le Mans 2025

```
   class  mean_distinct_stint_numbers  mean_pit_records  mean_distinct_drivers  stops_per_stint
HYPERCAR                        10.95             33.00                   3.15             3.07
   LMGT3                        10.25             28.33                   3.00             2.67
    LMP2                        11.65             32.59                   3.00             2.82
```

`stops_per_stint` near 1 means `stint_number` counts fuel stints, which is what `calibrate_stints` assumes. Near 3 means it counts driver stints, and the stint dial is wrong independently of scoping.

Where the counter steps:

```
 stint_steps  driver_changes  both  stint_steps_after_a_pit_lap
        1231            2063  1231                           70
```

`both` close to `stint_steps` means the counter follows the driver. `stint_steps_after_a_pit_lap` close to `stint_steps` means it follows the stop. Whichever it tracks is what the dial is actually measuring.


### wec / Le Mans 2026

```
   class  mean_distinct_stint_numbers  mean_pit_records  mean_distinct_drivers  stops_per_stint
HYPERCAR                        10.24             31.29                   3.18             3.03
   LMGT3                        10.12             28.36                   2.96             2.78
    LMP2                        11.74             31.58                   3.00             2.72
```

`stops_per_stint` near 1 means `stint_number` counts fuel stints, which is what `calibrate_stints` assumes. Near 3 means it counts driver stints, and the stint dial is wrong independently of scoping.

Where the counter steps:

```
 stint_steps  driver_changes  both  stint_steps_after_a_pit_lap
        1069            2023  1069                           50
```

`both` close to `stint_steps` means the counter follows the driver. `stint_steps_after_a_pit_lap` close to `stint_steps` means it follows the stop. Whichever it tracks is what the dial is actually measuring.


## E-H. One scoped race at a time


### imsa / Daytona 2025  (session_id 634, 37,885 lap records)

**E. Flag census.** `calibrate_cautions` treats everything that is not `GF` as a caution lap.

```
flag  laps  share  counted_as_caution_today
  GF 32799 0.8658                     False
 FCY  5046 0.1332                      True
  FF    40 0.0011                      True
```

**F. What `pit_time` measures.** Ratio of `pit_time` to the lap's excess over that car's median green lap.

```
 n_pit_laps  ratio_p25  ratio_median  ratio_p75
       1616      1.118         1.235      2.054
```

A median near 1.0 means `pit_time` is the whole time lost, lane to lane, and the low-quantile route to `pit_transit_frac` is coherent. Much below 1.0 means it is stationary time only, and a low quantile of it is not the transit delta.

**G. The pit column on green stops, per class.**

```
 class   n   mean     sd  sd_over_mean   p05   p50    p95     max
   GTD 338 131.91 313.53          2.38 68.53 91.92 124.52 3626.02
  LMP2 276 134.29 554.44          4.13 51.90 90.77 124.82 9136.01
GTDPRO 221 100.09 150.41          1.50 45.61 91.47 111.15 2312.67
   GTP 199  88.41  15.91          0.18 54.59 90.37  94.96  241.31
```

Gate condition two requires `sd` no greater than `mean`. `p05` is the candidate for `pit_transit_frac`, and it is only meaningful if F came back near 1.0.

**H. Degradation.** The frame `calibrate_pace` uses, then the same slope with a per-car-per-stint intercept removed.

```
 class  n_clean  slope_as_calibrated  slope_within_stint  slope_no_quartile_filter  max_tire_age
   GTD     5186              0.01476             0.01246                  -0.00260            28
GTDPRO     3705              0.00824             0.00771                   0.13235            28
   GTP     2965              0.00582             0.00877                   0.48791            27
  LMP2     3088              0.00301             0.00276                   0.06872            21
```

If `slope_as_calibrated` is negative and `slope_within_stint` is positive, the sign problem is fuel load leaking through the field-relative frame and the scoping fix will not touch it. If both are negative, condition four has a different cause again.


**Stint length on one scoped race**, grouped exactly as `calibrate_stints` groups it.

```
 class  n_stints  mean  q75  max
   GTD       234  47.9 60.0  127
GTDPRO       152  52.3 64.0   95
   GTP       131  50.8 58.0   88
  LMP2       147  47.5 56.5  116
```

**The observed classification** - what gate condition two compares against.

```
 class  car  laps  stops  green_laps_per_stop
   GTP   60   781     34                 23.0
  LMP2   74   765     42                 18.2
GTDPRO    3   723     29                 24.9
   GTD   83   719     29                 24.8
```

### imsa / Daytona 2026  (session_id 682, 36,756 lap records)

**E. Flag census.** `calibrate_cautions` treats everything that is not `GF` as a caution lap.

```
flag  laps  share  counted_as_caution_today
  GF 27625 0.7516                     False
 FCY  9082 0.2471                      True
  FF    49 0.0013                      True
```

**F. What `pit_time` measures.** Ratio of `pit_time` to the lap's excess over that car's median green lap.

```
 n_pit_laps  ratio_p25  ratio_median  ratio_p75
       1522      1.061         1.269      2.432
```

A median near 1.0 means `pit_time` is the whole time lost, lane to lane, and the low-quantile route to `pit_transit_frac` is coherent. Much below 1.0 means it is stationary time only, and a low quantile of it is not the transit delta.

**G. The pit column on green stops, per class.**

```
 class   n   mean     sd  sd_over_mean   p05   p50    p95     max
  LMP2 255 126.58 270.76          2.14 45.76 90.64 173.16 2710.23
   GTD 244 107.27 186.29          1.74 47.63 90.50 116.55 2276.99
GTDPRO 186 110.77 216.94          1.96 45.75 90.52 112.09 2883.26
   GTP 178 110.97 224.49          2.02 59.63 89.81  94.68 2814.11
```

Gate condition two requires `sd` no greater than `mean`. `p05` is the candidate for `pit_transit_frac`, and it is only meaningful if F came back near 1.0.

**H. Degradation.** The frame `calibrate_pace` uses, then the same slope with a per-car-per-stint intercept removed.

```
 class  n_clean  slope_as_calibrated  slope_within_stint  slope_no_quartile_filter  max_tire_age
   GTD     4086              0.00912             0.00960                   0.12327            33
GTDPRO     3208              0.00694             0.00663                   0.23756            29
   GTP     2695              0.00135             0.00109                   0.26465            28
  LMP2     2728              0.00379            -0.00122                   0.00680            23
```

If `slope_as_calibrated` is negative and `slope_within_stint` is positive, the sign problem is fuel load leaking through the field-relative frame and the scoping fix will not touch it. If both are negative, condition four has a different cause again.


**Stint length on one scoped race**, grouped exactly as `calibrate_stints` groups it.

```
 class  n_stints  mean  q75  max
   GTD       195  45.2 58.0  104
GTDPRO       140  49.0 61.0   99
   GTP       106  55.2 60.0   94
  LMP2       139  43.8 54.0   94
```

**The observed classification** - what gate condition two compares against.

```
 class  car  laps  stops  green_laps_per_stop
   GTP   60   705     37                 19.1
  LMP2    4   686     32                 21.4
GTDPRO    4   662     21                 31.5
   GTD   21   661     22                 30.0
```

### wec / Le Mans 2025  (session_id 944, 20,182 lap records)

**E. Flag census.** `calibrate_cautions` treats everything that is not `GF` as a caution lap.

```
flag  laps  share  counted_as_caution_today
  GF 19654 0.9738                     False
  SF   320 0.0159                      True
 FCY   159 0.0079                      True
  FF    49 0.0024                      True
```

**F. What `pit_time` measures.** Ratio of `pit_time` to the lap's excess over that car's median green lap.

```
 n_pit_laps  ratio_p25  ratio_median  ratio_p75
       1894      1.013         1.101      1.157
```

A median near 1.0 means `pit_time` is the whole time lost, lane to lane, and the low-quantile route to `pit_transit_frac` is coherent. Much below 1.0 means it is stationary time only, and a low quantile of it is not the transit delta.

**G. The pit column on green stops, per class.**

```
   class   n  mean     sd  sd_over_mean   p05   p50    p95     max
   LMGT3 657 97.35 186.46          1.92 58.25 78.79 100.24 4182.03
HYPERCAR 640 97.49 206.36          2.12 63.55 77.38  97.56 4369.91
    LMP2 538 90.42  68.80          0.76 54.49 84.73 105.82 1634.19
```

Gate condition two requires `sd` no greater than `mean`. `p05` is the candidate for `pit_transit_frac`, and it is only meaningful if F came back near 1.0.

**H. Degradation.** The frame `calibrate_pace` uses, then the same slope with a per-car-per-stint intercept removed.

```
   class  n_clean  slope_as_calibrated  slope_within_stint  slope_no_quartile_filter  max_tire_age
HYPERCAR     3227             -0.02526            -0.03444                  -0.07746            10
   LMGT3     2788              0.01428            -0.02339                   0.04427            38
    LMP2     2376             -0.02760             0.00901                  -0.07935             9
```

If `slope_as_calibrated` is negative and `slope_within_stint` is positive, the sign problem is fuel load leaking through the field-relative frame and the scoping fix will not touch it. If both are negative, condition four has a different cause again.


**Stint length on one scoped race**, grouped exactly as `calibrate_stints` groups it.

```
   class  n_stints  mean  q75  max
HYPERCAR       218  34.5 37.0   87
   LMGT3       241  27.0 33.0   52
    LMP2       198  28.5 33.0   47
```

**The observed classification** - what gate condition two compares against.

```
   class  car  laps  stops  green_laps_per_stop
HYPERCAR    6   387     31                 12.5
    LMP2   48   367     33                 11.1
   LMGT3   27   341     35                  9.7
```

### wec / Le Mans 2026  (session_id 1000, 20,163 lap records)

**E. Flag census.** `calibrate_cautions` treats everything that is not `GF` as a caution lap.

```
flag  laps  share  counted_as_caution_today
  GF 19179 0.9512                     False
  SF   786 0.0390                      True
 FCY   149 0.0074                      True
  FF    49 0.0024                      True
```

**F. What `pit_time` measures.** Ratio of `pit_time` to the lap's excess over that car's median green lap.

```
 n_pit_laps  ratio_p25  ratio_median  ratio_p75
       1841      1.034         1.106      1.155
```

A median near 1.0 means `pit_time` is the whole time lost, lane to lane, and the low-quantile route to `pit_transit_frac` is coherent. Much below 1.0 means it is stationary time only, and a low quantile of it is not the transit delta.

**G. The pit column on green stops, per class.**

```
   class   n   mean     sd  sd_over_mean   p05   p50    p95     max
   LMGT3 661  97.31 138.87          1.43 68.69 78.70 101.50 2271.68
    LMP2 569 104.47 174.97          1.67 81.75 88.59 118.07 3865.29
HYPERCAR 510 104.78 322.15          3.07 67.02 76.99  95.97 6855.81
```

Gate condition two requires `sd` no greater than `mean`. `p05` is the candidate for `pit_transit_frac`, and it is only meaningful if F came back near 1.0.

**H. Degradation.** The frame `calibrate_pace` uses, then the same slope with a per-car-per-stint intercept removed.

```
   class  n_clean  slope_as_calibrated  slope_within_stint  slope_no_quartile_filter  max_tire_age
HYPERCAR     2644             -0.03217            -0.03037                  -0.21241            10
   LMGT3     2936              0.01996             0.00023                   0.04172            21
    LMP2     2743             -0.03270            -0.04650                  -0.17693             9
```

If `slope_as_calibrated` is negative and `slope_within_stint` is positive, the sign problem is fuel load leaking through the field-relative frame and the scoping fix will not touch it. If both are negative, condition four has a different cause again.


**Stint length on one scoped race**, grouped exactly as `calibrate_stints` groups it.

```
   class  n_stints  mean  q75  max
HYPERCAR       174  34.7 38.0   77
   LMGT3       253  26.8 31.0   54
    LMP2       223  28.5 34.0   50
```

**The observed classification** - what gate condition two compares against.

```
   class  car  laps  stops  green_laps_per_stop
HYPERCAR    7   381     62                  6.1
    LMP2   43   361     31                 11.6
   LMGT3   33   336     32                 10.5
```

## J. Is `caution_mean_dur_s` reproducible?

```
series  call  caution_rate  mean_dur_s  episodes  reference_car
  imsa     1      0.210066  239.982707       668              4
  imsa     2      0.210066  239.982707       668              4
  imsa     3      0.210066  239.982707       668              4
  imsa     4      0.210066  239.982707       668              4
  imsa     5      0.210066  239.982707       668              4
   wec     1      0.048117  563.106324        37              7
   wec     2      0.048117  563.106324        37              7
   wec     3      0.048117  563.106324        37              7
   wec     4      0.048117  563.106324        37              7
   wec     5      0.048117  563.106324        37              7
```

Spread across five identical calls:

```
series     min     max   std
  imsa 239.983 239.983 0.000
   wec 563.106 563.106 0.000
```

Five identical calls returning five different episode lengths confirms the tie-ordering diagnosis, and explains the seven different values already sitting in `imsa.json`. A flat column means `imsa.json` was built by a different `calibrate.py` than the one in the tree, which is worth knowing before anything else.
