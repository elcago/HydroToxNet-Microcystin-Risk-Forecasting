from pathlib import Path
import numpy as np
import pandas as pd

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

ROOT = Path('/content/drive/MyDrive/hab_stgnn')
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'lake_erie_multimodal_master_2012_2022.csv'

LAKE_PATH = RAW / 'lake_monitoring.csv'
RIVER_PATH = RAW / 'maumee_river.csv'
WIND_PATH = RAW / 'ndbc_45005.csv'

START_DATE = '2012-01-01'
END_DATE = '2022-12-31'

LAKE_ALIASES = {
    'date': ['date', 'Date', 'sample_date', 'Sample Date'],
    'station': ['station', 'Station', 'station_id', 'Station ID'],
    'latitude': ['latitude', 'Latitude', 'lat'],
    'longitude': ['longitude', 'Longitude', 'lon', 'lng'],
    'particulate_microcystin': ['particulate_microcystin', 'Particulate Microcystin'],
    'water_temperature': ['water_temperature', 'Sample Temperature (°C)', 'temperature'],
    'turbidity': ['turbidity', 'Turbidity'],
    'secchi_depth': ['secchi_depth', 'Secchi Depth (m)', 'secchi'],
    'ctd_dissolved_oxygen': ['ctd_dissolved_oxygen', 'CTD Dissolved Oxygen (mg/L)'],
    'ctd_specific_conductivity': ['ctd_specific_conductivity', 'CTD Specific Conductivity (µS/cm)'],
    'chlorophyll_a': ['chlorophyll_a', 'Chlorophyll a'],
    'phycocyanin': ['phycocyanin', 'Phycocyanin'],
    'ammonia': ['ammonia', 'Ammonia'],
    'nitrate_nitrite': ['nitrate_nitrite', 'Nitrate + Nitrite', 'nitrate_plus_nitrite'],
    'total_phosphorus': ['total_phosphorus', 'Total Phosphorus'],
    'soluble_reactive_phosphorus': ['soluble_reactive_phosphorus', 'Soluble Reactive Phosphorus'],
}

RIVER_ALIASES = {
    'date': ['date', 'Date', 'datetime'],
    'river_discharge_cfs': ['river_discharge_cfs', 'discharge_cfs', 'discharge'],
    'river_nitrate': ['river_nitrate', 'nitrate_nitrite', 'nitrate'],
    'river_srp': ['river_srp', 'soluble_reactive_phosphorus', 'srp'],
}

WIND_ALIASES = {
    'date': ['date', 'Date', 'datetime', 'timestamp'],
    'wind_speed_ms': ['wind_speed_ms', 'wind_speed', 'WSPD'],
    'wind_direction_deg': ['wind_direction_deg', 'wind_direction', 'WDIR'],
}


def normalize_station(value):
    text = str(value).strip().upper()
    digits = ''.join(ch for ch in text if ch.isdigit())
    return f'WE{int(digits):02d}' if digits else text


def clean_numeric(series):
    text = series.astype(str)
    text = text.str.replace('<', '', regex=False)
    text = text.str.replace('>', '', regex=False)
    text = text.str.split('±').str[0]
    return pd.to_numeric(text, errors='coerce')


def rename_aliases(df, aliases):
    rename = {}
    for canonical, choices in aliases.items():
        for name in choices:
            if name in df.columns:
                rename[name] = canonical
                break
    return df.rename(columns=rename)


def require_columns(df, columns, name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f'{name} is missing columns: {missing}')


def load_lake():
    lake = pd.read_csv(LAKE_PATH, low_memory=False)
    lake = rename_aliases(lake, LAKE_ALIASES)
    require_columns(lake, ['date', 'station'], 'lake data')

    lake['date'] = pd.to_datetime(lake['date'], errors='coerce')
    lake['station'] = lake['station'].map(normalize_station)
    lake = lake.dropna(subset=['date', 'station'])
    lake = lake[lake['date'].between(START_DATE, END_DATE)]

    for col in LAKE_ALIASES:
        if col in {'date', 'station'} or col not in lake.columns:
            continue
        lake[col] = clean_numeric(lake[col])

    keep = [c for c in LAKE_ALIASES if c in lake.columns]
    return lake[keep].copy()


def load_river():
    river = pd.read_csv(RIVER_PATH, low_memory=False)
    river = rename_aliases(river, RIVER_ALIASES)
    require_columns(river, ['date'], 'river data')

    river['date'] = pd.to_datetime(river['date'], errors='coerce')
    river = river.dropna(subset=['date'])
    river = river[river['date'].between(START_DATE, END_DATE)]

    for col in ['river_discharge_cfs', 'river_nitrate', 'river_srp']:
        if col in river.columns:
            river[col] = clean_numeric(river[col])

    numeric = [c for c in ['river_discharge_cfs', 'river_nitrate', 'river_srp'] if c in river.columns]
    iso = river['date'].dt.isocalendar()
    river['iso_year'] = iso.year.astype(int)
    river['iso_week'] = iso.week.astype(int)
    return river.groupby(['iso_year', 'iso_week'], as_index=False)[numeric].mean()


def load_wind():
    wind = pd.read_csv(WIND_PATH, low_memory=False)
    wind = rename_aliases(wind, WIND_ALIASES)
    require_columns(wind, ['date', 'wind_speed_ms', 'wind_direction_deg'], 'wind data')

    wind['date'] = pd.to_datetime(wind['date'], errors='coerce')
    wind = wind.dropna(subset=['date'])
    wind = wind[wind['date'].between(START_DATE, END_DATE)]
    wind['wind_speed_ms'] = clean_numeric(wind['wind_speed_ms'])
    wind['wind_direction_deg'] = clean_numeric(wind['wind_direction_deg'])

    angle = np.deg2rad(wind['wind_direction_deg'])
    wind['wind_u_from'] = wind['wind_speed_ms'] * np.sin(angle)
    wind['wind_v_from'] = wind['wind_speed_ms'] * np.cos(angle)
    iso = wind['date'].dt.isocalendar()
    wind['iso_year'] = iso.year.astype(int)
    wind['iso_week'] = iso.week.astype(int)

    weekly = wind.groupby(['iso_year', 'iso_week'], as_index=False).agg(
        wind_speed_ms=('wind_speed_ms', 'mean'),
        wind_u_from=('wind_u_from', 'mean'),
        wind_v_from=('wind_v_from', 'mean'),
    )

    weekly['wind_direction_deg'] = (
        np.degrees(np.arctan2(weekly['wind_u_from'], weekly['wind_v_from'])) + 360.0
    ) % 360.0

    return weekly.drop(columns=['wind_u_from', 'wind_v_from'])


def main():
    lake = load_lake()
    river = load_river()
    wind = load_wind()

    iso = lake['date'].dt.isocalendar()
    lake['iso_year'] = iso.year.astype(int)
    lake['iso_week'] = iso.week.astype(int)

    master = lake.merge(
        river,
        on=['iso_year', 'iso_week'],
        how='left',
    )
    master = master.merge(
        wind,
        on=['iso_year', 'iso_week'],
        how='left',
    )

    master = master.sort_values(['station', 'date']).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT, index=False)

    print('Rows:', len(master))
    print('Stations:', master['station'].nunique())
    print('Start:', master['date'].min())
    print('End:', master['date'].max())
    print('Saved:', OUT)


if __name__ == '__main__':
    main()
