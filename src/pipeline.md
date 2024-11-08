# Energy Data Processing Pipeline

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
  - [Initialization](#initialization)
  - [Data Retrieval](#data-retrieval)
  - [Consumption Metrics](#consumption-metrics)
- [Data Structure](#data-structure)
  - [Input Files](#input-files)
  - [Processed Data](#processed-data)
- [Classes and Enums](#classes-and-enums)
  - [Pipeline](#pipeline)
- [Example](#example)

## Overview

The **Energy Data Processing Pipeline** is a Python-based tool designed to ingest, process, and analyze energy, solar, and meteorological data for various buildings. It consolidates data from multiple sources, performs necessary transformations, and provides aggregated insights on energy consumption and generation.

## Features

- **Data Ingestion**: Reads energy, solar, and meteorological data from CSV files.
- **Data Cleaning & Transformation**: Renames columns, handles missing values, and interpolates data.
- **Data Merging**: Combines energy, solar, and meteorological datasets for comprehensive analysis.
- **Aggregation**: Provides daily, weekly, and yearly consumption metrics.
- **Flexibility**: Easily extendable to accommodate additional buildings or data sources.
- **Visualization Support**: Integrated with Matplotlib for potential data visualization (commented out in the code).

## Usage

### Initialization

Initialize the `Pipeline` class to process the data.

```python
from pipeline_module import Pipeline, BuilingIdsEnum  # Adjust the import as necessary

# Initialize the pipeline
pipeline = Pipeline()
```

**Parameters:**

- `energy_path` (str): Path to the energy import/export CSV file.
- `solar_path` (str): Path to the solar self-consumption CSV file.
- `met_path` (str): Path to the meteorological data CSV file.

### Data Retrieval

Retrieve processed data for a specific building.

```python
# Get main building data
main_building_data = pipeline.get_data(BuilingIdsEnum.MAIN)

# Get Building A data
building_a_data = pipeline.get_data(BuilingIdsEnum.A)
```

### Consumption Metrics

Calculate net energy consumption over different time frequencies.

```python
# Daily consumption for main building
daily_consumption = pipeline.get_daily_consumption(BuilingIdsEnum.MAIN)
print(daily_consumption.head())

# Weekly consumption for Building B
weekly_consumption = pipeline.get_weekly_consumption(BuilingIdsEnum.B)
print(weekly_consumption.head())

# Yearly consumption for Building C
yearly_consumption = pipeline.get_yearly_consumption(BuilingIdsEnum.C)
print(yearly_consumption.head())
```

## Data Structure

### Input Files

1. **Energy Data (`energy_import_export.csv`)**

   - **Columns:**
     - `Energikilde`: Energy source
     - `Retning`: Direction (`IMPORT` or `EXPORT`)
     - `Målernavn`: Meter name
     - `Måler-Id`: Meter ID
     - `Verdi`: Value (energy amount)
     - `Tidspunkt`: Timestamp

2. **Solar Data (`solar_self_consumption_main_building.csv`)**

   - **Columns:**
     - `Unnamed: 0`: Index (will be dropped)
     - `starting_at`: Timestamp
     - `solar_consumption`: Solar energy consumption value

3. **Meteorological Data (`met_data.csv`)**

   - **Columns:**
     - `starting_at`: Timestamp
     - `property_id`: Building ID
     - `temperature`: Temperature value
     - `wind_speed`: Wind speed
     - `cloud_fraction`: Cloud coverage
     - `precipitation`: Precipitation amount
     - `wind_direction`: Wind direction in degrees

### Processed Data

For each building, the processed DataFrame includes:

- `timestamp`: Date and time of the record
- `value_import`: Imported energy value
- `value_export`: Exported energy value (only for the main building)
- `solar_consumption`: Solar energy consumption (only for the main building)
- `net_consumption`: Calculated net energy consumption
- `building`: Building name
- **Meteorological Columns:**
  - `temperature`
  - `wind_speed`
  - `cloud_fraction`
  - `precipitation`
  - `wind_direction`

## Classes and Enums

### BuilingIdsEnum

An enumeration for building identifiers.

```python
from enum import Enum

class BuilingIdsEnum(Enum):
    MAIN = 10724
    A = 10703
    B = 4462
    C = 4746
```

**Members:**

- `MAIN`: Main building (ID: 10724)
- `A`: Building A (ID: 10703)
- `B`: Building B (ID: 4462)
- `C`: Building C (ID: 4746)

### Pipeline

The core class responsible for data processing.

#### Initialization of class

```python
Pipeline(
    energy_path="data/energy_import_export.csv",
    solar_path="data/solar_self_consumption_main_building.csv",
    met_path="data/met_data.csv"
)
```

#### Methods

- `process()`: Orchestrates the data processing steps.
- `process_energy()`: Processes energy import/export data.
- `process_solar()`: Processes solar consumption data.
- `process_met()`: Processes meteorological data.
- `merge_solar_and_main()`: Merges solar data with the main building's energy data.
- `merge_met()`: Merges meteorological data with each building's energy data.
- `calculate_columns()`: Calculates additional columns like `net_consumption`.
- `order_columns()`: Orders the DataFrame columns for consistency.
- `get_data(building: BuilingIdsEnum) -> pd.DataFrame`: Retrieves processed data for a specified building.
- `get_consumption(building: BuilingIdsEnum, freq: str) -> pd.DataFrame`: General method to get consumption data with specified frequency.
- `get_daily_consumption(building: BuilingIdsEnum) -> pd.DataFrame`: Retrieves daily consumption data.
- `get_weekly_consumption(building: BuilingIdsEnum) -> pd.DataFrame`: Retrieves weekly consumption data.
- `get_yearly_consumption(building: BuilingIdsEnum) -> pd.DataFrame`: Retrieves yearly consumption data.
- `interpolate_circular(df: pd.DataFrame, column: str) -> pd.DataFrame`: Interpolates circular data (e.g., wind direction).

## Example

Here's a simple example of how to use the `Pipeline` class to retrieve and display daily energy consumption for the main building.

```python
from pipeline_module import Pipeline, BuilingIdsEnum  # Adjust the import as necessary

if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = Pipeline()

    # Retrieve main building data
    main_building = pipeline.get_data(BuilingIdsEnum.MAIN)

    # Get daily consumption
    daily_main = pipeline.get_daily_consumption(BuilingIdsEnum.MAIN)

    # Display the first few rows
    print(daily_main.head())
```

**Output:**

```bash
   timestamp  value_import  value_export  solar_consumption  net_consumption    building  temperature  wind_speed  cloud_fraction  precipitation  wind_direction
0 2024-01-01      1234.567      234.56789             100.123          1099.123  main building        5.6         3.2              0.1            0.0           180.0
1 2024-01-02      1300.890      250.89012             110.456          1160.346  main building        6.1         2.8              0.0            0.2           190.0
2 2024-01-03      1250.123      240.12345              90.789          900.789  main building        4.9         3.5              0.2            0.1           170.0
3 2024-01-04      1280.456      245.45678             105.321          940.321  main building        5.2         3.0              0.1            0.0           200.0
4 2024-01-05      1275.789      242.78901             102.654          935.654  main building        5.0         3.3              0.3            0.0           185.0
```
