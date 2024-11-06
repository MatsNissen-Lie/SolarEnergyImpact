# How does the installation of solar panels on the main building affect energy consumption compared to other buildings without these upgrades?

Lets find out!

## TODO:

- [ ] Analyse the data by energy consumption per square_meter

## DataPipeline Class Overview

The `DataPipeline` class provides a structured approach to managing, preprocessing, and analyzing energy, solar, and meteorological data for different buildings. It leverages the `pandas` library to load and merge datasets, calculate building-specific energy consumption, and compute daily and weekly summaries for easy access and analysis.

### Files

- `pipeline.py`: Contains the `DataPipeline` class for loading, preprocessing, and analyzing energy data.
- `data`: Directory containing the energy, solar, and meteorological data files in CSV format.

### Key Components

- **Building Mapping**: The class includes mappings for each building, defining property IDs, names, and areas for accurate data association.
- **Data Loading and Preprocessing**: Loads data from specified CSV files, converting timestamps to `datetime` format, standardizing column names, and merging datasets based on property ID and timestamps.
- **Data Segmentation**: Divides energy data into separate import (consumption) and export (production) DataFrames, allowing for specific building or energy direction queries.
- **Utility Methods**:
  - `get_energy_data`, `get_solar_data`, `get_met_data`: Return individual datasets.
  - `get_import_data`, `get_export_data`: Provide import/export data for all buildings.
  - `get_main_building_consumption_data`: Integrates solar self-consumption data for the main building.
  - `compute_daily_consumption`, `compute_weekly_consumption`: Aggregates energy data on a daily or weekly basis, with optional grouping by building.
  - `get_building_data`, `get_solar_data_for_building`, `get_met_data_for_building`: Retrieve data specific to a particular building.

### Usage

To use the class, instantiate it and load the data as follows:

```python
pipe = DataPipeline()
pipe.load_data()
```

This initializes the data pipeline, processes the data, and makes various datasets and calculated metrics available for further analysis.

### Example Analysis

To calculate the daily energy consumption for the main building, you can use the following code snippet:

```python
main_building_daily_consumption = pipe.compute_daily_consumption(
    building='Main Building',
    energy_type='import'
)
```

This will provide a DataFrame with the daily energy consumption data for the main building, which can then be used for further analysis and visualization.

### Conclusion

The `DataPipeline` class offers a convenient way to manage and analyze energy data for different buildings, enabling users to gain insights into energy consumption patterns, solar production, and meteorological conditions. By leveraging the structured approach provided by the class, users can perform complex analyses and answer questions related to energy efficiency and solar panel installations with ease.

For more information on how to use the `DataPipeline` class and conduct detailed analyses, please refer to the code documentation and examples provided in the Jupyter notebook.
