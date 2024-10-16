from enum import Enum
import pandas as pd
import datetime
from pathlib import Path


class BuilingIdsEnum(Enum):
    MAIN = 10724
    A = 10703
    B = 4462
    C = 4746


class DataPipeline:
    def __init__(
        self,
        energy_path="data/energy_import_export.csv",
        solar_path="data/solar_self_consumption_main_building.csv",
        met_path="data/met_data.csv",
    ):
        self.base_path = Path(__file__).resolve().parent.parent
        self.energy_path = self.base_path / energy_path
        self.solar_path = self.base_path / solar_path
        self.met_path = self.base_path / met_path
        self.energy_df = None
        self.solar_df = None
        self.met_df = None
        self.merged_df = None
        self.import_df = None
        self.export_df = None

        # Mappings for meter IDs to property IDs and names
        self.meter_property_mapping = {
            "707057500042745649": 10724,  # main building
            "707057500038344962": 10703,  # building A
            "707057500085390523": 4462,  # building B
            "707057500042201572": 4746,  # building C
        }
        self.id_to_name_property_mapping = {
            10724: "main building",
            10703: "building A",
            4462: "building B",
            4746: "building C",
        }
        # Building areas in square meters
        self.building_areas = {
            10724: 1199,  # Main building
            10703: 1167,
            4462: 1095,
            4746: 1384,
        }
        self.load_data()
        self.preprocess_data()

    def load_data(self):
        """Loads data from CSV files into DataFrames."""
        self.energy_df = pd.read_csv(self.energy_path)
        self.solar_df = pd.read_csv(self.solar_path)
        self.met_df = pd.read_csv(self.met_path)

    def preprocess_data(self):
        """Preprocesses the loaded data."""
        # Convert timestamp columns to datetime
        self.energy_df["Tidspunkt"] = pd.to_datetime(
            self.energy_df["Tidspunkt"], format="mixed"
        )
        self.solar_df["starting_at"] = pd.to_datetime(
            self.solar_df["starting_at"], format="mixed"
        )

        # Remove UTC offset from 'starting_at' in met_df and convert to datetime
        self.met_df["starting_at"] = self.met_df["starting_at"].str.split("+").str[0]
        self.met_df["starting_at"] = pd.to_datetime(
            self.met_df["starting_at"], format="mixed"
        )

        # Rename columns for consistency
        self.energy_df.rename(
            columns={
                "Energikilde": "energy_source",
                "Retning": "direction",
                "Målernavn": "meter_name",
                "Måler-Id": "meter_id",
                "Verdi": "value",
                "Tidspunkt": "timestamp",
            },
            inplace=True,
        )

        self.solar_df.rename(columns={"starting_at": "timestamp"}, inplace=True)
        self.met_df.rename(columns={"starting_at": "timestamp"}, inplace=True)

        # Map meter IDs to property IDs and building names
        self.energy_df["meter_id"] = self.energy_df["meter_id"].astype(str).str.strip()
        self.energy_df["property_id"] = self.energy_df["meter_id"].map(
            self.meter_property_mapping
        )
        self.energy_df["building"] = self.energy_df["property_id"].map(
            self.id_to_name_property_mapping
        )
        self.energy_df["area"] = self.energy_df["property_id"].map(self.building_areas)

        # Merge energy and meteorological data on 'property_id' and 'timestamp'
        self.merged_df = pd.merge(
            self.energy_df, self.met_df, on=["property_id", "timestamp"], how="left"
        )

        # Separate data into import and export DataFrames
        self.import_df = self.merged_df[self.merged_df["direction"] == "IMPORT"]
        self.export_df = self.merged_df[self.merged_df["direction"] == "EXPORT"]

    def get_energy_data(self):
        """Returns the processed energy DataFrame."""
        return self.energy_df

    def get_solar_data(self):
        """Returns the processed solar DataFrame."""
        return self.solar_df

    def get_met_data(self):
        """Returns the processed meteorological DataFrame."""
        return self.met_df

    def get_merged_data(self):
        """Returns the merged DataFrame of energy and meteorological data."""
        return self.merged_df

    def get_import_data(self):
        """Returns the DataFrame containing import (consumption) data."""
        return self.import_df

    def get_export_data(self):
        """Returns the DataFrame containing export (production) data."""
        return self.export_df

    def get_import_data_for_building(self, property_id):
        """Returns the import data for a specific building."""
        return self.import_df[self.import_df["property_id"] == property_id]

    def get_export_data_for_building(self, property_id):
        """Returns the export data for a specific building."""
        return self.export_df[self.export_df["property_id"] == property_id]

    def get_main_building_consumption_data(self):
        """
        Processes and returns the main building consumption data,
        including self-consumed solar energy.
        """
        # Get import data for main building
        main_building_import = self.get_import_data_for_building(10724)

        # Merge with solar data
        main_building_df = pd.merge(
            main_building_import, self.solar_df, on="timestamp", how="left"
        )

        # Replace NaN in 'solar_consumption' with 0
        if "solar_consumption" in main_building_df.columns:
            main_building_df["solar_consumption"] = main_building_df[
                "solar_consumption"
            ].fillna(0)
        else:
            main_building_df["solar_consumption"] = 0

        # Calculate total consumption including self-consumed solar energy
        main_building_df["total_consumption"] = (
            main_building_df["value"] + main_building_df["solar_consumption"]
        )

        return main_building_df

    def get_reference_buildings_import_data(self):
        """Returns the import data for reference buildings."""
        ref_buildings = [10703, 4462, 4746]
        ref_buildings_df = self.import_df[
            self.import_df["property_id"].isin(ref_buildings)
        ]
        return ref_buildings_df

    def compute_daily_consumption(self, df, group_by_building=False):
        """
        Computes daily consumption from a given DataFrame.

        Parameters:
            df (DataFrame): The DataFrame to compute consumption from.
            group_by_building (bool): Whether to group by building.

        Returns:
            DataFrame: A DataFrame with daily consumption.
        """
        if group_by_building:
            daily_consumption = (
                df.groupby(["property_id", df["timestamp"].dt.date])["value"]
                .sum()
                .reset_index()
            )
        else:
            daily_consumption = (
                df.groupby(df["timestamp"].dt.date)["value"].sum().reset_index()
            )

        daily_consumption.rename(
            columns={"timestamp": "date", "value": "daily_consumption"}, inplace=True
        )
        return daily_consumption

    def compute_weekly_consumption(self, df, group_by_building=False):
        """
        Computes weekly consumption from a given DataFrame.

        Parameters:
            df (DataFrame): The DataFrame to compute consumption from.
            group_by_building (bool): Whether to group by building.

        Returns:
            DataFrame: A DataFrame with weekly consumption and timestamps.
        """
        df["year"] = df["timestamp"].dt.isocalendar().year
        df["week"] = df["timestamp"].dt.isocalendar().week

        if group_by_building:
            weekly_consumption = (
                df.groupby(["property_id", "year", "week"])["value"].sum().reset_index()
            )
        else:
            weekly_consumption = (
                df.groupby(["year", "week"])["value"].sum().reset_index()
            )

        # Create a representative timestamp for the week (e.g., the first day of the week)
        weekly_consumption["timestamp"] = weekly_consumption.apply(
            lambda x: datetime.date.fromisocalendar(int(x["year"]), int(x["week"]), 1),
            axis=1,
        )
        weekly_consumption["timestamp"] = pd.to_datetime(
            weekly_consumption["timestamp"]
        )

        weekly_consumption.rename(columns={"value": "weekly_consumption"}, inplace=True)
        return weekly_consumption

    def get_building_data(self, building_name):
        """
        Returns data for a specific building.

        Parameters:
            building_name (str): The name of the building (e.g., 'main building').

        Returns:
            DataFrame: A DataFrame containing data for the specified building.
        """
        return self.merged_df[self.merged_df["building"] == building_name]

    def get_solar_data_for_building(self, building_name):
        """
        Returns solar data for a specific building.

        Parameters:
            building_name (str): The name of the building.

        Returns:
            DataFrame: A DataFrame containing solar data for the specified building.
        """
        # Assuming solar_df has a 'building' column or can be mapped similarly
        return self.solar_df[self.solar_df["building"] == building_name]

    def get_met_data_for_building(self, building_name):
        """
        Returns meteorological data for a specific building.

        Parameters:
            building_name (str): The name of the building.

        Returns:
            DataFrame: A DataFrame containing meteorological data for the specified building.
        """
        property_id = None
        for pid, name in self.id_to_name_property_mapping.items():
            if name == building_name:
                property_id = pid
                break
        if property_id:
            return self.met_df[self.met_df["property_id"] == property_id]
        else:
            return pd.DataFrame()  # Return empty DataFrame if building not found


if "__main__" == __name__:
    pipe = DataPipeline()
    pipe.load_data()
