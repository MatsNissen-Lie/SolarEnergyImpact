from enum import Enum
import sys
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import matplotlib.pyplot as plt


# sys add path
folder = Path().resolve().absolute().parent
sys.path.append(str(folder))

from utils import ColumnParam


# from src.utils import ColumnParam


class BuilingIdsEnum(Enum):  # name: building_id
    MAIN = 10724
    A = 10703
    B = 4462
    C = 4746


class Pipeline:
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
        self.prediction_path = (
            self.base_path / "data/prediciton_3features_08_11_2024.csv"
        )
        self.prediction_path = self.base_path / "data/pred/prediction_drita2.csv"
        # self.prediction_path = self.base_path / "data/pred/prediction_main_b.csv"
        # self.prediction_path = self.base_path / "data/pred/prediction_b.csv"
        self.prediction_path = self.base_path / "data/pred/prediction_winter.csv"
        self.energy_prices_path = self.base_path / "data/energy_prices.csv"
        self.eurnok_path = self.base_path / "data/eurnok.csv"

        # datasets unaltered
        self.met_df = pd.read_csv(self.met_path)
        self.energy_df = pd.read_csv(self.energy_path)
        self.solar_df = pd.read_csv(self.solar_path)

        # datasets final
        self.main = None
        self.a = None
        self.b = None
        self.c = None
        self.processed_met_data: dict[str, pd.DataFrame] = {}

        self.name_dict_energy = {
            "Energikilde": "energy_source",
            "Retning": "direction",
            "Målernavn": "meter_name",
            "Måler-Id": "meter_id",
            "Verdi": "value",  # maybe rename to import, export depending on value
            "Tidspunkt": "timestamp",
        }
        self.meter_property_mapping = {  # meter_id: building_id
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
        self.process()

    def process(self):
        self.process_energy()
        self.process_solar()
        self.process_met()
        # add

        self.merge_solar_and_main()
        self.merge_prediciton()
        self.merge_met()

        self.calculate_columns()
        self.order_columns()

        # add spot prices
        self.merge_spot_prices()

    def process_energy(self):
        self.energy_df.rename(
            columns={
                "Energikilde": "energy_source",
                "Retning": "direction",
                "Målernavn": "meter_name",
                "Måler-Id": "meter_id",
                "Verdi": "value",  # maybe rename to import, export depending on value
                "Tidspunkt": "timestamp",
            },
            inplace=True,
        )
        self.energy_df["timestamp"] = pd.to_datetime(self.energy_df["timestamp"])
        self.energy_df["meter_id"] = self.energy_df["meter_id"].astype(str).str.strip()
        self.energy_df["property_id"] = self.energy_df["meter_id"].map(
            self.meter_property_mapping
        )
        self.energy_df["building"] = self.energy_df["property_id"].map(
            self.id_to_name_property_mapping
        )
        self.energy_df["area"] = self.energy_df["property_id"].map(self.building_areas)

        # df = df.drop(["energy_source", "direction", "meter_name"], axis=1)
        self.energy_df.drop(
            ["meter_id", "energy_source", "meter_name"], axis=1, inplace=True
        )
        import_energy = self.energy_df[self.energy_df["direction"] == "IMPORT"].copy()
        export_energy = self.energy_df[self.energy_df["direction"] == "EXPORT"].copy()
        # drop direction column
        import_energy.drop(["direction"], axis=1, inplace=True)
        export_energy.drop(["direction"], axis=1, inplace=True)

        # make a df for each bulding and add the import and export values for the main building. Merge on time and drop all duplicates.
        self.main = (
            import_energy[import_energy["property_id"] == 10724]
            .merge(
                export_energy[export_energy["property_id"] == 10724][
                    ["timestamp", "value"]
                ],
                on="timestamp",
                # how="outer", # it does not matter because they have all the same timestamps. No difference.
                suffixes=("_import", "_export"),
            )
            .drop_duplicates()
        )

        # add the import values for the other buildings
        # rename to value_import
        import_energy.rename(columns={"value": "value_import"}, inplace=True)
        self.a = import_energy[import_energy["property_id"] == 10703]
        self.b = import_energy[import_energy["property_id"] == 4462]
        self.c = import_energy[import_energy["property_id"] == 4746]
        datasets = [self.main, self.a, self.b, self.c]

        for df in datasets:
            df.sort_values("timestamp")

            df.set_index("timestamp", inplace=True)
            numeric_cols = df.select_dtypes(include="number").columns
            string_cols = df.select_dtypes(exclude="number").columns

            # Resample and apply different functions for numeric and string columns
            df_resampled = pd.DataFrame()
            if not numeric_cols.empty:
                df_resampled[numeric_cols] = df[numeric_cols].resample("h").mean()
            if not string_cols.empty:
                df_resampled[string_cols] = df[string_cols].resample("h").first()

            for col in ["value_import", "value_export"]:
                if col in df_resampled.columns:
                    df_resampled[col] = df_resampled[col].interpolate(method="linear")
                    # df_resampled[col] = df_resampled[col].ffill()
                    # df_resampled[col] = df_resampled[col].bfill()

            df.reset_index(inplace=True)
            df.update(df_resampled.reset_index())

    def process_solar(self):
        # Ensuring all dates are in a consistent format with hour component
        self.solar_df.drop("Unnamed: 0", axis=1, inplace=True)
        self.solar_df.rename(columns={"starting_at": "timestamp"}, inplace=True)

        self.solar_df["timestamp"] = self.solar_df["timestamp"].apply(
            lambda x: f"{x} 00:00:00" if len(x.split()) == 1 else x
        )
        self.solar_df["timestamp"] = pd.to_datetime(
            self.solar_df["timestamp"], errors="coerce"
        )

        # Corrected line to reassign the grouped result
        self.solar_df = self.solar_df.groupby("timestamp", as_index=False).aggregate(
            {"solar_consumption": "sum"}
        )

        self.solar_df.set_index("timestamp", inplace=True)

        # print duplicate dates
        desired_frequency = "h"  # Ensure this matches main data frequency
        self.solar_df = self.solar_df.resample(desired_frequency).asfreq()
        self.solar_df["solar_consumption"] = self.solar_df[
            "solar_consumption"
        ].interpolate(method="linear")
        self.solar_df["solar_consumption"] = self.solar_df["solar_consumption"].ffill()
        self.solar_df["solar_consumption"] = self.solar_df["solar_consumption"].bfill()

        # Optional: Handle night-time values if applicable
        night_hours = (self.solar_df.index.hour >= 22) | (self.solar_df.index.hour <= 4)
        self.solar_df.loc[night_hours, "solar_consumption"] = 0

        # reset index
        self.solar_df.reset_index(inplace=True)

    def process_met(self):
        # Step 1: Rename columns for consistency
        self.met_df.rename(columns={"starting_at": "timestamp"}, inplace=True)
        self.met_df["timestamp"] = self.met_df["timestamp"].str.split("+").str[0]
        # Step 2: Convert 'timestamp' to datetime
        self.met_df["timestamp"] = pd.to_datetime(self.met_df["timestamp"])

        # Step 3: Sort the DataFrame
        self.met_df.sort_values(["property_id", "timestamp"], inplace=True)

        # Step 4: Group the DataFrame by 'property_id'
        grouped = self.met_df.groupby("property_id")

        # Step 5: Process each group separately
        for property_id, group in grouped:
            processed_group = self.process_property_group(group)
            # remove property_id
            processed_group.drop("property_id", axis=1, inplace=True)
            self.processed_met_data[property_id] = processed_group

            # Optional: Print or plot the processed group
            # print(processed_group.head())
            # processed_group.plot(subplots=True, figsize=(12, 10))
            # plt.suptitle(f"Property ID: {property_id}")
            # plt.tight_layout()
            # plt.show()

    def process_spot_prices(self):

        # Read the spot prices
        spot_prices = pd.read_csv(self.energy_prices_path, sep=";")
        # HourUTC;HourDK;PriceArea;SpotPriceDKK;SpotPriceEUR

        # select the relevant columns
        spot_prices = spot_prices[["HourDK", "SpotPriceEUR"]]
        spot_prices.rename(
            columns={"SpotPriceEUR": "spot_price_eur", "HourDK": "timestamp"},
            inplace=True,
        )
        spot_prices["timestamp"] = pd.to_datetime(spot_prices["timestamp"])
        # 97,269997 trun spot price to float
        spot_prices["spot_price_eur"] = (
            spot_prices["spot_price_eur"].str.replace(",", ".").astype(float)
        )

        spot_prices = spot_prices.groupby("timestamp").mean().reset_index()
        spot_prices = spot_prices.set_index("timestamp").resample("h").ffill()
        spot_prices.reset_index(inplace=True)

        # merge with dataset a and fill sopot price with the last value ffill
        a_dates = self.a["timestamp"]
        spot_prices = spot_prices.merge(a_dates, on="timestamp", how="outer")
        spot_prices["spot_price_eur"] = spot_prices["spot_price_eur"].ffill()

        eur_nok = pd.read_csv(self.eurnok_path)
        # datetime,open,high,low,close
        eur_nok = eur_nok[["datetime", "close"]]
        eur_nok.rename(
            columns={"datetime": "timestamp", "close": "eur_nok"}, inplace=True
        )
        eur_nok["timestamp"] = pd.to_datetime(eur_nok["timestamp"])

        spot_prices = spot_prices.merge(eur_nok, on="timestamp", how="left")
        spot_prices["eur_nok"] = spot_prices["eur_nok"].ffill()
        spot_prices["spot_price_nok"] = (
            spot_prices["spot_price_eur"] * spot_prices["eur_nok"]
        )
        # drop eur_nok and spot_price_eur
        spot_prices.drop(["eur_nok", "spot_price_eur"], axis=1, inplace=True)

        # mwatthour to kwh
        spot_prices["spot_price_nok"] = spot_prices["spot_price_nok"] / 1000

        return spot_prices

    def process_property_group(self, group: pd.DataFrame) -> pd.DataFrame:

        group = group.reset_index(drop=True)

        group.set_index("timestamp", inplace=True)

        desired_frequency = "h"
        group = group.resample(desired_frequency).asfreq()

        # group["property_id"] = group["property_id"].fillna(method="ffill") #dette gir ikke mening

        numeric_cols = ["temperature", "wind_speed", "cloud_fraction", "precipitation"]
        group[numeric_cols] = group[numeric_cols].interpolate(method="time")

        group = self.interpolate_circular(group, "wind_direction")

        group.ffill(inplace=True)
        group.bfill(inplace=True)

        group.reset_index(inplace=True)

        return group

    def interpolate_circular(self, df, column):
        """
        Interpolates circular data (e.g., wind_direction) by converting to sine and cosine,
        interpolating these components, and then converting back to degrees.
        """
        # Convert degrees to radians
        radians = np.deg2rad(df[column])

        # Compute sine and cosine components
        sin_col = np.sin(radians)
        cos_col = np.cos(radians)

        # Create temporary DataFrame for sine and cosine
        temp_df = pd.DataFrame(
            {f"{column}_sin": sin_col, f"{column}_cos": cos_col}, index=df.index
        )

        # Interpolate sine and cosine components
        temp_df_interpolated = temp_df.interpolate(method="time")

        # Convert back to degrees
        interpolated_radians = np.arctan2(
            temp_df_interpolated[f"{column}_sin"], temp_df_interpolated[f"{column}_cos"]
        )
        interpolated_degrees = np.rad2deg(interpolated_radians) % 360
        df[column] = interpolated_degrees

        return df

    def merge_solar_and_main(self):
        self.main = self.main.merge(self.solar_df, on="timestamp", how="left")

    def merge_prediciton(self):
        prediction = pd.read_csv(self.prediction_path)
        prediction.rename(
            columns={prediction.columns[0]: "predicted_consumption"}, inplace=True
        )
        if not prediction.index.equals(self.main.index):
            raise ValueError("Index does not match")

        avf_pred = prediction["predicted_consumption"].mean()
        # print("average prediction", avf_pred)
        if avf_pred < 1:
            # print("scaling prediction", avf_pred)

            area = self.building_areas[10724]
            prediction["predicted_consumption"] = (
                prediction["predicted_consumption"] * area
            )
        # merge on index. should be good
        self.main = self.main.merge(
            prediction, left_index=True, right_index=True, how="left"
        )

    def merge_met(self):
        datasets: pd.DataFrame = [self.main, self.a, self.b, self.c]
        for i, df in enumerate(datasets):
            property_id = df["property_id"].unique()[0]
            met = self.processed_met_data[property_id]
            merged = pd.merge(df, met, on="timestamp", how="left")

            # df.update(merged) # this does not work for some reason, quick fix
            if i == 0:
                self.main = merged
            elif i == 1:
                self.a = merged
            elif i == 2:
                self.b = merged
            elif i == 3:
                self.c = merged

    def merge_spot_prices(self):
        datasets = [self.main, self.a, self.b, self.c]
        spot_prices = self.process_spot_prices()
        for i, df in enumerate(datasets):
            merged = pd.merge(df, spot_prices, on="timestamp", how="left")
            if i == 0:
                self.main = merged
            elif i == 1:
                self.a = merged
            elif i == 2:
                self.b = merged
            elif i == 3:
                self.c = merged

    def get_consumption(
        self, building: BuilingIdsEnum, freq: str = "D"
    ) -> pd.DataFrame:
        """
        Calculate net energy consumption for a building over a specified frequency.

        Parameters:
            building (BuilingIdsEnum): The building to calculate consumption for.
            freq (str): Resampling frequency ('D' for daily, 'W' for weekly, 'Y' for monthly).

        Returns:
            pd.DataFrame: DataFrame containing net consumption over the specified period.
        """
        df = self.get_data(building).copy()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

        agg_funcs = {}
        to_sum = [
            "value_import",
            "value_export",
            "solar_consumption",
            "net_consumption",
            "predicted_consumption",
            "net_consumption_per_sqm",
        ]
        met_cols = ["temperature", "wind_speed", "cloud_fraction", "precipitation"]
        for col in df.columns:
            if col in to_sum:
                agg_funcs[col] = "sum"
            elif col in met_cols:
                agg_funcs[col] = "mean"
            else:  # Non-numeric columns
                agg_funcs[col] = "first"

        # Resample and aggregate
        df_resampled = df.resample(freq).agg(agg_funcs)

        # Reset index to have 'timestamp' as a column again
        df_resampled.reset_index(inplace=True)

        return df_resampled

    def get_daily_consumption(self, building: BuilingIdsEnum) -> pd.DataFrame:
        """
        Get daily net consumption for a building.

        Parameters:
            building (BuilingIdsEnum): The building to calculate daily consumption for.

        Returns:
            pd.DataFrame: DataFrame containing daily net consumption.
        """
        return self.get_consumption(building, freq="D")

    def get_weekly_consumption(self, building: BuilingIdsEnum) -> pd.DataFrame:
        """
        Get weekly net consumption for a building.

        Parameters:
            building (BuilingIdsEnum): The building to calculate weekly consumption for.

        Returns:
            pd.DataFrame: DataFrame containing weekly net consumption.
        """
        return self.get_consumption(building, freq="W")

    def get_monthly_consumption(self, building: BuilingIdsEnum) -> pd.DataFrame:
        """
        Get monthly net consumption for a building.

        Parameters:
            building (BuilingIdsEnum): The building to calculate monthly consumption for.

        Returns:
            pd.DataFrame: DataFrame containing monthly net consumption.
        """
        return self.get_consumption(building, freq="ME")

    def calculate_columns(self):
        self.main["net_consumption"] = (
            self.main["value_import"]
            - self.main["value_export"]
            + self.main["solar_consumption"]
        )
        self.a["net_consumption"] = self.a["value_import"]
        self.b["net_consumption"] = self.b["value_import"]
        self.c["net_consumption"] = self.c["value_import"]
        # add net_consumption per square meter
        self.main["net_consumption_per_sqm"] = (
            self.main["net_consumption"] / self.main["area"]
        )
        self.a["net_consumption_per_sqm"] = self.a["net_consumption"] / self.a["area"]
        self.b["net_consumption_per_sqm"] = self.b["net_consumption"] / self.b["area"]
        self.c["net_consumption_per_sqm"] = self.c["net_consumption"] / self.c["area"]

    def order_columns(self):
        ordered = [
            "timestamp",
            "value_import",
            "value_export",
            "solar_consumption",
            "net_consumption",
            "net_consumption_per_sqm",
            "predicted_consumption",
            "building",
        ]
        # if str in columns reorder them
        for i, building in enumerate([self.main, self.a, self.b, self.c]):
            # for col in building.columns:
            #     if col in ordered:
            #         building = building
            cols = building.columns.tolist()
            new_order = [col for col in ordered if col in cols]
            # append all cols that are not in ordered
            new_order += [col for col in cols if col not in ordered]
            if i == 0:
                self.main = building[new_order]
            elif i == 1:
                self.a = building[new_order]
            elif i == 2:
                self.b = building[new_order]
            elif i == 3:
                self.c = building[new_order]

    def get_data(self, building: BuilingIdsEnum):
        if building == BuilingIdsEnum.MAIN:
            return self.main.copy()
        elif building == BuilingIdsEnum.A:
            return self.a.copy()
        elif building == BuilingIdsEnum.B:
            return self.b.copy()
        elif building == BuilingIdsEnum.C:
            return self.c.copy()

    def select_and_merge_datasets(self, cols=["net_consumption_per_sqm"], periode="d"):
        """
        Select and merge datasets for all buildings.

        Parameters:
            cols (list[str]): List of columns to select.
            periode (str): The period to calculate consumption for ('d' for daily, 'w' for weekly, 'm' for monthly).

        """
        if periode == "h":
            get_data = self.get_data
        elif periode == "d":
            get_data = self.get_daily_consumption
        elif periode == "w":
            get_data = self.get_weekly_consumption
        elif periode == "m":
            get_data = self.get_monthly_consumption
        else:
            raise ValueError("Invalid period. Must be 'd', 'w', or 'm'.")
        dfs = [get_data(building) for building in BuilingIdsEnum]
        merged_df = None

        col_params = []
        for i, data in enumerate(dfs):
            name = data["building"].unique()[0].lower().replace(" ", "_")
            filted_cols = [col for col in cols if col in data.columns]
            data = data[["timestamp"] + filted_cols]
            if i == 0:
                merged_df = data
            else:
                merged_df = merged_df.merge(
                    data,
                    on="timestamp",
                    suffixes=("", "_" + name),
                )
            col_names = [col + "_" + name if i != 0 else col for col in filted_cols]
            col_params.extend(
                [ColumnParam(col, name.replace("_", " ")) for col in col_names]
            )

        return merged_df, col_params


if __name__ == "__main__":
    p = Pipeline()
    main = p.get_data(BuilingIdsEnum.MAIN)
    # daily_main = p.get_daily_consumption(BuilingIdsEnum.MAIN)
    select_and_merge_datasets = p.select_and_merge_datasets(
        ["value_import", "solar_consumption"], periode="h"
    )

    print(select_and_merge_datasets[0].tail())

    res = p.process_spot_prices()
    # print(res)
