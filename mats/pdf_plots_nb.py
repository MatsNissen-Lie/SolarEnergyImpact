# %%
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %% [markdown]
# ## 1. Load the Datasets

# %%
energy_df = pd.read_csv("../data/energy_import_export.csv")
solar_df = pd.read_csv("../data/solar_self_consumption_main_building.csv")
met_df = pd.read_csv("../data/met_data.csv")


# %%
solar_df.head()
solar_df = solar_df.drop(columns=["Unnamed: 0"])

# %% [markdown]
# ## 2. Preprocess the Data
#
# ### 2.1 Convert Timestamps to Datetime Objects

# %%
# Convert timestamp columns to datetime
energy_df["Tidspunkt"] = pd.to_datetime(energy_df["Tidspunkt"], format="mixed")
solar_df["starting_at"] = pd.to_datetime(solar_df["starting_at"], format="mixed")

# remove utc from the time string
met_df["starting_at"] = met_df["starting_at"].str.split("+").str[0]
met_df["starting_at"] = pd.to_datetime(met_df["starting_at"], format="mixed")

# %% [markdown]
# ### 2.2 Rename Columns for Consistency

# %%
energy_df.rename(
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


solar_df.rename(columns={"starting_at": "timestamp"}, inplace=True)
met_df.rename(columns={"starting_at": "timestamp"}, inplace=True)


# %% [markdown]
# ### 2.3 Map Property IDs to Meter IDs

# %%
meter_property_mapping = {
    "707057500042745649": 10724,  # main building
    "707057500038344962": 10703,  # building A
    "707057500085390523": 4462,  # building B
    "707057500042201572": 4746,  # building C
}
id_to_name_property_mapping = {
    10724: "main building",
    10703: "building A",
    4462: "building B",
    4746: "building C",
}


def get_name(id):
    return id_to_name_property_mapping[id]


energy_df["meter_id"] = energy_df["meter_id"].astype(str)
energy_df["meter_id"] = energy_df["meter_id"].str.strip()

energy_df["property_id"] = energy_df["meter_id"].map(meter_property_mapping)
energy_df["building"] = energy_df["property_id"].map(id_to_name_property_mapping)

# %% [markdown]
# ### 2.4 Merge Datasets

# %%
merged_df = pd.merge(energy_df, met_df, on=["property_id", "timestamp"], how="left")

# %%
# Check for missing values
print("total number of rows", merged_df.shape[0])
print("rows missing weather data \n", merged_df.isnull().sum())

rows_with_missing_values = merged_df[merged_df.isnull().any(axis=1)]
# rows_with_missing_values.head()


# %%
# Separate import (consumption) and export (production)
import_df = merged_df[merged_df["direction"] == "IMPORT"]
export_df = merged_df[merged_df["direction"] == "EXPORT"]


# %%
main_building_import = import_df[import_df["property_id"] == 10724]
main_building_export = export_df[export_df["property_id"] == 10724]

main_building_df = pd.merge(
    main_building_import,
    solar_df,
    left_on="timestamp",
    right_on="timestamp",
    how="left",
)

# Replace NaN in solar_consumption with 0
main_building_df["solar_consumption"] = main_building_df["solar_consumption"].fillna(0)

# Calculate total consumption including self-consumed solar energy
main_building_df["total_consumption"] = (
    main_building_df["value"] + main_building_df["solar_consumption"]
)


# %%
# Get import data for reference buildings
ref_buildings = [10703, 4462, 4746]
ref_buildings_df = import_df[import_df["property_id"].isin(ref_buildings)]

# Aggregate data to daily or monthly consumption
main_daily_consumption = main_building_df.groupby(
    main_building_df["timestamp"].dt.date
)[
    "value"
].sum()  # .reset_index()
ref_daily_consumption = (
    ref_buildings_df.groupby(["property_id", ref_buildings_df["timestamp"].dt.date])[
        "value"
    ]
    .sum()
    .reset_index()
)

# # aggragate weekly sum
# main_building_df['year'] = main_building_df['timestamp'].dt.year
# main_building_df['week'] = main_building_df['timestamp'].dt.isocalendar().week

# ref_buildings_df['year'] = ref_buildings_df['timestamp'].dt.year
# ref_buildings_df['week'] = ref_buildings_df['timestamp'].dt.isocalendar().week

# main_weekly_consumption = main_building_df.groupby(['year', 'week'])['value'].sum()
# ref_weekly_consumption = ref_buildings_df.groupby(['property_id', 'year', 'week'])['value'].sum().reset_index()

# # aggregate monthly sum
# main_monthly_consumption = main_building_df.groupby(main_building_df['timestamp'].dt.month)['value'].sum()#.reset_index()
# ref_monthly_consumption = ref_buildings_df.groupby(['property_id', ref_buildings_df['timestamp'].dt.month])['value'].sum().reset_index()

# main_daily_consumption

# %%
# Add 'year' and 'week' columns
import datetime


main_building_df["year"] = main_building_df["timestamp"].dt.isocalendar().year
main_building_df["week"] = main_building_df["timestamp"].dt.isocalendar().week

ref_buildings_df["year"] = ref_buildings_df["timestamp"].dt.isocalendar().year
ref_buildings_df["week"] = ref_buildings_df["timestamp"].dt.isocalendar().week

# Weekly Aggregation for Main Building
main_weekly_consumption = main_building_df.groupby(["year", "week"])["value"].sum()
main_weekly_consumption.index = pd.MultiIndex.from_tuples(
    main_weekly_consumption.index, names=["year", "week"]
)

main_weekly_consumption = main_weekly_consumption.reset_index()
main_weekly_consumption["timestamp"] = main_weekly_consumption.apply(
    lambda x: datetime.date.fromisocalendar(int(x["year"]), int(x["week"]), 1), axis=1
)
main_weekly_consumption["timestamp"] = pd.to_datetime(
    main_weekly_consumption["timestamp"]
)
main_weekly_consumption.index = main_weekly_consumption["timestamp"]
main_weekly_consumption.drop(columns=["timestamp", "year", "week"], inplace=True)

# Weekly Aggregation for Reference Buildings
ref_weekly_consumption = (
    ref_buildings_df.groupby(["property_id", "year", "week"])["value"]
    .sum()
    .reset_index()
)
# Optionally, create a representative timestamp for the week (e.g., the first day of the week)
# Convert 'year' and 'week' to integers
ref_weekly_consumption["year"] = ref_weekly_consumption["year"].astype(int)
ref_weekly_consumption["week"] = ref_weekly_consumption["week"].astype(int)

ref_weekly_consumption["timestamp"] = ref_weekly_consumption.apply(
    lambda x: datetime.date.fromisocalendar(int(x["year"]), int(x["week"]), 1), axis=1
)
ref_weekly_consumption["timestamp"] = pd.to_datetime(
    ref_weekly_consumption["timestamp"]
)


# %%
# Monthly Aggregation for Main Building
main_monthly_consumption = (
    main_building_df.groupby(main_building_df["timestamp"].dt.to_period("M"))["value"]
    .sum()
    .reset_index()
)
main_monthly_consumption["timestamp"] = main_monthly_consumption[
    "timestamp"
].dt.to_timestamp()
main_monthly_consumption.index = main_monthly_consumption["timestamp"]
# Monthly Aggregation for Reference Buildings
ref_monthly_consumption = (
    ref_buildings_df.groupby(
        ["property_id", ref_buildings_df["timestamp"].dt.to_period("M")]
    )["value"]
    .sum()
    .reset_index()
)
ref_monthly_consumption["timestamp"] = ref_monthly_consumption[
    "timestamp"
].dt.to_timestamp()

# %%
# Plot total consumption over time for the main building
plt.figure(figsize=(12, 6))
plt.plot(
    main_building_df["timestamp"],
    main_building_df["total_consumption"],
    label="Main Building",
)
plt.xlabel("Time")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Main Building Energy Consumption Over Time")
plt.legend()
plt.show()


# %%
# Plot average daily consumption

# plt.figure(figsize=(12, 6))
plt.figure(figsize=(18, 6))

sns.lineplot(data=main_daily_consumption, label="Main Building")
# plt.bar(main_daily_consumption.index, main_daily_consumption.values , width=4, label='Main Building', alpha=0.7)

for pid in ref_buildings:
    data = ref_daily_consumption[ref_daily_consumption["property_id"] == pid]
    sns.lineplot(x="timestamp", y="value", data=data, label=f"Building {pid}")
plt.xlabel("Date")
plt.ylabel("Daily Energy Consumption (kWh)")
plt.title("Daily Energy Consumption Comparison")
plt.legend()
plt.show()


# %%

x = main_weekly_consumption.index
y = main_weekly_consumption["value"]


plt.figure(figsize=(12, 6))
plt.bar(x, y, width=4, label="Main Building", alpha=0.7)

colors = ["red", "green", "orange", "purple", "brown", "pink"]  # Add as many as needed

for i, pid in enumerate(ref_buildings):
    data = ref_weekly_consumption[ref_weekly_consumption["property_id"] == pid]
    plt.plot(
        data["timestamp"],
        data["value"],
        label=f"Building {get_name(pid)}",
        color=colors[i % len(colors)],
    )

# Customize the plot
plt.xlabel("Timestamp")
plt.ylabel("kwh per uke")
plt.title("Main Building Production vs. Reference Buildings")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# %%

x = main_monthly_consumption.index
y = main_monthly_consumption["value"]

assert len(x) == len(y)


plt.figure(figsize=(12, 6))
plt.bar(x, y, width=10, label="Main Building", alpha=0.7)

colors = ["red", "green", "orange", "purple", "brown", "pink"]  # Add as many as needed

for i, pid in enumerate(ref_buildings):
    data = ref_monthly_consumption[ref_monthly_consumption["property_id"] == pid]
    plt.plot(
        data["timestamp"],
        data["value"],
        label=f"Building {get_name(pid)}",
        color=colors[i % len(colors)],
    )

# Customize the plot
plt.xlabel("Timestamp")
plt.ylabel("kwh per måned")
plt.title("Main Building Production vs. Reference Buildings")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# %%
# Analyze correlation between temperature and energy consumption
plt.figure(figsize=(12, 6))
sns.scatterplot(x="temperature", y="total_consumption", data=main_building_df)
plt.xlabel("Temperature (°C)")
plt.ylabel("Total Energy Consumption (kWh)")
plt.title("Temperature vs Energy Consumption")
plt.show()


# %%
# det hadde kanskje gitt mer mening å se på korrelasjonen været og solforbruk / export.
correlation_columns = [
    "temperature",
    "wind_speed",
    "wind_direction",
    "cloud_fraction",
    "precipitation",
    "solar_consumption",
    "total_consumption",
]

correlation_matrix = main_building_df[correlation_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")

plt.title("Correlation Heatmap")
plt.show()


# %% [markdown]
# ## 3. Task 1: Analyze the Amount of Energy Exported

# %%
# Sum of exported energy for the main building
total_export_main = main_building_export["value"].sum()
print(f"Total Exported Energy for Main Building (kWh): {total_export_main:.2f}")

# Exported energy over time
main_export_daily = main_building_export.groupby(
    main_building_export["timestamp"].dt.date
)["value"].sum()

plt.figure(figsize=(12, 6))
plt.plot(main_export_daily.index, main_export_daily.values, label="Exported Energy")
plt.xlabel("Date")
plt.ylabel("Exported Energy (kWh)")
plt.title("Daily Exported Energy for Main Building")
plt.legend()
plt.show()


# %% [markdown]
# ## 4. Task 2: Compare Import and Export for the Buildings
#

# %% [markdown]
# ### 4.1 Visualize Import vs. Export

# %%
# Filter import and export data for all buildings
import_df = energy_df[energy_df["direction"] == "IMPORT"]
export_df = energy_df[energy_df["direction"] == "EXPORT"]

# List of all property IDs including main and reference buildings
all_buildings = [10724, 10703, 4462, 4746]

# Total import and export per building
import_total = (
    import_df.groupby("property_id")["value"].sum().reset_index(name="total_import")
)
export_total = (
    export_df.groupby("property_id")["value"].sum().reset_index(name="total_export")
)

# Merge import and export totals
import_export_total = pd.merge(import_total, export_total, on="property_id", how="left")
import_export_total["total_export"].fillna(0, inplace=True)

# Display the totals
print(import_export_total)
# Plot import and export comparison
import_export_total.set_index("property_id", inplace=True)
import_export_total[["total_import", "total_export"]].plot(kind="bar", figsize=(10, 6))
plt.xlabel("Property ID")
plt.ylabel("Energy (kWh)")
plt.title("Total Import and Export per Building")
plt.legend()
plt.show()


# %% [markdown]
# ### 4.2 Analyze Import Peaks (Which Lead to Higher Costs)
#
# #### 4.2.1 Identify Peak Import Hours for Each Building


# %%
def find_peak_import(import_df: pd.DataFrame) -> pd.DataFrame:
    # Note that it is not only about having a low energy consumption but also important to reduce the monthly max kW peak as a high peak will give high tariffs.
    # TODO: skal vi finne peak per time eller forbrukt for en dag eller en uke eller en måned?
    import_df["month"] = import_df["timestamp"].dt.to_period("M")
    peak_import = (
        import_df.groupby(["property_id", "month"])["value"].max().reset_index()
    )
    return peak_import


peak_import_df = find_peak_import(import_df)


# %% [markdown]
# #### 4.2.2 Visualize Peak Import

# %%
# Plot peak import for each building
plt.figure(figsize=(12, 6))
for pid in all_buildings:
    data = peak_import_df[peak_import_df["property_id"] == pid]
    data = data[1:-1]  # skip the first and last value to avoid half weeks.
    plt.plot(
        data["month"].dt.to_timestamp(),
        data["value"],
        marker="o",
        label=f"{get_name(pid)}",
    )
plt.xlabel("Month")
plt.ylabel("Peak Import (kW)")
plt.title("Monthly Peak Import for Each Building")
plt.legend()
plt.show()


# %%
# Compare average peak imports
average_peaks = (
    peak_import_df.groupby("property_id")["value"]
    .mean()
    .reset_index(name="average_peak")
)
average_peaks["building"] = average_peaks["property_id"].apply(get_name)
print(average_peaks)


# %% [markdown]
# ### 5. Task 3: Calculate Solar Production as Solar Consumption Plus Export
# #### 5.1 Calculate Solar Production for the Main Building

# %%
# Solar production = solar consumption + export
# Sum solar consumption
total_solar_consumption = main_building_df["solar_consumption"].sum()

# Sum exported energy
total_solar_export = main_building_export["value"].sum()

# Total solar production
total_solar_production = total_solar_consumption + total_solar_export
print(f"Total Solar Production for Main Building (kWh): {total_solar_production:.2f}")


# %%
# Create a DataFrame for visualization
solar_components = pd.DataFrame(
    {
        "Component": ["Solar Consumption", "Solar Export"],
        "Energy (kWh)": [total_solar_consumption, total_solar_export],
    }
)

# Plotting the components
plt.figure(figsize=(8, 6))
sns.barplot(x="Component", y="Energy (kWh)", data=solar_components)
plt.title("Components of Solar Production for Main Building")
plt.ylabel("Energy (kWh)")
plt.show()


# %%
# Merge solar consumption and export data
solar_export = main_building_export[["timestamp", "value"]].rename(
    columns={"value": "solar_export"}
)
solar_production = pd.merge(
    main_building_df[["timestamp", "solar_consumption"]],
    solar_export,
    on="timestamp",
    how="outer",
)
solar_production.fillna(0, inplace=True)
solar_production["total_solar_production"] = (
    solar_production["solar_consumption"] + solar_production["solar_export"]
)

# Set timestamp as index
solar_production.set_index("timestamp", inplace=True)

# Resample to daily totals
solar_daily_production = solar_production.resample("D").sum()

# Plot daily solar production
plt.figure(figsize=(12, 6))
plt.plot(
    solar_daily_production.index,
    solar_daily_production["total_solar_production"],
    label="Total Solar Production",
)
plt.plot(
    solar_daily_production.index,
    solar_daily_production["solar_consumption"],
    label="Solar Consumption",
)
plt.plot(
    solar_daily_production.index,
    solar_daily_production["solar_export"],
    label="Solar Export",
)
plt.xlabel("Date")
plt.ylabel("Energy (kWh)")
plt.title("Daily Solar Production Components for Main Building")
plt.legend()
plt.show()


# %% [markdown]
# ### 6. energy import per square meter
# #### 6.1 Normalize Energy Consumption by Building Area

# %%
# Building areas (in m²)
building_areas = {10724: 1199, 10703: 1167, 4462: 1095, 4746: 1384}  # Main building

# Add building areas to import_export_total DataFrame
import_export_total["area_m2"] = import_export_total.index.map(building_areas)

# Calculate energy consumption per m²
import_export_total["import_per_m2"] = (
    import_export_total["total_import"] / import_export_total["area_m2"]
)
import_export_total["export_per_m2"] = (
    import_export_total["total_export"] / import_export_total["area_m2"]
)

# Display normalized values
import_export_total[["total_import", "import_per_m2", "total_export", "export_per_m2"]]


# %%
import_export_total[["import_per_m2"]].plot(kind="bar", figsize=(10, 6))
plt.xlabel("Property ID")
plt.ylabel("Energy Consumption per m² (kWh/m²)")
plt.title("Normalized Energy Consumption per Building")
plt.legend()
plt.show()


# %% [markdown]
# ### 7. Correlate Import Peaks with Meteorological Data
# #### 7.1 Merge Import Data with Meteorological Data

# %%
# Merge import data with meteorological data
import_met_df = pd.merge(
    import_df,
    met_df,
    left_on=["property_id", "timestamp"],
    right_on=["property_id", "timestamp"],
    how="left",
)

# For main building only
main_import_met = import_met_df[import_met_df["property_id"] == 10724]

# Merge import data with meteorological data
import_met_df = pd.merge(
    import_df,
    met_df,
    left_on=["property_id", "timestamp"],
    right_on=["property_id", "timestamp"],
    how="left",
)

# For main building only
main_import_met = import_met_df[import_met_df["property_id"] == 10724]


# %%
# Extract peak import times for main building
main_peaks = main_import_met.groupby(
    main_import_met["timestamp"].dt.to_period("M")
).apply(lambda x: x.loc[x["value"].idxmax()])

# Plot temperature vs peak import
plt.figure(figsize=(8, 6))
sns.scatterplot(x="temperature", y="value", data=main_peaks)
plt.xlabel("Temperature (°C)")
plt.ylabel("Peak Import (kW)")
plt.title("Temperature vs Peak Import for Main Building")
plt.show()
