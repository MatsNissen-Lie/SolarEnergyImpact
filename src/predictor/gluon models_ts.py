# %%
# %load_ext autoreload

# %%
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from autogluon.tabular import TabularPredictor

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", None)

# %%
# %autoreload
from pathlib import Path
import sys

root = Path().resolve().absolute().parent.parent
sys.path.append(str(root))

from pipeline import Pipeline, BuilingIdsEnum

pipe = Pipeline()


# %%
# time series data for buildings A, B, C
building_a = pipe.get_data(BuilingIdsEnum.A)
building_b = pipe.get_data(BuilingIdsEnum.B)
building_c = pipe.get_data(BuilingIdsEnum.C)
building_c


# %%
print(building_a.columns)

# %%
# Train the model using AutoGluon
predictor = TabularPredictor(label=target, eval_metric="mean_absolute_error").fit(
    train_data, presets="best_quality", excluded_model_types=["KNN"]
)

# %%
# Evaluate on test data
performance = predictor.evaluate(test_data)
# best model: ag-20241022_161331

print("Evaluation Performance:")
print(performance)  # This will show various metrics such as R^2, RMSE, etc.

# To see feature importance
global_importance = predictor.feature_importance(test_data)
print("\nFeature Importance:")
print(
    global_importance
)  # Shows which features had the most impact on model predictions

# %%
# model location => AutogluonModels/ag-20241016_095906
main_building = pipe.get_import_data_for_building(BuilingIdsEnum.MAIN.value)
main_building

data_predict = main_building[
    ["timestamp", "temperature", "area"]
]  #'wind_speed', 'wind_direction', 'cloud_fraction', 'precipitation'
# data_predict['timestamp'] = pd.to_datetime(data_predict['timestamp'])
data_predict


# %%
prediciton1 = predictor.predict(data_predict)

# %%
# save predicitons as a csv in data folder from root.
from pathlib import Path


prediciton1_df = pd.DataFrame(prediciton1)
my_path = Path().resolve().parent / "data" / "prediciton_3features.csv"
prediciton1_df.to_csv(my_path, index=False)
