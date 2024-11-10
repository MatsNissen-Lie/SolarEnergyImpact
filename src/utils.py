import os
import pandas as pd
import plotly.graph_objects as go

column_param: list[dict[str, str]] = [
    {
        "name": "Actual Energy (kWh)",
        "column": "actual",
    },
]
COLORS = [
    "rgba(255, 90, 71, 0.9)",
    "rgba(30, 144, 255, 0.9)",
    "rgba(20, 208, 104, 0.9)",  # Light Green
    "rgba(120, 0, 128, 0.9)",  # Purplen
    # "rgba(250, 250, 210, 0.9)",  # Light Goldenrod Yellow
    "rgba(255, 182, 193, 0.9)",  # Light Pin
]
COLORS_DARK = [
    "rgba(255, 0, 0, 0.9)",  # Red
    "rgba(0, 0, 255, 0.9)",  # Blue
    "rgba(0, 128, 0, 0.9)",  # Green
    "rgba(255, 165, 0, 0.9)",  # Orange
    "rgba(128, 0, 128, 0.9)",  # Purplen
    # "rgba(0, 191, 255, 0.9)",  # Deep Sky Blue
]

FILL_COLORS = [
    "rgba(255, 99, 71, 0.3)",
    "rgba(30, 144, 255, 0.3)",
    "rgba(144, 238, 144, 0.3)",  # Light Green
    "rgba(255, 182, 193, 0.3)",  # Light Pin
    "rgba(250, 250, 210, 0.3)",  # Light Goldenrod Yellow
]


class ColumnParam:
    def __init__(self, column: str, name: str):
        self.name = name
        self.column = column


def plot_energy_usage(
    df_display: pd.DataFrame,
    column_params: list[ColumnParam] = [ColumnParam("Actual Energy (kWh)", "actual")],
    titel="Actual vs Forecasted Energy Usage",
    yaxis_title="Energy Usage (kWh)",
    tozeroy=True,
    dark_mode=False,
) -> None:
    # df_display = pd.DataFrame(df_display.copy()[["actual", "prediction", "timestamp"]])
    if "timestamp" not in df_display.columns:
        raise ValueError("timestamp column not found in the DataFrame.")

    df_display.set_index("timestamp", inplace=True)
    # Create the figure
    fig = go.Figure()

    # Plot each column
    for idx, param in enumerate(column_params):
        name = param.name
        column = param.column
        if column not in df_display.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        # Get the color for this column
        if dark_mode:
            color = COLORS_DARK[idx % len(COLORS_DARK)]
        else:
            color = COLORS[idx % len(COLORS)]
        # fill_color = fill_colors[idx % len(fill_colors)]

        # Add trace for the column
        fig.add_trace(
            go.Scatter(
                x=df_display.index,
                y=df_display[column],
                mode="lines",
                name=name,
                line=dict(color=color),
            )
        )

        # Add fill area if tozeroy is True
        if tozeroy:
            # color make the opacity of the fill area 0.3
            fill_color = color[:-4] + "0.3)"
            fig.add_trace(
                go.Scatter(
                    x=df_display.index.tolist(),
                    y=df_display[column].tolist(),
                    fill="tozeroy",
                    fillcolor=fill_color,
                    opacity=0.3,
                    line=dict(color="rgba(0,0,0,0)"),  # Hide the fill boundary
                    showlegend=False,
                )
            )

    # Add layout details
    fig.update_layout(
        title=titel,
        xaxis_title="Datetime",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        showlegend=True,
        height=600,
    )

    # Add range slider and range selector for zoom functionality
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            rangeslider=dict(visible=True),
            type="date",
        )
    )
    fig.show()

    # output_file = True
    # if output_file:
    #     # Ensure kaleido is installed
    #     output_file = titel.replace(" ", "_").lower()
    #     output_file = f"images/{output_file}.png"
    #     # if the file exists add 1 or more to the end of the file name
    #     if not os.path.exists(f"{output_file}.png"):
    #         try:
    #             fig.write_image("output.png", engine="kaleido")
    #             print(f"Figure saved as {output_file}")
    #         except Exception as e:
    #             print(f"An error occurred while saving the figure: {e}")
    # Otherwise, display the interactive figure
