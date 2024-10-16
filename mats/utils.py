import pandas as pd
import plotly.graph_objects as go


def plot_energy_usage(df_display: pd.DataFrame) -> None:
    """
    Plots the actual vs forecasted energy usage with fill areas, range slider,
    and range selector.

    Parameters:
    df_display (pd.DataFrame): DataFrame with 'actual' and 'prediction' columns. The index should be timestamps.

    Returns:
    None: Shows the plotly figure.
    """
    df_display = pd.DataFrame(df_display.copy()[["actual", "prediction", "timestamp"]])
    df_display.set_index("timestamp", inplace=True)
    # Create the figure
    fig = go.Figure()

    # Add trace for actual energy usage
    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=df_display["actual"],
            mode="lines",
            name="Actual Energy (kWh)",
            line=dict(color="rgba(255, 99, 71, 0.9)"),
        )
    )

    # Add trace for predicted energy usage
    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=df_display["prediction"],
            mode="lines",
            name="Forecasted Energy (kWh)",
            line=dict(color="rgba(30, 144, 255, 0.9)"),
        )
    )

    # Add fill between actual energy usage and baseline (y=0)
    fig.add_trace(
        go.Scatter(
            x=df_display.index.tolist(),
            y=df_display["actual"].tolist(),
            fill="tozeroy",  # Fill to the x-axis (y=0)
            fillcolor="rgba(255, 160, 122, 0.3)",  # Lighter pastel red with opacity 0.3
            opacity=0.3,
            line=dict(color="rgba(255, 99, 71, 0)"),  # Hide the fill boundary
            showlegend=False,
        )
    )

    # Add fill between forecasted energy usage and baseline (y=0)
    fig.add_trace(
        go.Scatter(
            x=df_display.index.tolist(),
            y=df_display["prediction"].tolist(),
            fill="tozeroy",  # Fill to the x-axis (y=0)
            fillcolor="rgba(135, 206, 250, 0.3)",  # Lighter pastel blue with opacity 0.3
            opacity=0.3,
            line=dict(color="rgba(30, 144, 255, 0)"),  # Hide the fill boundary
            showlegend=False,
        )
    )

    # Add layout details
    fig.update_layout(
        title="Actual vs Forecasted Energy Usage",
        xaxis_title="Date",
        yaxis_title="Energy Usage (kWh)",
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

    # Show the plot
    fig.show()
