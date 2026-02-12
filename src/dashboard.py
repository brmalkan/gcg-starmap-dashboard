from dash import Dash, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from data_loader import load_starset_data, load_orbits
from layout import app_layout


class Dashboard:

    def __init__(self) -> None:
        """
        Initialize app and load layout. Dark mode css
        themes extracted from assets directory
        """

        self.app: Dash = Dash(assets_folder="../assets")
        self.app.layout = app_layout()

    def start(self) -> None:

        @self.app.callback(
            Output("star_map_2d", "figure"),
            Output("star_map_3d", "figure"),
            Input("refresh_button", "n_clicks"),
            State("data_filepath", "value"),
            State("orbit_filepath", "value"),
            State("map_range", "value"),
            State("map_center_x", "value"),
            State("map_center_y", "value"),
            State("map_radio_options", "value")
        )
        def update_starmap(
            _refresh_button: int,
            data_filepath: str,
            orbit_filepath: str,
            range: float,
            center_x: float,
            center_y: float,
            radio_options: list[bool],
        ) -> None:
            if not data_filepath:
                return go.Figure() # TODO Add Error

            star_data = load_starset_data(data_filepath)
            show_name = True

            fig_2d = go.Scatter(
                x=star_data["x"],
                y=star_data["y"],
                xaxis={
                    "range": [center_x + range, center_x - range]
                },
                yaxis={
                    "range": [center_y + range, center_y - range]
                },
                mode="markers+text" if show_name else "text",
                text=star_data["name"] if show_name else None,
                marker={
                    "size": 1,
                    "color": "cyan",
                    "opacity": 0.7
                }
            )
            fig_3d = go.Scatter3d()

            if orbit_filepath:
                orbit_data = load_orbits(orbit_filepath)

            return fig_2d, None

        self.app.run(port=8050)
