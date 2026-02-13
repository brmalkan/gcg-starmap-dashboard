from dash import Dash, Input, Output, State
import numpy as np
import plotly.graph_objects as go

from data_loader import (
    find_neighbor_stars,
    load_starset_data,
    load_orbits
)
from gcwork.starset import StarSet
from layout import app_layout


class Dashboard:

    def __init__(self) -> None:
        """
        Initialize app and load layout. Dark mode css
        themes extracted from assets directory
        """

        self.app: Dash = Dash(assets_folder="../assets")
        self.app.layout = app_layout()

        self.star_data: None | StarSet = None
        self.orbit_data: None | tuple[list[any], list[any]] = None

    def start(self) -> None:

        @self.app.callback(
            Output("star_map_2d", "figure"),
            # Output("star_map_3d", "figure"),
            Output("star_reference_list", "options"),
            Input("refresh_button", "n_clicks"),
            [State("data_filepath", "value"),
            State("orbit_filepath", "value"),
            State("map_range", "value"),
            State("map_center_x", "value"),
            State("map_center_y", "value"),
            State("enable_orbits", "on"),
            State("enable_names", "on")],
        )
        def update_starmap(
            _refresh_button: int,
            data_filepath: str,
            orbit_filepath: str,
            range: float,
            center_x: float,
            center_y: float,
            enable_orbits: bool,
            enable_names: bool
        ) -> None:
            if not data_filepath:
                return go.Figure() # TODO Add Error

            self.star_data = load_starset_data(data_filepath)
            if enable_orbits:
                self.orbit_data = load_orbits(orbit_filepath)

            hover_text = [
                f"Name: {name}<br>X: {x}<br>Y: {y}<br>Mag: {mag}"
                for name, x, y, mag in zip(
                    self.star_data['name'],
                    self.star_data['x'],
                    self.star_data['y'],
                    self.star_data['mag']
                )
            ]
            marker_size = [
                np.exp(-(mag-19.0)/2.5)*10.0
                for mag in self.star_data['mag']
            ]
            marker_color = [
                "red" if name == "SgrA" else "cyan"
                for name in self.star_data["name"]
            ]
            marker_symbol = [
                "star" if name == "SgrA" else "circle"
                for name in self.star_data["name"]
            ]
            marker_opacity = [
                1 if name == "SgrA" else 0.7
                for name in self.star_data["name"]
            ]

            fig_2d = go.Figure()
            fig_2d.add_trace(
                go.Scattergl(
                    x=self.star_data["x"],
                    y=self.star_data["y"],
                    customdata=self.star_data["name"],
                    mode="markers+text" if enable_names else "markers",
                    text=self.star_data["name"] if enable_names else None,
                    textposition="top center",
                    textfont={
                        "size": 8,
                        "color": "grey",
                        "family": "Times New Roman"
                    },
                    hovertext=hover_text,
                    hoverinfo="text",
                    marker={
                        "size": marker_size,
                        "color": marker_color,
                        "symbol": marker_symbol,
                        "opacity": marker_opacity,
                        "line": {
                            "color": "white",
                            "width": 1
                        }
                    }
                )
            )
            fig_2d.update_layout(
                template="plotly_dark",
                clickmode="event",
                dragmode="pan",
                xaxis={
                    "scaleanchor": "y",
                    "scaleratio": 1,
                    "range": [center_x - range, center_x + range],
                    "color": "white",
                    "gridcolor": "rgba(0, 0, 0, 0.3)"
                },
                yaxis={
                    "range": [center_y - range, center_y + range],
                    "color": "white",
                    "gridcolor": "rgba(0, 0, 0, 0.3)"
                },
                showlegend=False,
                margin={
                    "t": 20,
                    "b": 20,
                    "r": 20,
                    "l": 20
                },
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            fig_3d = go.Figure()
            fig_3d.add_trace(
                go.Scatter3d(

                )
            )

            if enable_orbits and orbit_filepath:
                for orbit, name in zip(self.orbit_data[0], self.orbit_data[1]):
                    orbit = np.array(orbit)
                    fig_2d.add_trace(
                        go.Scatter(
                            x=orbit[:, 0],
                            y=orbit[:, 1],
                            mode="lines",
                            opacity=0.4,
                            hovertext=name,
                            hoverinfo="text"
                        )
                    )

            return fig_2d, self.star_data["name"]

        @self.app.callback(
            Output("star_reference_list", "value"),
            Input("star_map_2d", "clickData"),
            prevent_initial_call=True
        )
        def update_neighbor_reference(star_data: dict[any]) -> None:
            star_name = star_data["points"][0]["customdata"]
            return star_name

        @self.app.callback(
            Output("neighbor_table", "data"),
            Input("star_reference_list", "value"),
            prevent_initial_call=True
        )
        def update_neighbor_table(star_name: str) -> None:
            table_data = find_neighbor_stars(self.star_data, star_name)
            return table_data

        self.app.run(
            port=8050,
            dev_tools_silence_routes_logging=True,
            debug=False
        )
