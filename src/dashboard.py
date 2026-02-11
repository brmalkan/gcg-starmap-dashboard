from dash import Dash
import dash_bootstrap_components as dbc

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
        self.app.run(port=8050)
