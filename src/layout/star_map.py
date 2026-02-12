from dash import dash_table, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px

CARD_HEIGHT = "87vh"

def draw_star_map() -> html.Div:
    return html.Div([
        dbc.Card(
            dbc.CardBody(
                children=[
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[draw_config_box()],
                                width=3
                            ),
                            dbc.Col(
                                children=[EXAMPLE()],
                                width=6
                            ),
                            dbc.Col(
                                children=[draw_neighbor_table()],
                                width=3
                            )
                        ],
                        className="dbc",
                    )
                ],
                style={
                    "height": "100%"
                }
            ),
            style={
                "margin-top": 25
            }
        )
    ])

def draw_config_box() -> html.Div:
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                dbc.Stack(
                    [
                        html.H3(
                            children="Configuration",
                            style={
                                "textAlign": "center",
                                "color": "white"
                            }
                        ),

                        html.Div([
                            html.H6("Root Data Directory"),
                            dcc.Input(
                                type="text",
                                id="data_filepath"
                            )
                        ]),

                        html.Div([
                            html.H6("Orbit Data File"),
                            dcc.Input(
                                type="text",
                                id="orbit_filepath"
                            )
                        ]),

                        html.Div([
                            html.H6("Range (arcseconds)"),
                            dcc.Input(
                                type="number",
                                id="map_range"
                            )
                        ]),

                        html.Div([
                            html.H6("Center Point"),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Input(
                                        value=0, type="number", id="map_center_x"
                                    ),
                                    html.Label("x (arcseconds)")
                                ]),
                                dbc.Col([
                                    dcc.Input(
                                        value=0, type="number", id="map_center_y"
                                    ),
                                    html.Label("y (arcseconds)")
                                ]),
                            ])
                        ]),

                        dcc.RadioItems(
                            options=[
                                "Enable Names",
                                "Enale Orbits"
                            ],
                            id="map_radio_options"
                        ),

                        dbc.Button(
                            "Refresh Map",
                            id="refresh_button",
                            color="info"
                        )
                    ],
                    gap=4
                )
            ]),
            color="dark",
            style={
                "height": CARD_HEIGHT
            }
        )
    ])

def draw_star_plot() -> html.Div:
    return html.Div()

def draw_neighbor_table() -> html.Div:
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    dbc.Stack(
                        [
                            html.H3(
                                children="Neighboring Stars",
                                style={
                                    "textAlign": "center",
                                    "color": "white"
                                }
                            ),
                            html.Div([
                                html.H6(children="Select Reference Star"),
                                dcc.Dropdown(
                                    options=['NYC', 'MTL', 'SF'],
                                    value='NYC'
                                )
                            ]),
                            dash_table.DataTable(
                                columns=[
                                    {"name": "Star", "id": "star"},
                                    {"name": "Distance", "id": "distance"},
                                ],
                                data = [
                                    {"star": "Alpha", "distance": 1.2},
                                    {"star": "Beta", "distance": 3.4},
                                    {"star": "Gamma", "distance": 2.1},
                                    {"star": "Delta", "distance": 4.8},
                                    {"star": "Epsilon", "distance": 0.9},
                                    {"star": "Zeta", "distance": 5.6},
                                    {"star": "Eta", "distance": 3.2},
                                    {"star": "Theta", "distance": 6.7},
                                    {"star": "Iota", "distance": 1.8},
                                    {"star": "Kappa", "distance": 2.9},
                                    {"star": "Lambda", "distance": 7.1},
                                    {"star": "Mu", "distance": 4.2},
                                    {"star": "Nu", "distance": 5.0},
                                    {"star": "Xi", "distance": 3.7},
                                    {"star": "Omicron", "distance": 6.3},
                                    {"star": "Pi", "distance": 2.4},
                                    {"star": "Rho", "distance": 1.5},
                                    {"star": "Sigma", "distance": 4.9},
                                    {"star": "Tau", "distance": 5.8},
                                    {"star": "Upsilon", "distance": 3.0},
                                    {"star": "Phi", "distance": 6.0},
                                    {"star": "Chi", "distance": 2.6},
                                    {"star": "Psi", "distance": 7.4},
                                    {"star": "Omega", "distance": 8.2},
                                    {"star": "Vega", "distance": 9.1},
                                ],
                                id="neighbor_table",
                                fixed_rows={
                                    "headers": True
                                },
                                style_header={
                                    "backgroundColor": "black",
                                    "fontWeight": "bold",
                                    "color": "#AAAAAA"
                                },
                                style_cell={
                                    "textAlign": "center"
                                }
                            )
                        ],
                        gap=4
                    )
                ])
            ]),
            color="dark",
            className="dbc dbc-ag-grid",
            style={
                "height": CARD_HEIGHT
            }
        ),
    ])

def EXAMPLE():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dbc.Stack(
                    [
                        dcc.Graph(
                            figure=px.bar(
                                px.data.iris(),
                                x="sepal_width",
                                y="sepal_length",
                                color="species"
                            ).update_layout(
                                template='plotly_dark',
                                plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                paper_bgcolor= 'rgba(0, 0, 0, 255)',
                            ),
                            id="star_map_2d",
                            config={
                                'displayModeBar': False
                            }
                        ),
                        dcc.Graph(
                            figure=px.scatter_3d(
                                px.data.iris(),
                                x='sepal_length',
                                y='sepal_width',
                                z='petal_width',
                                color="species",
                                hover_data=['petal_width'],
                            ).update_layout(
                                template='plotly_dark',
                                plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                paper_bgcolor= 'rgba(0, 0, 0, 255)',
                            ),
                            id="star_map_3d",
                            config={
                                'displayModeBar': False
                            }
                        )
                    ],
                    gap=0
                )
            ]),
            color="dark",
            style={
                "height": CARD_HEIGHT
            }
        ),
    ])
