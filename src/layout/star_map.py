from dash import dash_table, dcc, html
import dash_bootstrap_components as dbc
import dash_daq as daq
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
                                value="/Users/dev/Desktop/galactic_center/gc_map_app/data/align_d_rms_1000_abs",
                                type="text",
                                id="data_filepath"
                            )
                        ]),

                        html.Div([
                            html.H6("Orbit Data File"),
                            dcc.Input(
                                value="/Users/dev/Desktop/galactic_center/gc_map_app/data/orbits.dat",
                                type="text",
                                id="orbit_filepath"
                            )
                        ]),

                        html.Div([
                            html.H6("Range (arcseconds)"),
                            dcc.Input(
                                value=0.3,
                                step=0.1,
                                type="number",
                                id="map_range"
                            )
                        ]),

                        html.Div([
                            html.H6("Center Point"),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Input(
                                        value=0,
                                        step=0.1,
                                        type="number",
                                        id="map_center_x"
                                    ),
                                    html.Label("x (arcseconds)")
                                ]),
                                dbc.Col([
                                    dcc.Input(
                                        value=0,
                                        step=0.1,
                                        type="number",
                                        id="map_center_y"
                                    ),
                                    html.Label("y (arcseconds)")
                                ]),
                            ])
                        ]),

                        html.Br(),

                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Enable Orbits")
                                ], width=9),
                                dbc.Col([
                                    daq.BooleanSwitch(
                                        id="enable_orbits",
                                        on=True,
                                        color="#20c997"
                                    ),
                                ], width=3)
                            ])
                        ]),

                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Enable Star Names")
                                ], width=9),
                                dbc.Col([
                                    daq.BooleanSwitch(
                                        id="enable_names",
                                        on=True,
                                        color="#20c997"
                                    ),
                                ], width=3)
                            ])
                        ]),

                        html.Br(),

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
                                    options=[],
                                    id="star_reference_list"
                                )
                            ]),
                            dash_table.DataTable(
                                columns=[
                                    {"name": "Star", "id": "star"},
                                    {"name": "Distance (arcseconds)", "id": "distance"},
                                ],
                                data=[],
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
                                },
                                style_table={
                                    "height": "50vh"
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
                            style={
                                "height": "83vh"
                            },
                            config={
                                'displayModeBar': False,
                                # "scrollZoom": True
                            }
                        ),
                        # dcc.Graph(
                        #     figure=px.scatter_3d(
                        #         px.data.iris(),
                        #         x='sepal_length',
                        #         y='sepal_width',
                        #         z='petal_width',
                        #         color="species",
                        #         hover_data=['petal_width'],
                        #     ).update_layout(
                        #         template='plotly_dark',
                        #         plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        #         paper_bgcolor= 'rgba(0, 0, 0, 255)',
                        #     ),
                        #     id="star_map_3d",
                        #     config={
                        #         'displayModeBar': False,
                        #         "scrollZoom": True
                        #     }
                        # )
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
