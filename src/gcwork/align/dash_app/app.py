import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from data_loader import load_data, load_orbits, find_star_coordinates
from plot_utils import create_scatter_plot, update_layout, center_plot_on_coordinates, highlight_star
import json
import plotly.graph_objects as go  # Import plotly.graph_objects
import numpy as np  # Import numpy

app = dash.Dash(__name__, 
    suppress_callback_exceptions=True,
    title='GCG Atlas'  # This sets the browser tab title
)

# Add password (in practice, use environment variables or secure storage)
VALID_PASSWORD = "gcg2024"

# Create login layout
login_layout = html.Div([
    html.H1("GCG Atlas"),
    html.Div([
        dcc.Input(
            id='password-input',
            type='password',
            placeholder='Enter password',
            n_submit=0
        ),
        html.Button('Login', id='login-button', n_clicks=0),
        html.Div(id='login-status', style={'color': 'red'})
    ], style={'padding': '20px'})
])

# Store the main layout in a variable
main_layout = html.Div([
    html.H1("GCG Atlas"),
    html.P("2024-12-20 - Tuan Do - Created"),
    html.Div([
        html.Div([
            html.Label("Align Root:"),
            dcc.Input(id='align-root', value='/u/ghezgroup/data/papers/gr2018/18_10_02/align/align_d_rms_1000_abs', type='text'),
            
            html.Label("Range:"),
            dcc.Input(id='range', value=0.4, type='number'),
            
            html.Label("Center (x, y):"),
            dcc.Input(id='xcenter', value=0, type='number'),
            dcc.Input(id='ycenter', value=0, type='number'),
            html.Br(),

            dcc.Checklist(id='show-names', options=[{'label': 'Show Names', 'value': True}], value=[True]),
            
            html.Label("Orbits File:"),
            dcc.Input(id='orbits-file', value='/u/tdo/research/source_list/orbits.dat', type='text'),
            
            dcc.Checklist(
                id='plot-orbits', 
                options=[{'label': 'Plot Orbits', 'value': True}],
                value=[True]  # This sets the default to checked
            ),
            
            html.Button('Load Data', id='load-data-button', n_clicks=0),
            html.Button('Update Plot', id='update-plot-button', n_clicks=0),
            html.Div(id="loading-output", style={'padding-top': '10px'})
        ], style={'padding': '10px'})
    ]),
    html.Div([
        html.Label("Search Star:"),
        dcc.Input(
            id='search-input',
            type='text',
            placeholder='Enter star name',
            n_submit=0  # Add this to enable Enter key submissions
        ),
        html.Button('Search', id='search-button', n_clicks=0),
    ], style={'padding': '10px'}),
    dcc.Loading(id="loading", type="default", children=[dcc.Store(id='stored-data', storage_type='memory'), dcc.Store(id='stored-orbits', storage_type='memory')]),
    dcc.Graph(id='star-plot', config={'scrollZoom': True, 'modeBarButtonsToRemove': ['zoom2d'], 'displayModeBar': True, 'doubleClick': 'reset+autosize', 'modeBarButtonsToAdd': ['pan2d'], 'displaylogo': False}, style={'width': '100%', 'height': '80vh'})
    
])

# Initialize with login layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='authentication-status', storage_type='session'),
    html.Div(id='page-content', children=login_layout)  # Show login layout by default
])

# Add authentication callback
@app.callback(
    [Output('page-content', 'children'),
     Output('authentication-status', 'data'),
     Output('login-status', 'children')],
    [Input('login-button', 'n_clicks'),
     Input('password-input', 'n_submit')],
    [State('password-input', 'value'),
     State('authentication-status', 'data')]
)
def authenticate(n_clicks, n_submit, password, auth_status):
    # Don't trigger on page load
    if n_clicks is None and n_submit is None:
        return login_layout, None, ''
        
    # Check if already authenticated
    if auth_status == 'authenticated':
        return main_layout, auth_status, ''
    
    # Handle login attempt
    if password:
        if password == VALID_PASSWORD:
            return main_layout, 'authenticated', ''
        return login_layout, 'unauthenticated', 'Incorrect password'
    
    # Default to login page
    return login_layout, 'unauthenticated', ''

# Modify existing callback to check authentication
@app.callback(
    Output('loading-output', 'children'),
    Output('stored-data', 'data'),
    Output('stored-orbits', 'data'),
    Output('star-plot', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('update-plot-button', 'n_clicks'),
     Input('search-button', 'n_clicks'),
     Input('search-input', 'n_submit')],  # Add this input
    [State('align-root', 'value'),
     State('range', 'value'),
     State('xcenter', 'value'),
     State('ycenter', 'value'),
     State('show-names', 'value'),
     State('orbits-file', 'value'),
     State('plot-orbits', 'value'),
     State('stored-data', 'data'),
     State('stored-orbits', 'data'),
     State('search-input', 'value'),
     State('star-plot', 'figure'),
     State('authentication-status', 'data')]
)
def handle_callbacks(load_n_clicks, update_n_clicks, search_n_clicks, search_submit,
                    align_root, range_val, xcenter, ycenter, 
                    show_names, orbits_file, plot_orbits, 
                    stored_data, stored_orbits, search_value, current_figure, auth_status):
    if auth_status != 'authenticated':
        return "", None, None, {}

    ctx = dash.callback_context
    if not ctx.triggered:
        return "", None, None, {}

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Treat both search button click and search input submit as search triggers
    if button_id in ['search-button', 'search-input']:
        if not search_value or not stored_data:
            return "", stored_data, stored_orbits, current_figure
        
        data = json.loads(stored_data)
        coords = find_star_coordinates(data, search_value)
        
        if coords['found']:
            # Create new plot with highlighted star
            highlight_idx = data['name'].index(search_value)
            fig = create_scatter_plot(data, range_val, xcenter, ycenter, 
                                   show_names=show_names, use_webgl=True, 
                                   highlight_idx=highlight_idx)
            
            # Add orbits if they exist
            if stored_orbits:
                orbits_data = json.loads(stored_orbits)
                if orbits_data and plot_orbits:
                    for orbit, name in zip(orbits_data['orbits'], orbits_data['names']):
                        orbit = np.array(orbit)
                        fig.add_trace(go.Scatter(
                            x=orbit[:, 0],
                            y=orbit[:, 1],
                            mode='lines',
                            name=f'Orbit: {name}'
                        ))
            
            # Add highlighting and animation
            fig = highlight_star(fig, coords, range_val)
            return f"Found star '{search_value}'", stored_data, stored_orbits, fig
            
        return f"Star '{search_value}' not found", stored_data, stored_orbits, current_figure
    
    elif button_id == 'load-data-button':
        try:
            # Load data with proper unpacking
            star_data, from_cache = load_data(align_root, range=range_val, xcenter=xcenter, ycenter=ycenter)
            loading_message = "Loading star data from cache..." if from_cache else "Loading star data from source..."
            
            if star_data is None:
                return "Error loading data", None, None, {}
            
            # Load orbits
            orbits_data = None
            if orbits_file and plot_orbits:
                orbits_result, orbit_names, orbits_from_cache = load_orbits(orbits_file)
                loading_message += "\nLoading orbit data from " + ("cache..." if orbits_from_cache else "source...")
                if orbits_result is not None:
                    orbits_data = {'orbits': orbits_result, 'names': orbit_names}
            
            # Create plot
            loading_message += "\nCreating plot..."
            fig = create_scatter_plot(star_data, range_val, xcenter, ycenter, show_names=show_names, use_webgl=True)
            fig = update_layout(fig, range_val, xcenter, ycenter)
            fig.update_layout(dragmode='pan')
            
            # Add orbits to plot
            if orbits_data:
                for orbit, name in zip(orbits_data['orbits'], orbits_data['names']):
                    orbit = np.array(orbit)
                    fig.add_trace(go.Scatter(
                        x=orbit[:, 0],
                        y=orbit[:, 1],
                        mode='lines',
                        name=f'Orbit: {name}'
                    ))
            
            return loading_message + "\nData loaded successfully", json.dumps(star_data), json.dumps(orbits_data), fig
            
        except Exception as e:
            return f"Error: {str(e)}", None, None, {}
    
    elif button_id == 'update-plot-button':
        if stored_data is None:
            return "", None, None, {}
        
        # Load data from stored data
        data = json.loads(stored_data)
        
        # Load orbits from stored data
        orbits_data = json.loads(stored_orbits) if stored_orbits else None
        
        # Create plot
        use_webgl = True  # Set this to False if WebGL is not supported
        fig = create_scatter_plot(data, range_val, xcenter, ycenter, show_names=show_names, use_webgl=use_webgl)
        fig = update_layout(fig, range_val, xcenter, ycenter)
        fig.update_layout(
        dragmode='pan')
        
        # Add orbits to plot
        if orbits_data and plot_orbits:
            for orbit, name in zip(orbits_data['orbits'], orbits_data['names']):
                orbit = np.array(orbit)  # Convert list back to numpy array
                fig.add_trace(go.Scatter(
                    x=orbit[:, 0],
                    y=orbit[:, 1],
                    mode='lines',
                    name=f'Orbit: {name}'
                ))
        
        return "", stored_data, stored_orbits, fig

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)