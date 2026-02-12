import plotly.graph_objects as go
import numpy as np

def create_scatter_plot(data, range_val, xcenter, ycenter, show_names=False, use_webgl=True, highlight_idx=None):
    """Create scatter plot with optional names using WebGL if available"""
    fig = go.Figure()
    
    # Scale the size of the data points inversely by the mag value
    marker_size = [np.exp(-(mag-19.0)/2.5)*10.0 for mag in data['mag']]
    
    # Create colors array, yellow for highlighted star
    colors = ['LightSkyBlue'] * len(data['x'])
    if highlight_idx is not None:
        colors[highlight_idx] = '#ffff00'  # Bright yellow
    
    # Add scatter plot
    scatter_kwargs = {
        'x': data['x'],
        'y': data['y'],
        'mode': 'markers+text' if show_names else 'markers',
        'name': 'Stars',
        'text': data['name'] if show_names else None,  # Text label next to the points
        'hovertext': [f"Name: {name}<br>X: {x}<br>Y: {y}<br>Mag: {mag}" for name, x, y, mag in zip(data['name'], data['x'], data['y'], data['mag'])],  # Hover info
        'hoverinfo': 'text',
        'marker': {
            'size': marker_size,
            'color': colors,
            'opacity': 0.7,
            'line': {
                'width': [2 if i == highlight_idx else 0 for i in range(len(data['x']))],
                'color': 'white'
            }
        }
    }
    
    if show_names:
        scatter_kwargs['textposition'] = 'top center'
    
    if use_webgl:
        fig.add_trace(go.Scattergl(**scatter_kwargs))
    else:
        fig.add_trace(go.Scatter(**scatter_kwargs))
    
    # Set initial 1:1 layout
    fig.update_layout(
        height=800,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
            range=[xcenter + range_val, xcenter - range_val]
            #autorange='reversed'  # Flip the x-axis
        ),
        yaxis=dict(
            title='Y Position (arcsec)',
            range=[ycenter - range_val, ycenter + range_val]  # Set y-axis range
        )
    )
    return fig

def update_layout(fig, range_val, xcenter, ycenter):
    """Update plot layout with 1:1 aspect ratio"""
    fig.update_layout(
        title='Star Positions',
        xaxis=dict(
            title='X Position (arcsec)',
            scaleanchor='y',
            scaleratio=1,
            range=[xcenter + range_val, xcenter - range_val],
            #autorange='reversed'  # Flip the x-axis
        ),
        yaxis=dict(
            title='Y Position (arcsec)',
            range=[ycenter - range_val, ycenter + range_val]  # Set y-axis range
        ),
        #height=800,  # Fixed height
        margin=dict(l=50, r=50, t=50, b=50)  # Add margins
    )
    return fig

def center_plot_on_coordinates(fig, coords, range_val):
    """Center the plot on given coordinates"""
    if coords['found']:
        fig.update_layout(
            xaxis=dict(
                scaleanchor='y',
                scaleratio=1,
                range=[coords['x'] + range_val, coords['x'] - range_val]
            ),
            yaxis=dict(
                range=[coords['y'] - range_val, coords['y'] + range_val]
            )
        )
    return fig

def highlight_star(fig, star_coords, range_val):
    """Center the plot on a star"""
    return center_plot_on_coordinates(fig, star_coords, range_val)