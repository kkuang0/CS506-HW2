from flask import Flask
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import plotly.graph_objs as go
from kmeans import KMeans

app = Flask(__name__)
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/')

# Initialize kmeans globally
kmeans = None
manual_centers = []  # To store the manually selected centers

# Layout for the Dash app
dash_app.layout = html.Div([
    html.H1("KMeans Clustering Visualization"),
    dcc.Graph(id='cluster-plot'),

    html.Div([
        dcc.Dropdown(
            id='init-method',
            options=[
                {'label': 'Random', 'value': 'random'},
                {'label': 'Farthest First', 'value': 'farthest'},
                {'label': 'KMeans++', 'value': 'kmeans++'},
                {'label': 'Manual', 'value': 'manual'}
            ],
            value='kmeans++',
            placeholder="Select initialization method"
        ),
        html.Label("Number of Clusters:"),
        dcc.Slider(
            id='num-clusters',
            min=1,
            max=10,
            step=1,
            value=3,  # Default value for clusters
            marks={i: str(i) for i in range(1, 11)}
        ),
        html.Button('Generate New Dataset', id='generate-dataset', n_clicks=0),
        html.Button('Step', id='step-button', n_clicks=0),
        html.Button('Go to Convergence', id='converge-button', n_clicks=0),
        html.Button('Reset', id='reset-button', n_clicks=0),
        html.Div(id='manual-instructions', children=""),
    ], style={'display': 'inline-block', 'margin': '20px'}),

    html.Div(id='status-output', style={'marginTop': 20}),
])

# Callbacks
@dash_app.callback(
    [Output('cluster-plot', 'figure'),
     Output('status-output', 'children'),
     Output('manual-instructions', 'children')],
    [Input('init-method', 'value'),
     Input('num-clusters', 'value'),
     Input('generate-dataset', 'n_clicks'),
     Input('step-button', 'n_clicks'),
     Input('converge-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('cluster-plot', 'clickData')],
    [State('init-method', 'value')]
)
def update_plot(init_method, num_clusters, gen_dataset_clicks, step_clicks, converge_clicks, reset_clicks, clickData, current_method):
    global kmeans, manual_centers
    ctx = dash.callback_context
    triggered_by = ctx.triggered[0]['prop_id'].split('.')[0]

    status_text = ''
    manual_instructions = ""

    # Re-generate dataset if 'Generate New Dataset' is clicked
    if triggered_by == 'generate-dataset' or kmeans is None:
        kmeans = KMeans(k=num_clusters, init_method=init_method)
        kmeans.initialize()  # Explicitly generate a new dataset
        if kmeans.data is None:  # Ensure the data is generated
            status_text = "Error: Dataset not initialized."
        else:
            status_text = f"New dataset generated with {init_method} method and {num_clusters} clusters."
        manual_centers = []  # Reset manual centers for new dataset

    elif triggered_by == 'step-button':
        # Ensure that all manual centroids are selected before stepping
        if init_method == 'manual' and len(manual_centers) < kmeans.k:
            status_text = f"Please select {kmeans.k} manual centers first."
        else:
            if init_method == 'manual' and len(manual_centers) == kmeans.k and kmeans.centers is None:
                kmeans.centers = np.array(manual_centers)
            kmeans.step()  # Perform one step
            status_text = "Performed one step of KMeans."
            if kmeans.converged:
                status_text = "KMeans has converged!"

    elif triggered_by == 'converge-button':
        # Ensure that all manual centroids are selected before converging
        if init_method == 'manual' and len(manual_centers) < kmeans.k:
            status_text = f"Please select {kmeans.k} manual centers first."
        else:
            if init_method == 'manual' and len(manual_centers) == kmeans.k and kmeans.centers is None:
                kmeans.centers = np.array(manual_centers)
            while not kmeans.converged:
                kmeans.step()  # Continue until convergence
            status_text = "KMeans has converged!"

    elif triggered_by == 'reset-button':
        kmeans.assignment = [-1 for _ in range(len(kmeans.data))]
        manual_centers = []  # Reset manual centers when reset is clicked
        if init_method != 'manual':
            kmeans.initialize()
            status_text = f"Reset with {init_method} method and {num_clusters} clusters."
        else:
            status_text = f"Select {kmeans.k} centers manually by clicking on the plot."
            kmeans.centers = None  # Clear kmeans.centers when reset

    # Handle manual centroid selection via user clicks
    if init_method == 'manual' and clickData is not None:
        if len(manual_centers) < kmeans.k:
            clicked_point = clickData['points'][0]
            manual_centers.append([clicked_point['x'], clicked_point['y']])
            status_text = f"Selected {len(manual_centers)} of {kmeans.k} manual centers."

        if len(manual_centers) == kmeans.k and kmeans.centers is None:
            # Set the manually selected centers once fully selected
            kmeans.centers = np.array(manual_centers)
            status_text = "Manual centroids selected, ready to proceed with KMeans."

        manual_instructions = f"Click to select {kmeans.k} centers. Selected {len(manual_centers)} so far."

    # Ensure that data is generated before attempting to plot
    if kmeans.data is None:
        return go.Figure(), "No data available. Please generate a dataset.", ""

    # Plot data points and centroids
    data_points = go.Scatter(
        x=kmeans.data[:, 0],
        y=kmeans.data[:, 1],
        mode='markers',
        marker=dict(color=kmeans.assignment, size=10, showscale=True),
        name='Data Points'
    )

    # Plot selected manual centroids in real-time
    manual_centroid_points = go.Scatter(
        x=[center[0] for center in manual_centers],
        y=[center[1] for center in manual_centers],
        mode='markers',
        marker=dict(color='red', size=15, symbol='x'),
        name='Manual Centroids'
    )

    # Plot actual centroids only if they are initialized (i.e., not None)
    fig_data = [data_points, manual_centroid_points]
    if kmeans.centers is not None:
        centroid_points = go.Scatter(
            x=kmeans.centers[:, 0],
            y=kmeans.centers[:, 1],
            mode='markers',
            marker=dict(color='black', size=15, symbol='x'),
            name='Centroids'
        )
        fig_data.append(centroid_points)

    fig = go.Figure(data=fig_data)
    fig.update_layout(title='KMeans Clustering')

    return fig, status_text, manual_instructions



# Run the app
if __name__ == '__main__':
    dash_app.run_server(debug=True, host='127.0.0.1', port=3000)
