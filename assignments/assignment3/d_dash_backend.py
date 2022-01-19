from typing import Tuple

import dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Output, Input, State
from matplotlib.widgets import Button, Slider
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from assignments.assignment1.a_load_file import *
from assignments.assignment1.e_experimentation import *
from assignments.assignment3.a_libraries import *
import json
from assignments.assignment3.b_simple_usages import *

##############################################
# Now let's use dash, a library built on top of flask (a backend framework for python) and plotly
# Check the documentation at https://dash.plotly.com/
# For a complete example, check https://dash-bootstrap-components.opensource.faculty.ai/examples/iris/
# Example(s). Read the comments in the following method(s)
##############################################
def dash_simple_example():
    """
    Here is a simple example from https://dash.plotly.com/layout
    The focus is to create a fig from plotly and add it to dash, but differently from using just plotly, now we can use html elements,
    such as H1 for headers, Div for dividers, and all interations (buttons, sliders, etc).
    Check dash documentation for html and core components.
    """
    app = dash.Dash(__name__)

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    # You create a fig just as you did in a_
    fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),
        html.Div(children='Dash: A web application framework for Python.'),
        dcc.Graph(
            id='example-graph',
            figure=fig  # and include the fig here as a dcc.Graph
        )
    ])

    # To run the app, use app.run_server(debug=True)
    # However, do return the app instead so that the file can be run as in the end of this file
    return app


def dash_with_bootstrap_example():
    """
    Here is a simple example from https://dash.plotly.com/layout
    See examples of components from the bootstrap library at https://dash-bootstrap-components.opensource.faculty.ai/docs/components/alert/
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    fig1 = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
    fig2 = px.line(df, x="Fruit", y="Amount", color="City")

    app.layout = dbc.Container([
        html.H1(children='Hello Dash'),
        html.Hr(),
        dbc.Row([
            dbc.Col(html.Div(children='Dash: A web application framework for Python.'), md=4),
            dbc.Col(dbc.Button('Example Button', color='primary', style={'margin-bottom': '1em'}, block=True), md=8)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='example-graph1', figure=fig1)),
            dbc.Col(dcc.Graph(id='example-graph2', figure=fig2))
        ])
    ])

    # To run the app, use app.run_server(debug=True)
    # However, do return the app instead so that the file can be run as in the end of this file
    return app


def dash_callback_example():
    """
    Here is a more complex example that uses callbacks. With this example, I believe you will suddenly perceive why dash (and webapps)
    are so much better for visual analysis.
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    app.layout = dbc.Container([
        html.H1(children='Hello Dash'),
        html.Div(children='Dash: A web application framework for Python.'),
        html.Hr(),
        dbc.FormGroup([
            dbc.Label("Choose Dataset"),
            dcc.Dropdown(id="dropdown", value=1,
                         options=[{"label": "First Data", "value": 1}, {"label": "Second Data", "value": 2}]),
        ]),
        dbc.FormGroup([
            dbc.Label(id='slider-value'),
            dcc.Slider(id="slider", min=1, max=10, step=0.5, value=1),
        ]),
        dbc.Button('Run Callback', id='example-button', color='primary', style={'margin-bottom': '1em'}, block=True),
        dbc.Row([
            dbc.Col(dcc.Graph(id='example-graph')),
            # Not including fig here because it will be generated with the callback
        ])
    ])

    @app.callback(  # See documentation or tutorial to see how to use this
        Output('example-graph', 'figure'),
        # Outputs is what you wish to update with the callback, which in this case is the figure
        [Input('example-button', 'n_clicks')],
        # Use inputs to define when this callback is called, and read from the values in the inputs as parameters in the method
        [State('dropdown', 'value'),
         # Use states to read values from the interface, but values only in states will not trigger the callback when changed
         State('slider',
               'value')])  # For example, here if you change the slider, this method will not be called, it will only be called when you click the button
    def update_figure(n_clicks, dropdown_value, slider_value):
        df2 = df[:]
        df2.Amount = df2.Amount * slider_value
        if dropdown_value == 1:
            return px.bar(df2, x="Fruit", y="Amount", color="City", barmode="group")
        else:
            return px.line(df2, x="City", y="Amount", color="Fruit")

    @app.callback(Output('slider-value', 'children'), [Input('slider', 'value')])
    def update_slider_value(slider):
        return f'Multiplier: {slider}'

    #  You can also use app.callback to get selection from any of the plotly graphs, including tables and maps, and update anything you wish.
    #  See some examples at https://dash-gallery.plotly.host/Portal/

    return app


##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
##############################################
def dash_task():
    """
    There is only only one task to do, a web app with:
    1. Some nice title
    2. One visualization placeholder for dataset visualization
        a. A dropdown to allow me to select which dataset I want to see (iris, video_game and life_expectancy)
        b. Two other dropdowns for me to choose what column to put in x and what column to put in y of the visualization
        c. Another dropdown for me to choose what type of graph I want (see examples in file a_) (at least 3 choices of graphs)
        d. Feel free to change the structure of the dataset if you prefer (e.g. change life_expectancy so that
            there is one column of "year", one for "country" and one for "value")
    4. A https://dash-bootstrap-components.opensource.faculty.ai/docs/components/card/ with the number of rows being showed on the above graph
    5. Another visualization with:
        a. It will containing the figure created in the tasks in a_, b_ or c_ related to plotly's figures
        b. Add a dropdown for me to choose among 3 (or more if you wish) different graphs from a_, b_ or c_ (choose the ones you like)
        c. In this visualization, if I select data in the visualization, update some text in the page (can be a new bootstrap card with text inside)
            with the number of values selected. (see https://dash.plotly.com/interactive-graphing for examples)
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    iris = pd.read_csv(Path('..', '..', 'iris.csv'))
    ratings = pd.read_csv(Path('..', '..', 'ratings_Video_Games.csv'))
    ley = process_life_expectancy_dataset()

    # Default is Iris dataset columns (Not necessary)
    default_columns = []
    for col in iris.columns:
        default_columns.append({"label": col, "value": col})

    app.layout = dbc.Container([
        # Req. 1
        html.Title("VA Assignment"),
        html.H1(children='Dashboard - 1'),
        html.Div(children='Assignment 3'),
        html.Hr(),
        dcc.Tabs([
            dcc.Tab(label='Visualization 1', children=[
                visualize1(default_columns, iris)
            ]),
            dcc.Tab(label='Visualization 2', children=[
                visualize2()
            ])
        ])
    ])

    # Req. 2 - a,b,c
    @app.callback(
        dash.dependencies.Output('dropdown2', 'options'),
        dash.dependencies.Output('dropdown3', 'options'),
        [dash.dependencies.Input('dropdown1', 'value')]
    )
    def update_columns_dropdown(name):
        columns = getDatasetColumns(name, iris, ratings, ley)
        return columns, columns

    # Req. 4 (Number of rows)
    @app.callback(
        dash.dependencies.Output('n_rows', 'children'),
        [dash.dependencies.Input('dropdown1', 'value')]
    )
    def update_number_of_rows(name):
        if name == "iris":
            return len(iris)
        elif name == "ratings":
            return len(ratings)
        elif name == "ley":
            return len(ley)

    # Req. 2 Cont.
    @app.callback(
        dash.dependencies.Output('id1', 'figure'),
        [Input('button1', 'n_clicks')],
        [State('dropdown1', 'value'),
         State('dropdown2', 'value'),
         State('dropdown3', 'value'),
         State('dropdown4', 'value')]
    )
    def updateVisualization1(n_clicks, dataset, x, y, graph_type):
        if n_clicks:
            fig = getGraph(n_clicks, dataset, x, y, graph_type, iris, ratings, ley)
            return fig

        return {}

    graph_function_mapping = {"scatter": plotly_scatter_plot_chart, "bar": plotly_bar_plot_chart,
                              "map": plotly_map}


    # (Req. 5)
    @app.callback(
        dash.dependencies.Output('id2', 'figure'),
        [Input('button2', 'n_clicks')],
        [State('dropdown5', 'value')]
    )
    def updateVisualization2(n_clicks, graph_type):
        if n_clicks:
            fig = graph_function_mapping[graph_type]()
            return fig

        return {}

    # Req. 5c
    @app.callback(
        dash.dependencies.Output('selected_data', 'children'),
        [dash.dependencies.Input('id2', 'clickData')]
    )
    def display_selected_data(selected_data):
        if selected_data:
            return json.dumps(selected_data, indent=2)

        return {}

    return app




# Helper functions to keep the code clean
def getDatasetColumns(name, iris, ratings, ley):
    columns = []

    if name == "iris":
        columns = dropdown_of_column_values(iris, columns)
    elif name == "ratings":
        columns = dropdown_of_column_values(ratings, columns)
    elif name == "ley":
        columns = dropdown_of_column_values(ley, columns)
    return columns

def getGraph(n_clicks, dataset, x, y, graph_type, iris, ratings, ley):
    fig = None
    if dataset == "iris":
        iris_df_copy = iris.copy()
        iris_df_copy['x'] = iris_df_copy[x]
        iris_df_copy['y'] = iris_df_copy[y]
        fig = getGraph_byType(iris_df_copy, graph_type, fig)

    elif dataset == "ratings":
        ratings_vg_df_copy = ratings.copy()
        ratings_vg_df_copy['x'] = ratings_vg_df_copy[x]
        ratings_vg_df_copy['y'] = ratings_vg_df_copy[y]
        fig = getGraph_byType(ratings_vg_df_copy, graph_type, fig)

    elif dataset == "ley":
        ley_df_copy = ley.copy()
        ley_df_copy['x'] = ley_df_copy[x]
        ley_df_copy['y'] = ley_df_copy[y]
        fig = getGraph_byType(ley_df_copy, graph_type, fig)

    return fig

def getGraph_byType(df, graph_type, fig):
    graph_function_mapping = {"bar": plotly_bar_chart, "pie": plotly_pie_chart,"histogram": plotly_histogram}
    if graph_type == "histogram":
        fig = graph_function_mapping[graph_type](df, 8)
    else:
        fig = graph_function_mapping[graph_type](df)

    return fig


def visualize1(default_columns, default_df):
    vis1layout = html.Div([
        dbc.FormGroup([
            dbc.Label("Choose dataset"),
            dcc.Dropdown(id="dropdown1", value=1, options=[{"label": "Iris", "value": "iris"},
                                                           {"label": "Video Game Ratings", "value": "ratings"},
                                                           {"label": "Life Expectancy Years", "value": "ley"}]),
            dbc.Label("x column"),
            dcc.Dropdown(id="dropdown2", value=2, options=default_columns),
            dbc.Label("y column"),
            dcc.Dropdown(id="dropdown3", value=3, options=default_columns),
            dbc.Label("graph"),
            dcc.Dropdown(id="dropdown4", value=4, options=[{"label": "Bar", "value": "bar"},
                                                           {"label": "Pie", "value": "pie"},
                                                           {"label": "Histogram", "value": "histogram"}]),
            html.Br()
        ]),
        dbc.Button('Update Graph', id='button1', color='primary', style={'margin-bottom': '1em'}, block=True),
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4("Number of rows", className="card-title"),
                        html.P(str(len(default_df)), id="n_rows", className="card-text"),
                    ]
                )
            ],
            style={"width": "15rem"},
        ),
        dbc.Row([
            dbc.Col(dcc.Graph(id='id1')),
        ])
    ])

    return vis1layout


def visualize2():
    vis2layout = html.Div([
        dbc.FormGroup([
            dbc.Label("Choose visualization"),
            dcc.Dropdown(id="dropdown5", value=1, options=[{"label": "Scatter", "value": "scatter"},
                                                           {"label": "Bar", "value": "bar"},
                                                           {"label": "Map", "value": "map"}]),
            html.Br(),
            dbc.Button('Show', id='button2', color='primary', style={'margin-bottom': '1em'}, block=True),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='id2')),
        ]),
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4("Clicked data", className="card-title"),
                        html.P("", id="selected_data", className="card-text")
                    ]
                )
            ],
            style={"width": "18rem"},
        )
    ])

    return vis2layout


def dropdown_of_column_values(xdf, new_options):
    for col in xdf.columns:
        new_options.append({"label": col, "value": col})

    return new_options


if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    # dash_simple_example().run_server(debug=True)
    app_ce = dash_callback_example()
    app_b = dash_with_bootstrap_example()
    app_c = dash_callback_example()
    app_t = dash_task()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # app_ce.run_server(debug=True)
    # app_b.run_server(debug=True)
    # app_c.run_server(debug=True)
    # app_t.run_server(debug=True)
