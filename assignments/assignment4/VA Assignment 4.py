#Import all the necessary dependables
from pathlib import Path

from dash import dash
import dash
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from dash.dependencies import Output, Input, State
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.b_data_profile import get_numeric_columns
from assignments.assignment1.c_data_cleaning import fix_nans
from assignments.assignment2.c_clustering import simple_k_means


def task(df: pd.DataFrame):
    """
    For the last assignment, there is only one task, which will use your knowledge from all previous assignments.
    If your methods of a1, a2 and a3 are well implemented, a4 will be fairly simple, so reuse the methods if possible for your own
    benefit! If you prefer, you can reimplement any logic you with in the assignment4 folder.

    For this task, feel free to create new files, modules or methods in this file as you see fit. Our test will be done by calling this
    task() method, and we expect to receive the dash app back (similar to a3) and we will run it. No other method will be called by us, so
    make sure your code is running as expected. We will basically run this code: `task().run_server(debug=True)`

    For this task, you will build a dash app where we can perform a simple form of interactivity on it. We will use the accidents.csv
    dataset. This accidents.csv dataset has information about traffic accidents in the UK, and is available to you now.
    You will show the most accident prone areas in the UK on a map, so the regular value shown in the map should let us know the number of accidents
    that occurred in a specific area of the map (which should include the accident's severity as well as a weight factor). That said, the purpose of
    this map is to be used by the police to identify how they can do things better.

    **This is raw data, so preprocess the data as per requirement. Drop columns that you feel are unnecessary for the analysis or clustering. 
    Don't forget to add comments about why you did what you did**

    
    ##############################################################
    # Your task will be to Implement all the below functionalities
    ##############################################################

    1. (40pts) Implement a map with the GPS points for the accidents per month. Have a slider(#slider1) that can be used to filter accident data for the month I need.
        You are free to choose a map style, but I suggest using a scatter plot map.

    2. (10pts) Implement a dropdown to select few of the numerical columns in the dataset that can be used meaningfully to represent the size of the GPS points. 
        By default the size of the GPS point on map should be based on the value of "accident_severity" column.

    3. (30pts) You have to Cluster the points now. Be sure to have a text somewhere on the webpage that says what clustering algorithm you are using (e.g. KMeans, dbscan, etc).
        For clustering, you should run a clustering method over the dataset (it should run fairly fast for quick iteration, so make sure to use a simple clustering procedure)
        **COLOR** the GPS points based on the clustering label returned from the algorithm.

    4. (10pts) Have a button(#run_clustering) to run or rerun the clustering algorithm over the filtered dataset (filtered using #slider1 to select months).

    5. (10pts) At least one parameter of the clustering algorithm should be made available for us to tinker with as a button/slider/dropdown. 
        When I change it and click #run_clustering button, your code should rerun the clustering algorithm. 
        example: change the number of clusters in kmeans or eps value in dbscan.

        Please note: distance values should be in meters, for example dbscan uses eps as a parameter. This value should be read in mts from users and converted appropriately to be used in clustering, 
        so input_eps=100 should mean algorithm uses 100mts circle for finding core and non-core points. 
  
    The total points is 100pts
    """


    #for task 1 created a col array that contains columns significant for plotting GPS points
    col = ["accident_year","location_easting_osgr","location_northing_osgr","police_force","accident_severity","number_of_vehicles","number_of_casualties"]

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = dbc.Container([
        html.H1(children='Hello Dash'),
        html.Hr(),
        dbc.FormGroup([
            dbc.Label("Choose Column to represent GPS point size"),
            dcc.Dropdown(id="dropdown", value="accident_severity",  #(task 2): creating a dropdown for dataframe columns
                         options=[
                                {"label": key, "value": key} for key in col]),
        ]),
        # (task 1): creating a slider to choose the month for plotting GPS points
        html.H6('Choose Month to display data points for a specific month'),
        dbc.FormGroup([
            dbc.Label(id='slider-value'),
            dcc.Slider(id="slider", min=1, max=12, step=1,marks={
                1: 'Jan',
                2: 'Feb',
                3: 'Mar',
                4: 'April',
                5: 'May',
                6: 'June',
                7: 'July',
                8: 'Aug',
                9: 'Sep',
                10: 'Oct',
                11: 'Nov',
                12: 'Dec',
                }, value=1, tooltip = { 'always_visible': True })
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='example-graph')),
        ]),
        html.H3('The clustering alogrithm used to cluster the data points is K-Means'),
        html.H6('Choose number of Clusters for your K-Means clustering algorithm'),
        #(task 5): created another cluster for changing the cluster numbers
        dbc.FormGroup([
            dbc.Label(id='slider-value2'),
            dcc.Slider(id="clusteringslider", min=2, max=7, step=1, value=2, marks={i: '{}'.format(i) for i in range(12)}, tooltip = { 'always_visible': True })
        ]),
        #(task 4): created a button to run clustering
        dbc.Button('Run Clustering', id='example-button', color='primary', style={'margin-bottom': '1em'},
                   block=True),

    ])

    #call back function assigning order to the graph, sliders and buttons
    @app.callback(
        Output('example-graph', 'figure'),
        [Input('dropdown', 'value'),
        Input('slider','value')],
        [State('clusteringslider','value')],
        [Input('example-button', 'n_clicks')]
    )

    # this function is used to perfrom clustering on the data points as well as plot a scatter plot
    # using scatter_mapbox and assigning the column value chosen from the drop down as the size of the cluster points
    def update_figure( dropdown_value, slider_value,clusteringslider_value,n_clicks):

        returned_data = simple_k_means(df, n_clusters=clusteringslider_value)
        fig, ax = plt.subplots()
        fig = px.scatter_mapbox(df, lat=df["latitude"], lon=df["longitude"], size_max=15, zoom=10,
                                   color=returned_data['model'].labels_,size=dropdown_value)
        fig.update_layout(mapbox_style="light",
                              mapbox_accesstoken="pk.eyJ1IjoiZGVla3NoYXNhcmVlbiIsImEiOiJja3dlYm93bGUwMzVpMndwOWJ1M2M0dnlpIn0.aCVlRBubTH65ExTcmOuNPg")

        return fig

    return app


if __name__ == "__main__":

    df = read_dataset(Path('..', '..', 'accidents.csv'))
    df[["day", "month", "year"]] = df["date"].str.split("/", expand=True)
    # created 3 columns by splitting date on "/" and removed columns from the dataset which seemed to add no significant
    #value to the dataset for analysis
    df = df.drop(["accident_index", "accident_reference", "date", "time", "local_authority_district",
                  "local_authority_ons_district", "local_authority_highway", "first_road_class", "first_road_number",
                  "road_type", "speed_limit", "junction_detail", "junction_control", "second_road_class",
                  "second_road_number", "pedestrian_crossing_human_control", "pedestrian_crossing_physical_facilities",
                  "light_conditions", "weather_conditions", "road_surface_conditions", "special_conditions_at_site",
                  "carriageway_hazards", "urban_or_rural_area", "did_police_officer_attend_scene_of_accident",
                  "trunk_road_flag", "lsoa_of_accident_location"], axis=1)

    #filtered to get only the numeric columns from the dataset and removed all the NaNs
    for c in list(get_numeric_columns(df)):
        df = fix_nans(df, c)

    #sampled the dataset to 10000 data points
    df = df.sample(10000)

    app_c = task(df)

    app_c.run_server(debug=True)
