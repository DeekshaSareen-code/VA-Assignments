from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.b_data_profile import *
from assignments.assignment3 import a_libraries
from assignments.assignment1.e_experimentation import process_life_expectancy_dataset
from assignments.assignment2.c_clustering import cluster_iris_dataset_again
from assignments.assignment2.a_classification import decision_tree_classifier



##############################################
# In this file, we will use data and methods of previous assignments with visualization.
# But before you continue on, take some time to look on the internet about the many existing visualization types and their usages, for example:
# https://extremepresentation.typepad.com/blog/2006/09/choosing_a_good.html
# https://datavizcatalogue.com/
# https://plotly.com/python/
# https://www.tableau.com/learn/whitepapers/which-chart-or-graph-is-right-for-you
# Or just google "which visualization to use", and you'll find a near-infinite number of resources
#
# You may want to create a new visualization in the future, and for that I suggest using JavaScript and D3.js, but for the course, we will only
# use python and already available visualizations
##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
# For ALL methods return the fig and ax of matplotlib or fig from plotly!
##############################################
def matplotlib_bar_chart() -> Tuple:
    """
    Create a bar chart with a1/b_data_profile's get column max.
    Show the max of each numeric column from iris dataset as the bars
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    col_max = []
    for col in get_numeric_columns(df):
        max_val = get_column_max(df, col)
        # print(max_val)
        col_max.append(max_val)

    # Can't put column names since the method only takes numpy array
    fig, ax = a_libraries.matplotlib_bar_chart(np.array(col_max))

    return fig, ax


def matplotlib_pie_chart() -> Tuple:
    """
    Create a pie chart where each piece of the chart has the number of columns which are numeric/categorical/binary
    from the output of a1/e_/process_life_expectancy_dataset
    """
    df = process_life_expectancy_dataset()
    numericCols = get_numeric_columns(df)
    # print(numericCols)
    binaryCols = get_binary_columns(df)
    # print(binaryCols)
    categoryCols = get_text_categorical_columns(df)
    # print(categoryCols)

    fig, ax = a_libraries.matplotlib_pie_chart(np.array([len(numericCols), len(binaryCols), len(categoryCols)]))

    return fig, ax


def matplotlib_histogram() -> Tuple:
    """
    Build 4 histograms as subplots in one figure with the numeric values of the iris dataset
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    df.drop("species", axis=1, inplace=True)
    numericCols = list(df.columns)[:4]
    fig, ax = plt.subplots(nrows=2, ncols=2)

    c = 0
    for row in ax:
        for col in row:
            df_column = numericCols[c]
            col.hist(df[df_column].values)
            c = c + 1

    return fig, ax


def matplotlib_heatmap_chart() -> Tuple:
    """
    Remember a1/b_/pandas_profile? There is a heat map over there to analyse the correlation among columns.
    Use the pearson correlation (e.g. https://docs.scipy.org/doc/scipy-1.5.3/reference/generated/scipy.stats.pearsonr.html)
    to calculate the correlation between two numeric columns and show that as a heat map. Use the iris dataset.
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    df.drop("species", axis=1, inplace=True)
    corrDf = df.corr()
    fig, ax = a_libraries.matplotlib_heatmap_chart(corrDf.values)

    return fig, ax


# There are many other possibilities. Please, do check the documentation and examples so you
# may have a good breadth of tools for future work (in assignments, projects, and your own career)
###################################
# Once again, for ALL methods return the fig and ax of matplotlib or fig from plotly!


def plotly_scatter_plot_chart():
    """
    Use the result of a2/c_clustering/cluster_iris_dataset_again() as the color of a scatterplot made from the original (unprocessed)
    iris dataset. Choose among the numeric values to be the x and y coordinates.
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    data = cluster_iris_dataset_again()
    df['clusters'] = data['clusters']

    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="clusters")

    return fig


def plotly_bar_plot_chart():
    """
    Use the result of a2/c_clustering/cluster_iris_dataset_again() and use x as 3 groups of bars (one for each iris species)
    and each group has multiple bars, one for each cluster, with y as the count of instances in the specific cluster/species combination.
    The grouped bar chart is like https://plotly.com/python/bar-charts/#grouped-bar-chart (search for the grouped bar chart visualization)
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    data = cluster_iris_dataset_again()
    df['clusters'] = data['clusters']

    count_df = df.groupby(["species", "clusters"]).size().unstack(fill_value=0).stack().reset_index()
    count_df.columns = ["species", "clusters", "count"]
    count_df['clusters'] = count_df['clusters'].astype(str)

    fig = px.bar(count_df, x="species", color="clusters", y="count", barmode='group')

    return fig


def plotly_polar_scatterplot_chart():
    """
    Do something similar to a1/e_/process_life_expectancy_dataset, but don't drop the latitude and longitude.
    Use these two values to figure out the theta to plot values as a compass (example: https://plotly.com/python/polar-chart/).
    Each point should be one country and the radius should be thd value from the dataset (add up all years and feel free to ignore everything else)
    """

    le = process_life_expectancy_dataset()

    geo = pd.read_csv(Path('..', '..', 'geography.csv'))
    le = le[["country", "value"]].groupby(["country"]).sum().reset_index()


    le = pd.merge(le, geo[["name", "Latitude", "Longitude"]], left_on="country", right_on="name")
    le.drop("name", axis=1, inplace=True)
    Eradius = 6371


    # https://stackoverflow.com/a/1185413/7550476
    le['x'] = Eradius * np.cos(le["Latitude"]) * np.cos(le["Longitude"])
    le['y'] = Eradius * np.cos(le["Latitude"]) * np.sin(le["Longitude"])

    # Calculating theta from the coordinates
    le["theta"] = np.arctan2(le["y"], le["x"]) * 255

    # print(le.head().to_string())
    # Plotting
    fig = px.scatter_polar(le, r="value", theta="theta", color="country")

    return fig


def plotly_table():
    """
    Show the data from a2/a_classification/decision_tree_classifier() as a table
    See https://plotly.com/python/table/ for documentation
    """
    iris = read_dataset(Path('../../iris.csv'))
    label_col = "species"
    feature_cols = iris.columns.tolist()
    feature_cols.remove(label_col)
    x_iris = iris[feature_cols]
    y_iris = iris[label_col]
    df = decision_tree_classifier(x_iris, y_iris)
    df['model'] = str(df['model'])


    # pd.DataFrame.from_dict(df)
    print(df.keys())
    print(df.values())
    # df = pd.DataFrame(df.values(), columns=df.keys())
    print(df)


    # keys = df.keys()
    # print(  keys)
    # values = df.values()
    # print(values)

    fig = go.Figure(data=[go.Table(header=dict(values=list(df.keys()), fill_color='red', align='center', height=30),cells=dict(values=list(df.values()), height=30))])


    return fig


def plotly_composite_line_bar():
    """
    Use the data from a1/e_/process_life_expectancy_dataset and show in a single graph on year on x and value on y where
    there are 5 line charts of 5 countries (you choose which) and one bar chart on the background with the total value of all 5
    countries added up.
    """
    df = process_life_expectancy_dataset()

    final_df = df[((df["country"] == "Canada") | (df["country"] == "United States") | (df["country"] == "Pakistan") | (df["country"] == "India") | (df["country"] == "Zimbabwe"))]

    final_df.drop_duplicates(inplace=True)

    fig = px.line(final_df[['year', 'value']], x='year', y='value', color=final_df["country"])
    fig.add_bar(x=final_df['year'], y=final_df['value'])

    return fig


def plotly_map():
    """
    Use the data from a1/e_/process_life_expectancy_dataset on a plotly map (anyone will do)
    Examples: https://plotly.com/python/maps/, https://plotly.com/python/choropleth-maps/#using-builtin-country-and-state-geometries
    Use the value from the dataset of a specific year (e.g. 1900) to show as the color in the map
    """
    df = process_life_expectancy_dataset()
    filter = (df['year'] == '1900')
    df = df[filter]
    fig = px.choropleth(data_frame=df, locations=df['country'], locationmode="country names", color=df['value'],
                        hover_name=df['country'], color_continuous_scale=px.colors.sequential.Blackbody,
                        title="Life Expectancy plot")

    return fig


def plotly_tree_map():
    """
    Use plotly's treemap to plot any data returned from any of a1/e_experimentation or a2 tasks
    Documentation: https://plotly.com/python/treemaps/
    """

    df = process_life_expectancy_dataset()
    filter = ((df['year'] == '2000') | (df['year'] == '1900'))
    df = df[filter]
    print(df.head().to_string())

    fig = px.treemap(df, path=[px.Constant("World"), df['year'],  df['country']], values='value',
                     color='value',
                     color_continuous_scale='RdBu',
                     color_continuous_midpoint=np.average(df['value']),
                     title="Tree Map for Life Expectancy in the Year 2000 & 1900")
    fig.update_layout(margin=dict(t=80, l=30, r=30, b=80))
    # fig.show()
    return fig


if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    fig_m_bc, _ = matplotlib_bar_chart()
    # fig_m_bc.show()
    fig_m_pc, _ = matplotlib_pie_chart()
    # fig_m_pc.show()
    fig_m_h, _ = matplotlib_histogram()
    # fig_m_h.show()
    fig_m_hc, _ = matplotlib_heatmap_chart()
    # fig_m_hc.show()

    fig_p_s = plotly_scatter_plot_chart()
    # fig_p_s.show()
    fig_p_bpc = plotly_bar_plot_chart()
    # fig_p_bpc.show()
    fig_p_psc = plotly_polar_scatterplot_chart()
    # fig_p_psc.show()
    fig_p_t = plotly_table()
    # fig_p_t.show()
    fig_p_clb = plotly_composite_line_bar()
    # fig_p_clb.show()
    fig_p_map = plotly_map()
    # fig_p_map.show()
    fig_p_treemap = plotly_tree_map()
    # fig_p_treemap.show()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # fig_m_bc.show()
    # fig_m_pc.show()
    # fig_m_h.show()
    # fig_m_hc.show()
    #
    # fig_p_s.show()
    # fig_p_bpc.show()
    # fig_p_psc.show()
    # fig_p_t.show()
    # fig_p_clb.show()
    # fig_p_map.show()
    # fig_p_treemap.show()
