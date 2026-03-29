"""
TThe UtilPlot module is created to visualize the vehicle measurements
The plotly dash is used to plot particular signals, for example. wheel velocity and measured
longitudinal and lateral velocity.

 The goal of the plot is to check
   the length of each measurement and its drive maneuver

"""

from pathlib import Path
import os
import numpy as np
import plotly
from asammdf import MDF
from dash import Dash, html, dcc
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler

app = Dash()

def load_measurements(filelist):
    """Load vehicle measurements, convert the data as panda data frame.
    :param filelist: a list with mdf compatible files, acutally all the files shall be mf4 files and have
         expected time series signals.
    :return measurements: a list with dictionary element, the dictionary has two keys filename and dataframe.
    """
    measurements = []
    # Define signals to be ploted
    signal_to_plot = ['VEL_FL', 'Corrsys_VL', 'Corrsys_VQ', 'RT_Msg604_VelLateral','YR',
                      'RT_Msg604_VelForward']
    for file in filelist:
        # Load measurement and convert it to a panda dataframe
        data_frame = MDF(file).to_dataframe()
        # Remove unused signals from dataframe
        data_frame.drop(columns=data_frame.columns.difference(signal_to_plot), inplace=True)
        # Add load data and file name as dictionary to lists
        measurements.append({
            "filename": os.path.basename(file),
            "dataframe": data_frame})
    return measurements


def plotting(measurements):
    """Return a figure with sub plots
    each row of the figure has 2 plots, the 1st plot prints the wheel velocity and the measured vehicle longitudinal velocity
    the 2nd measured vehicle lateral velocity.

    :param measurements: refer the return of the function
    :return fig: figure with sub plots:
    """
    num_rows = len(measurements)
    spacing = (1 / (num_rows + 1)) * 0.3
    # Initialize a fig from make_subplots, it has 2 columns, number of rows is the number of measurements
    list_4_subtitles = [[f"{measurement['filename']}","ref"] for measurement in measurements]
    list_4_subtitles = [x for xs in list_4_subtitles for x in xs]
    fig = make_subplots(rows = num_rows,
                        cols = 2,
                        subplot_titles = list_4_subtitles,
                        vertical_spacing = spacing,
                        )
    loglist = []
    # Iterate the measurements
    # Pick signals which shall be plotted
    for idx, measurement in enumerate(measurements):
        # Create a graphical objects list for the 1st plot of each row
        go_objects = []
        # Add signal VEL_FL to the go_objects
        vel_fl = measurement['dataframe']['VEL_FL']
        go_objects.append(
            go.Scatter(x=np.arange(vel_fl.shape[0]), y=vel_fl, mode="lines", name='Vel_FL', showlegend=False,
                       legendgroup=str(idx) + "_VelFl",
                       line=dict(color=plotly.colors.qualitative.Dark24[1 % len(plotly.colors.qualitative.Dark24)]),
                       visible=True)
        )
        # Try to add signal Corrsys_VL to the go_objects, the signal is only valid, when it is available in the data frame
        # and its mean value not close to 0, 3 is used as threshold for closing 0
        # Corrsys_VL has to be transferred to the vehicle central of gravity(cog)
        try:
            corrsys_lgt_v = measurement['dataframe']['Corrsys_VL']
            if corrsys_lgt_v.mean() < 3:
                raise KeyError("no corrsys")
            # Transfer to cog
            corrsys_lgt_v = (corrsys_lgt_v/3.6 - (0.47) * measurement['dataframe']['YR'] / 180 * 3.1416)*3.6
            go_objects.append(
                go.Scatter(x=np.arange(corrsys_lgt_v.shape[0]), y=corrsys_lgt_v, mode="lines", name='corrsys_lgt_v', showlegend=False,
                           legendgroup=str(idx) + "_corrsys_lgt_v",
                           line=dict(color=plotly.colors.qualitative.Dark24[2 % len(plotly.colors.qualitative.Dark24)]),
                           visible=True)
            )
        except KeyError:
            # In case the corrsys_VL is invalid, add information to the plot title
            fig.layout.annotations[idx*2].text = fig.layout.annotations[idx*2].text+ " no corrsys"
            pass
        # Try to add signal RT_Msg604_VelForward to the go_objects, the signal is in unit m/s
        # Multiplication 3.6 is to convert it to km/h
        try:
            rt_vl = measurement['dataframe']['RT_Msg604_VelForward'] * 3.6
            go_objects.append(
                go.Scatter(x=np.arange(rt_vl.shape[0]), y=rt_vl, mode="lines", name='rt_vl', showlegend=False,
                           legendgroup=str(idx) + "_rt_vl",
                           line=dict(color=plotly.colors.qualitative.Dark24[2 % len(plotly.colors.qualitative.Dark24)]),
                           visible=True)
            )
        except KeyError:
            fig.layout.annotations[idx*2].text = fig.layout.annotations[idx*2].text + " no rt box"
            pass

        # Add the go_objects of current measurement data to the 1st plot of the row
        for plt in go_objects:
            fig.add_trace(plt, row=idx+1, col=1)

        # Empty the graphical object for 2nd plot of the row
        go_objects = []
        # Try to add signal Corrsys_VQ to the go_objects, the signal is only valid, when it is available in the data frame
        # and the Corrsys_VL mean value not close to 0, 3 is used as threshold for closing 0
        # Corrsys_VQ has to be transferred to the vehicle central of gravity(cog)
        try:
            cr_vl = measurement['dataframe']['Corrsys_VL']
            if cr_vl.mean() < 3:
                raise KeyError("no corrsys")
            cr_vq = measurement['dataframe']['Corrsys_VQ']
            # Transfer to cog
            cr_vq = ( cr_vq/3.6 + (1.5+0.74) * measurement['dataframe']['YR'] / 180 * 3.1416) * 3.6
            go_objects.append(
                go.Scatter(x=np.arange(cr_vq.shape[0]), y=cr_vq, mode="lines", name='cr_vq', showlegend=False,
                           legendgroup=str(idx) + "_cr_vq",
                           line=dict(color=plotly.colors.qualitative.Dark24[2 % len(plotly.colors.qualitative.Dark24)]),
                           visible=True)
            )
        except KeyError:
            pass
        # Try to add RT_Msg604_VelLateral to go_objects
        try:
            rt_vq = measurement['dataframe']['RT_Msg604_VelLateral'] * 3.6
            go_objects.append(
                go.Scatter(x=np.arange(rt_vq.shape[0]), y=rt_vq, mode="lines", name='rt_vq', showlegend=False,
                           legendgroup=str(idx) + "_rt_vq",
                           line=dict(color=plotly.colors.qualitative.Dark24[2 % len(plotly.colors.qualitative.Dark24)]),
                           visible=True)
            )
        except KeyError:
            pass

        # Add the go_objects of current measurement data to the 2nd plot of the row
        for plt in go_objects:
            fig.add_trace(plt, row=idx+1, col=2)

        # The following line is add plot title with information of
        # the availability of the Corrsys and RT signals
        # The loglist will be wrote to a log file late on,
        # only used for one time

        # loglist.append(fig.layout.annotations[idx*2].text)

    fig.update_layout(
        height = 300 * (num_rows + 1),
        width = 1800,
        barmode = "group",
        boxmode = "group",
        boxgroupgap = 0.0,
        boxgap=0,
    )
    spacing = (1 / (num_rows + 1)) * 0.3
    fig.update_layout(
        legend = dict( orientation = "v", yanchor = "bottom", y = 0, xanchor = "left", x = 1.5),
        margin = dict( t = 100)
    )

    # no need for the rewrite logfile
    # with open("logfile.txt", "w") as f:
    #     for item in loglist:
    #         f.write(f"{item}\n")
    return fig


def create_layout(fig):
    """ standard function to create html layout based on the fig
    :param fig: refer the return of the function plotting
    :return layout: a html layout
    """
    fig = FigureResampler(fig)
    layout = html.Div(
        [
            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Measurements",
                        children=[
                            dcc.Graph(
                                figure=fig,
                                style={"display": "inline-block"},
                            )
                        ],
                    ),
                ]
            ),
        ]
    )
    return layout


if __name__ == '__main__':
    # Collect mf4 files in the dataset folder
    fileList = [p.as_posix() for p in Path('./DataSets/TrainData').iterdir() if p.suffix == '.mf4']
    # Create figure
    fig = plotting(load_measurements(fileList))
    # Create html layout
    app.layout = create_layout(fig)
    app.run(debug=True)
