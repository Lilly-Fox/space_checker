# Imports
from astropy import units as u
from external_photometry import event_cutout
from external_photometry import _DESI_phot
from external_photometry import _delve_objects
from astropy.coordinates import SkyCoord
# from astroquery.vizier import Vizier
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import heapq

from matplotlib.markers import MarkerStyle
import matplotlib.colors as mcolors


import os
import csv
import sys
import warnings

def classify_lightcurve(event_width, peaks):
    """
    Classifies a lightcurve event based on its duration and the number of peaks

    Parameters:
    event_width (float): The width of the event.
    peaks (int): The number of peaks in the lightcurve.

    Returns:
    tuple: A tuple containing the classification of the lightcurve event and the likely color of the source. Possible values are:
        - ("dwarf_nova", "blue"): If the event width is greater than 0.8 and there is one peak.
        - ("gb_or_sf", "none"): If there is one peak (could be a gamma burst or solar flare).
        - ("solar_flare", "red"): If there are multiple peaks.
        - ("mystery", "none"): If the event does not fit any of the above criteria.
    """
    if event_width > 0.8 and peaks == 1:    # Duration can be changed 
        return ("dwarf_nova","blue")
    elif peaks == 1 and event_width < 0.17:   #Solar flares and gamma bursts occur on the order of 4 hours = 0.17 days
        return ("gb_or_sf", "none")   # Gamma burst or solar flare
    elif peaks > 1 and event_width < 0.17:
        return ("solar_flare", "red")
    else:
        return ("mystery", "none")


def display_lightcurve(x, y, event_width):
    """
    Plots the TESS light curve data with a specified time range around the event.

    Parameters:
    x (array-like): Array of time values (in Modified Julian Date).
    y (array-like): Array of corresponding light curve counts.
    event_width (float): The width of the event to be displayed on the graph.

    The plot is displayed in a figure with a size of 10x6 inches.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='pink')
    plt.xlabel("Time (MDJD)")
    plt.ylabel("Counts")
    plt.title("Light Curve")
    plt.grid(True)

    # Add a box with event_width
    textstr = f'Event Width: {event_width:.2f} days, {event_width*24:.2f} hours'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    plt.show()


def tess_lightcurve(data):
    """
    Processes a TESS lightcurve data file to identify and analyze events.

    Parameters:
    data (pd.DataFrame): The data from the lightcurve csv file.

    Returns:
    tuple: A tuple containing the event duration (float) and the number of peaks (int).

    The function reads the lightcurve data from the specified CSV file, filters the data to 
    identify events, and extracts a window of data surrounding the event. It then converts the 
    time and counts arrays from the CSV to floats, identifies peaks in the counts data, and prints the event 
    duration and number of peaks. 
    
    If no event occurs, it prints a message indicating so.
    """
    
    # Extract the data during the event
    event_data = data[data["event_flag"] != 0.0]

    # Process the event data to identify the light curve type
    if event_data.empty:
        print("No event occurs!")
    else:
        #Filter the entire dataset to return the events and some time surrounding it
        event_time, event_width = event_data["time_MDJD"].iloc[0], event_data["time_MDJD"].iloc[-1] - event_data["time_MDJD"].iloc[0]
        window_data = data[(data["time_MDJD"] >= event_time - 4 * event_width) & (data["time_MDJD"] <= event_time + 4 * event_width)]
        
        #Rename the time and counts (of the event) arrays as time, counts and convert the arrays to floats
        time, counts = window_data["time_MDJD"], window_data["counts"]
        time = np.asarray(time, dtype=float)
        counts = np.asarray(counts, dtype=float)

        display_lightcurve(time, counts, event_width)

        # Identify peaks
        threshold = (np.max(counts) - np.min(counts))/1.5    #Adjust 1.5 if inaccurate for a larger dataset
        peaks =  len(find_peaks(counts, prominence = threshold)[0])

        # Classify the lightcurve type
        lightcurve_type = classify_lightcurve(event_width, peaks)
    return lightcurve_type

