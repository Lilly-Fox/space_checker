#Get the final probability list that every source in the found catalogue is the transient
# Imports
from astropy import units as u
from astropy.coordinates import SkyCoord

from external_photometry import event_cutout
from external_photometry import _DESI_phot
from external_photometry import _delve_objects

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
import re


def is_color_match_single(color_from_lightcurve, source):
    """
    Determines if the color of the lightcurve matches the color of the object in the catalog,
    and then scales the list to make it usable as a list of probabilities

    Parameters
    ----------
    color_from_lightcurve : str
        The expected colour based on the lightcurve type
    cat : pd.DataFrame
        The catalog of objects

    Returns
    -------
    list: The scaled list of probabilities

    Problem: This function is biased to the results of the catalogue search
    """
    # Extract the magnitude in g and r of the catalogue as lists
    g_mag = source[2] #magnitude in g band
    r_mag = source[3] #magnitude in r band
    i_mag = source[4] #magnitude in i band
    z_mag = source[5] #magnitude in z band

    color = g_mag-r_mag

    #Set the probability to the same amount for every source if there is no specific color of source from the lightcurve
    if color_from_lightcurve == "none":
        return 0.5

    #Introduce handling for NANs

    if np.isnan(color):
        if np.isnan(g_mag) and np.isnan(r_mag):
            color = 0   # Sources not observed in g or r are very red [problem here]
        elif np.isnan(z_mag) and np.isnan(i_mag):
            color = 0  # Sources not observed in z or i are very blue [problem here]
        # Set all other sources as having a mean magnitude so as not to reward or punish for having a not-observed color
        else:
            color = 0

    return color


def calculate_gaussian(mean, stddev, x):
    """
    Calculate the value of x on a gaussian curve

    Parameters
    ----------
    mean : float
        The mean of the distribution
    stddev : float
        The standard deviation of the distribution
    x : float
        The value to calculate the value for

    Returns
    -------
    float: The value at the point x
    """
    return (1 / (stddev * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
   
def find_distance(source, event_ra, event_dec):
    # Extract the position data

    coords = SkyCoord(source[1], source[2], unit="deg", frame="icrs")
    event_sky_pos = SkyCoord(event_ra, event_dec, unit="deg", frame="icrs")


    # Calculate the distance
    separation = coords.separation(event_sky_pos)
    distance = separation.arcsecond
    return distance


def distance_prob_single(source, event_coords):
    """
    Assign a number for the probability if the source being the transient 
    based on the distance of the source from the TESS transient event

    Parameters
    ----------
    cat : pd.DataFrame
        Full catalogue of possible sources
    event_coords : (float, float)
        The coordinates of the calculated event center in degrees

    """

    # Create an empty list to populate with probabilities
    prob_list = []

    # set up Guassian function variables
    mean = 0
    stddev = 20   #20 arcseconds is the maximum distance possible

    # Loop through each object
    distance = find_distance(source, event_coords[0], event_coords[1])
        
    # Calculate the probability
    prob = calculate_gaussian(mean, stddev, distance)
    scaled_prob = prob * (1 / calculate_gaussian(mean, stddev, 0))
    return scaled_prob

def magnitude_prob_single(source):
    """
    Assign a number for the probability based on the magnitude of each object

    Parameters
    ----------
    all_objects : Full catalogue of information from cat

    Returns the scaled probabilities
    """

    mags = [source[3], source[4], source[5], source[6]]  # Extract psf magnitudes in griz bands
    filtered_mags = [mag for mag in mags if not np.isnan(mag)]  # Remove the unmeasured filters
    try:
        # Take the brightest (minimum) magnitude
        mag = min(filtered_mags)
    except ValueError:
        print(f"Object ID {source[0]} has no psf measurements in griz bands")

    # Calculate probability for each magnitude
    mag_probability = 4 * np.exp(-mag / 7)
    return mag_probability

def is_type_match_single(event_type, source):
    """
    Assign a percentage based on if each object type matches the expected object type

    Parameters
    ----------
    event_type: str
        The event type determined from the light curve classify_lightcurve(event_width, peaks) function
    cat: pandas.DataFrame
        The object catalogue
    """
    solar_flare_probs = {0: 0, 1: 1, 2: 1}

    if event_type == "solar_flare":
        probability_type = (solar_flare_probs[source[23]])
    elif event_type == "dwarf_nova":
        if source['star'] in [1, 2]:
            probability_type = 1
        else:
            mag = max([mag for mag in [source.mag_psf_g, source.mag_psf_r, source.mag_psf_i, source.mag_psf_z] if not np.isnan(mag)])
            probability_type = (4 * np.exp(-mag / 7))
    else:
        probability_type = 0.8
    
    return probability_type


def final_prob_list(source_list, color_from_lightcurve, object_from_lightcurve, coords, cat):
    
    #Check if the star field is crowded
    # is_crowded(cat, coords)

    probability_color = []
    probability_distance = []
    probability_magnitude = []
    probability_object = []

    for source in source_list:
        # Find the probability of each factor
        probability_color.append(is_color_match_single(color_from_lightcurve, source))
        probability_distance.append(distance_prob_single(source, coords))
        probability_magnitude.append(magnitude_prob_single(source))
        probability_object.append(is_type_match_single(object_from_lightcurve, source))

    # Modify the list based on the predicted colour from the lightcurve
    if color_from_lightcurve == "blue":
        probability_color = -probability_color   #Flip the list so that blue sources (negative g-r values) and valued higher
    
    #bias the colors (how to get around this?)
    probability_color = np.array(probability_color, dtype=np.float64)
    if color_from_lightcurve != "none":
        scaled_colors = (probability_color  - np.nanmin(probability_color)) / (np.nanmax(probability_color) - np.nanmin(probability_color)).tolist()
 
    # Combine the probabilities
    final_probabilities = [w * x * y * z for w, x, y, z in zip(probability_color, probability_magnitude, probability_distance,  probability_object)]

    return final_probabilities
