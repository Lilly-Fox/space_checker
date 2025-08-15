#the design is very human

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


from prob_source_is_transient import find_distance



def _add_colored_sources(fig, coords, cat, top_3, error):
    """
    Adds colored sources to the given figure based on their probabilities and types.

    Parameters:
    fig (matplotlib.figure.Figure): The figure to which the sources will be added.
    coords (tuple): The coordinates (RA, Dec) of the target source.
    cat (pandas.DataFrame): The catalog containing the sources with their RA, Dec, and probabilities.
    top_3 (list): The indices of the 3 highest probabilities for the given transient.
    error (tuple or None): The error in the coordinates, used to draw an ellipse around the target source.

    Returns:
    matplotlib.figure.Figure: The figure with the added sources.
    """

    axs = fig.get_axes()  # Get the axes from the figure derived from event_cutout
    cmap = plt.cm.RdYlGn  # Define the red-yellow-green colormap

    count = 0  # Used for the legend

    for ax in axs:  # Loop through each axis in the figure

        # 'x' marks the spot for the center of the observed TESS transient
        ax.scatter(coords[0], coords[1], transform=ax.get_transform('fk5'),
                   edgecolors='w', marker='x', s=30, facecolors='w', linewidths=1, label='Target')

        # Create an ellipse the size of a TESS pixel
        if error is not None:  # Check if error is provided
            xerr, yerr = error if len(error) > 1 else (error, error)  # Set xerr and yerr based on error length
            ellipse = Ellipse(xy=(coords[0], coords[1]), width=error[0], height=error[1],
                              edgecolor='white', facecolor='none', linestyle=':', linewidth=2,
                              transform=ax.get_transform('fk5'))
            ax.add_patch(ellipse)

        # Scatter the sources from the catalog with their color based on their probability of being the observed transient
        for star_type, marker, label in zip([1, 0, 2], ['^', 'o', 'D'], ['Star', 'Galaxy', 'Possible galaxy']):  # Loop through star types and markers
            objects = cat.loc[cat['star'] == star_type]
            norm = mcolors.Normalize(vmin=min(objects.final_probabilities), vmax=max(objects.final_probabilities))
            ax.scatter(objects.ra, objects.dec, transform=ax.get_transform('fk5'),
                       edgecolors=cmap(norm(objects.final_probabilities)), marker=marker, s=80,
                       facecolors='none', label=label)

        # colour highest chance
        ax.scatter(cat.iloc[top_3[0]].ra, cat.iloc[top_3[0]].dec, transform=ax.get_transform('fk5'),
                   edgecolors="xkcd:neon purple", marker='x', s=60,
                   facecolors='xkcd:neon purple', label='Most likely')

        # if count == 0:  # Add legend only for the first axis
            # legend = ax.legend(loc=2, facecolor='black', fontsize=10)
            # for text in legend.get_texts():
            #     text.set_color('white')

        count += 1
    
    # Create a legend that only displays the coloured version of the sources
    series, labels = axs[0].get_legend_handles_labels()
    # print(series, labels)
    legend = axs[0].legend(series[int((len(series) - 1)/2):], labels[int((len(series) - 1)/2):], loc=2, facecolor='black', fontsize=10)
    for text in legend.get_texts():
        text.set_color('white')

    return fig


def colored_figure(testcase, cat, indices):
    """
    Generates a figure with colored sources based on their probabilities and types.

    Parameters:
    testcase (tuple): A tuple containing the filename and coordinates (RA, Dec) of the target source.

    Returns:
    None
    """
    # Unpack the information
    filename, coords = testcase

    # Define the error (approx. 20 arcseconds ie size of TESS pixel)
    error = (0.005556, 0.005556) 
    fig, wcs, outsize, phot, cat_0 = event_cutout(coords, real_loc=None, error=error, size=100, phot=None)

    # Add the colored sources to the figure (where the color is based on the probabilities)
    # Commented out while the DESI servers are offline (back ~2 feb)
    fig = _add_colored_sources(fig, coords, cat, indices, error)

    # Save the figure
    # fig.savefig(f"{filename}.png")
    plt.show()


def user_input(cat, transient_index):
    while True:
        response = input("Does the most likely source match the transient and does the image look reasonable? (y/n/m/q/skip): ").strip().lower()
        if response in ["y", "n", "m"]:
            cat['human_check'] = response
            return cat, transient_index
        elif response == "q":
            print("Quitting the program.")
            sys.exit()

        elif response == "skip":
            while True:
                transient_index = input("Skip to what index? (Indices start at 0)").strip().lower()
                if re.match(r'^\d+$', transient_index):
                    transient_index = int(transient_index)
                    print("\n Skipping to index:", int(transient_index) + 1)
                    
                else:
                    print("Invalid input. Please enter a valid number.")

                print(f"Skipping to a transient number {transient_index}")
                cat['human_check'] = "None"
                return cat, transient_index
        else:
            print("Invalid input. Please enter 'y' for yes, 'n' for no, 'm' for unsure/maybe, 'q' for quit program, or 'skip' for skipping to a different index.")

def human_check(cat, indices, testcase, object_from_lightcurve, color_from_lightcurve, transient_index):
    "displays the result of this notebook and asks a person if this seems right"
    filename, event_coords = testcase
    print(f"\nDetermined from lightcurve:\n Object type - {object_from_lightcurve}\n Transient color - {color_from_lightcurve}\n")

    source_list = [source.tolist() for index, source in cat.iterrows()]  # Extract a list of the source IDs
    index = indices[0]
    source = source_list[index]

    distance = find_distance(source, event_coords[0], event_coords[1])
    object_source = source[23]
    obj_types = ['Star', 'Galaxy', 'Possible galaxy (more likely star)']
    color = source[2] - source[3]
    color = "blue" if color < 0 else "red"
    print(f"Most likely source: \n Distance = {distance:.3f}arcsec \n Object type = {obj_types[object_source]}\n color = {color}\n Program confidence = {source[24]:.3f}\n")

    try:
        # Suppress specific warnings
        warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
        # Redirect standard output to null
        sys.stdout = open(os.devnull, 'w')
        colored_figure(testcase, cat, indices)
        # Restore standard output
        sys.stdout = sys.__stdout__
    except Exception as e:
        # Restore standard output in case of an exception
        sys.stdout = sys.__stdout__
        print(f"Couldn't get colored figure: {e}")

    cat, transient_index = user_input(cat, transient_index)

    return cat, transient_index