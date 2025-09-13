import warnings
from astropy import units as u
from astropy.coordinates import SkyCoord

from external_photometry import event_cutout
#from external_photometry import _DESI_phot
#from external_photometry import _delve_objects

# from astroquery.vizier import Vizier
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, convolve
import heapq

from matplotlib.markers import MarkerStyle
import matplotlib.colors as mcolors

import os
import csv
import sys
import warnings
import re

from lightcurve_classifier_for_tess import tess_lightcurve
from prob_source_is_transient import final_prob_list
from human_component import human_check



def identifier_creator(source, filename):

	data = pd.read_csv(filename, names=["time_MJD", "counts", "event_flag"], skiprows=1).apply(pd.to_numeric, errors='coerce')
	event_data = data[data["event_flag"] != 0.0]
	MJD = event_data["time_MJD"].iloc[0]

	source_ra, source_dec = source[2], source[3]

	coord = SkyCoord(ra=source_ra*u.degree, dec=source_dec*u.degree, frame='icrs')
	angle_source_ra = coord.ra.to_string(unit=u.hour, sep=':', precision=0, pad=True)[:5]
	angle_source_dec = coord.dec.to_string(unit=u.degree, sep=':', precision=0, alwayssign=True, pad=True)[:6]

	# Adjust MJD precision to 5 characters
	MJD = f"{MJD:.0f}"


	identifier = "TSS_J" + angle_source_ra + angle_source_dec +"T" + MJD 
	return identifier



class prob_association():
	def __init__(self,ra,dec,era=None,edec=None,lc=None,verbose=1,zp=25):
		self.ra = np.array(ra)
		self.dec = np.array(dec)
		self.era = era
		self.edec = edec
		self.lc = np.array(lc)
		self.verbose = verbose

		self._check_params()



	def _check_params(self):
		if (len(self.ra) != len(self.dec)):
			raise ValueError('ra and dec must be the same length')
		if (self.lc is None) & (self.verbose > 0):
			print('No lightcurve specified')
		else:
			if (len(self.ra) != len(self.lc)):
				m = 'The coordinates and lc must have the same length\n len(ra)={len(self.ra)\nlen(lc)={len(self.lc)}}'
				raise ValueError(m)
		if self.era is None:
			m = "No positional error provided, assuming an error of 5''"
			print(m)
			self.era = np.ones_like(self.ra) * 5
			self.edec = np.ones_like(self.ra) * 5

	def get_external(self,ra = None, dec = None, pos_error = None,index=None,imsize=100):
		if ra is None:
			ra = self.ra 
			dec = self.dec
		if pos_error is None:
			pos_error = [self.era,self.edec]
		
		if len(ra) > 1:
			if index is None:
				m = '!! Only 1 coordinate can be queried, and no index is set.\nSetting index = 0 !!'
				warnings.warn(m)
				index = 0
			ra = ra[index]; dec = dec[index]; pos_error = pos_error[index]

		coords = [ra,dec]

		fig, wcs, outsize, phot, cat = event_cutout(coords,error=pos_error,size=imsize)

		cat['g-r'] = cat['gmag'] - cat['rmag']

		self.phot_wcs = wcs
		self.fig = fig 
		self.instrument = phot
		self.cat = cat



	def _compute_distance_prob(self,index=None):
		if self.cat is None:
			raise ValueError('No catalog is specified')
		if len(self.ra) > 1:
			if index is None:
				m = '!! Only 1 coordinate can be queried, and no index is set.\nSetting index = 0 !!'
				warnings.warn(m)
				index = 0
		ra = self.ra[index]; dec = self.dec[index]; era = self.era[index]; edec = self.edec[index]
		dist = np.sqrt((ra - self.cat['ra'].values)**2 + (dec - self.cat['dec'].values)**2)
		dra = ((ra - self.cat['ra'].values) / era)**2
		ddec = ((dec - self.cat['dec'].values) / edec)**2
		sig = np.exp(-1/2*(dra+ddec))
		self.cat['dist'] = dist
		self.cat['dist_sig'] = sig

	def _lightcurve_analysis(self,lc=None,index=None):
		if lc is None:
			if self.lc is None:
				m = 'No lc specified'
				raise ValueError(m)
			else:
				lc = self.lc
		if type(lc) == type(np.array(1)):
			if index is None:
				m = 'Only 1 lc can be processed at a time, and no index is set.\nSetting index = 0'
				warnings.warn(m)
				index = 0
			lc = lc[index]
		events = np.nanmax(lc['event'])
		duration = []
		brightness = []
		std = []
		rise_grad = []
		fall_grad = []
		fall_grad_std = []
		rise_time = []
		fall_time = []
		datapoints = []
		index = np.arange(1,events+1,dtype=int)
		for i in index:
			eind = convolve(lc['event'].values == i,np.ones(2))		
			event = lc.iloc[eind == 1]
			maxind = np.nanargmax(event['counts'].values)

			duration += [event['mjd'].iloc[-1] - event['mjd'].iloc[0]]
			brightness += [np.nanmax(event['counts'].values)]
			std += [np.nanstd(event['counts'].values)]
			try:
				grad = np.nanmedian(np.gradient(event['counts'].values[:maxind+1]))
			except:
				grad = np.nan
			rise_grad += [grad]
			rise_time += [event['mjd'].values[maxind] - event['mjd'].values[0]]
			fall_grad += [np.nanmedian(np.gradient(event['counts'].values[maxind:]))]
			fall_grad_std += [np.nanstd(np.gradient(event['counts'].values[maxind:]))]
			fall_time += [abs(event['mjd'].values[maxind] - event['mjd'].values[-1])]
			datapoints += [len(event)]
		data = {'event':index,'duration':duration,'max_counts':brightness,
				'std':std,'datapoints':datapoints,'rise_time':rise_time,'fall_time':fall_time,
				'rise_grad':rise_grad,'fall_grad':fall_grad,'fall_grad_std':fall_grad_std}

		event_stats = pd.DataFrame(data)
		event_stats['time_ratio'] = event_stats['rise_time'] / event_stats['fall_time']
		event_stats['grad_ratio'] = event_stats['rise_grad'] / event_stats['fall_grad']
		event_stats['total_events'] = int(events)
		self.event_stats = event_stats


	def _lc_types(self):

		self.event_stats['flaring'] = self.event_stats['total_events'].values > 1
		self.event_stats['fast'] =  self.event_stats['duration'].values < 2/24

		active_star = self.event_stats['flaring'] & self.event_stats['fast']
		#self.event_stats['grb'].iloc[active_star] = 0.1 As these lines aren't in the dataframe
		#self.event_stats['flare'].iloc[active_star] = 0.9

		nova = ~self.event_stats['fast']
		self.event_stats['nova'].iloc[nova] = 1

	def _cat_probs(self):
		pass


