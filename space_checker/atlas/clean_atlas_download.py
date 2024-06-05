#!/usr/bin/env python
'''
Code adapted from Qinan Wang and Armin Rest by Sofia Rest
'''

import configparser, sys, argparse, requests, re, time, io, math, os
import pandas as pd
import numpy as np
from getpass import getpass
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import Angle, SkyCoord
from copy import deepcopy
from pdastro import pdastrostatsclass, AorB, AnotB
from atlas_lc import atlas_lc

def RaInDeg(ra):
	s = re.compile('\:')
	if isinstance(ra,str) and s.search(ra):
		A = Angle(ra, u.hour)
	else:
		A = Angle(ra, u.degree)
	return(A.degree)
		   
def DecInDeg(dec):
	A = Angle(dec, u.degree)
	return(A.degree)

class download_atlas_lc:
	def __init__(self,ra,dec,discdate,name,save_path='.',
				 mjd_min=50000.0,mjd_max = float(Time.now().mjd),
				 flux2mag_sigmalimit=3, radius=17,num_controls = 3,
				 controls=True,overwrite=False,run=True,verbose=True):
		#vars 
		self.ra = RaInDeg(ra)
		self.dec = DecInDeg(dec)
		self.discdate = discdate
		self.name = name
		self.verbose = verbose

		# credentials
		self.username = 'cheerfuluser'
		self.password = '53ACnTmkqk@ApXN'
		self.token = self.connect_atlas()
		if self.token is None: 
			raise RuntimeError('ERROR in connect_atlas(): No token header!')

		# input/output
		self.output_dir = save_path
		self.overwrite = overwrite
		self.flux2mag_sigmalimit = flux2mag_sigmalimit

		self.mjd_min = mjd_min#50000.0
		self.mjd_max = mjd_max#float(Time.now().mjd)

		# other
		self.controls = controls
		self.control_coords = pdastrostatsclass()
		self.radius = radius
		self.num_controls = num_controls
		self.closebright = False
		self.closebright_coords = None
		self.closebright_min_dist = 3
		self.control_coords_filename = None
		if run:
			self.download_lcs()
			
	def connect_atlas(self):
		baseurl = 'https://fallingstar-data.com/forcedphot'
		resp = requests.post(url=f"{baseurl}/api-token-auth/",data={'username':self.username,'password':self.password})
		if resp.status_code == 200:
			token = resp.json()['token']
			print(f'Token: {token}')
			headers = {'Authorization':f'Token {token}','Accept':'application/json'}
		else:
			raise RuntimeError(f'ERROR in connect_atlas(): {resp.status_code}')
			print(resp.json())
		return headers

	# API GUIDE: https://fallingstar-data.com/forcedphot/apiguide/
	def get_result(self, ra, dec):
		if self.mjd_min > self.mjd_max:
			raise RuntimeError(f'ERROR: max MJD {self.mjd_max} less than min MJD {self.mjd_min}')
			sys.exit()
		else:
			print(f'Min MJD: {self.mjd_min}; max MJD: {self.mjd_max}')
		
		baseurl = 'https://fallingstar-data.com/forcedphot'
		task_url = None
		while not task_url:
			with requests.Session() as s:
				resp = s.post(f"{baseurl}/queue/",headers=self.token,
							  data={'ra':ra,'dec':dec,'send_email':False,
							  	    "mjd_min":self.mjd_min,"mjd_max":self.mjd_max})
				if resp.status_code == 201: 
					task_url = resp.json()['url']
					print(f'Task url: {task_url}')
				elif resp.status_code == 429:
					message = resp.json()["detail"]
					print(f'{resp.status_code} {message}')
					t_sec = re.findall(r'available in (\d+) seconds', message)
					t_min = re.findall(r'available in (\d+) minutes', message)
					if t_sec:
						waittime = int(t_sec[0])
					elif t_min:
						waittime = int(t_min[0]) * 60
					else:
						waittime = 10
					print(f'Waiting {waittime} seconds')
					time.sleep(waittime)
				else:
					print(f'ERROR {resp.status_code}')
					print(resp.text)
					sys.exit()
		
		result_url = None
		taskstarted_printed = False
		
		print('Waiting for job to start...')
		while not result_url:
			with requests.Session() as s:
				resp = s.get(task_url, headers=self.token)
				if resp.status_code == 200: 
					if not(resp.json()['finishtimestamp'] is None):
						result_url = resp.json()['result_url']
						print(f"Task is complete with results available at {result_url}")
						break
					elif resp.json()['starttimestamp']:
						if not taskstarted_printed:
							print(f"Task is running (started at {resp.json()['starttimestamp']})")
							taskstarted_printed = True
						time.sleep(2)
					else:
						#print(f"Waiting for job to start (queued at {resp.json()['timestamp']})")
						time.sleep(4)
				else:
					print(f'ERROR {resp.status_code}')
					print(resp.text)
					sys.exit()
			
		with requests.Session() as s:
			if result_url is None:
				print('WARNING: Empty light curve--no data within this MJD range...')
				dfresult = pd.DataFrame(columns=['MJD','m','dm','uJy','duJy','F','err','chi/N','RA','Dec','x','y','maj','min','phi','apfit','Sky','ZP','Obs','Mask'])
			else:
				result = s.get(result_url, headers=self.token).text
				dfresult = pd.read_csv(io.StringIO(result.replace("###", "")), delim_whitespace=True)
		
		return dfresult

	def get_filt_lens(self, sn_lc):
		if len(sn_lc.lcs) > 0: # SN lc has been downloaded
			return sn_lc.get_filt_lens()
		else: # SN lc has not been downloaded, so temporarily load both c and o lcs to get indices
			temp_o = pdastrostatsclass()
			temp_o.load_spacesep(sn_lc.get_filename('o', 0, self.output_dir), delim_whitespace=True)
			o_len = len(temp_o.t)

			temp_c = pdastrostatsclass()
			temp_c.load_spacesep(sn_lc.get_filename('c', 0, self.output_dir), delim_whitespace=True)
			c_len = len(temp_c.t)
			return o_len, c_len

	# convert ra string to angle
	def ra_str2angle(self, ra):
		return Angle(RaInDeg(ra), u.degree)

	# convert dec string to angle
	def dec_str2angle(self, dec):
		return Angle(DecInDeg(dec), u.degree)

	# get distance between 2 locations specified by ra and dec angles
	def get_distance(self, ra1, dec1, ra2, dec2):
		c1 = SkyCoord(ra1, dec1, frame='fk5')
		c2 = SkyCoord(ra2, dec2, frame='fk5')
		return c1.separation(c2)


	# get RA and Dec coordinates of control light curves in a circle pattern around SN location and add to control_coords table
	def get_control_coords(self, sn_lc):
		self.control_coords.t = pd.DataFrame(columns=['name','control_id','ra','dec','ra_offset','dec_offset','radius_arcsec','n_detec','n_detec_o','n_detec_c'])

		sn_ra = self.ra_str2angle(sn_lc.ra)
		sn_dec = self.dec_str2angle(sn_lc.dec)
		o_len, c_len = self.get_filt_lens(sn_lc)

		# set first row of control_coords table according to closebright status
		if not self.closebright: # pattern around SN location
			r = Angle(self.radius, u.arcsec)

			# circle pattern center is SN location
			ra_center = sn_ra
			dec_center = sn_dec

			# add SN coordinates as first row
			self.control_coords.newrow({'name':deepcopy(sn_lc.name),
										'control_id':0,
										'ra': f'{sn_ra.degree:0.8f}',
										'dec': f'{sn_dec.degree:0.8f}',
										'ra_offset':0,
										'dec_offset':0,
										'radius_arcsec':0,
										'n_detec':o_len+c_len,
										'n_detec_o':o_len,
										'n_detec_c':c_len})
		
		else: # pattern around close bright object
			# coordinates of close bright object
			cb_ra = self.ra_str2angle(self.closebright_coords[0])
			cb_dec = self.dec_str2angle(self.closebright_coords[1])

			# circle pattern radius is distance between SN and bright object
			r = self.get_distance(sn_ra, sn_dec, cb_ra, cb_dec).arcsecond

			# circle pattern center is close bright object location
			ra_center = cb_ra
			dec_center = cb_dec

			# add SN coordinates as first row; columns like ra_offset, dec_offset, etc. do not apply here
			self.control_coords.newrow({'name':sn_lc.name,
										'control_id':0,
										'ra': f'{sn_ra.degree:0.8f}',
										'dec': f'{sn_dec.degree:0.8f}',
										'ra_offset':np.nan,
										'dec_offset':np.nan,
										'radius_arcsec':np.nan,
										'n_detec':o_len+c_len,
										'n_detec_o':o_len,
										'n_detec_c':c_len},ignore_index=True)

		
		 # calculate control lc coordinates
		for i in range(1,self.num_controls+1):
			angle = Angle(i*360.0 / self.num_controls, u.degree)
			
			ra_distance = Angle(r.degree * math.cos(angle.radian), u.degree)
			ra_offset = Angle(ra_distance.degree * (1.0/math.cos(dec_center.radian)), u.degree)
			ra = Angle(ra_center.degree + ra_offset.degree, u.degree)

			dec_offset = Angle(r.degree * math.sin(angle.radian), u.degree)
			dec = Angle(dec_center.degree + dec_offset.degree, u.degree)

			if self.closebright: # check to see if control light curve location is within minimum distance from SN location
				offset_sep = self.get_distance(sn_ra, sn_dec, ra, dec).arcsecond
				if offset_sep < self.closebright_min_dist:
					print(f'Control light curve {i+1:3d} too close to SN location ({offset_sep}\" away) with minimum distance to SN as {self.closebright_min_dist}; skipping control light curve...')
					continue

			# add RA and Dec coordinates to control_coords table
			self.control_coords.newrow({'name':np.nan,
										'control_id':i,
										'ra': f'{ra.degree:0.8f}',
										'dec': f'{dec.degree:0.8f}',
										'ra_offset': f'{ra_offset.degree:0.8f}',
										'dec_offset': f'{dec_offset.degree:0.8f}',
										'radius_arcsec':f'{r.arcsecond}',
										'n_detec':np.nan,
										'n_detec_o':np.nan,
										'n_detec_c':np.nan})

			with pd.option_context('display.float_format', '{:,.8f}'.format):
				print('Control light curve coordinates calculated: \n',self.control_coords.t[['name','control_id','ra','dec','ra_offset','dec_offset','radius_arcsec']])

	# update number of control light curve detections in control_coords table
	def update_control_coords(self, lc, control_index):
		o_ix = lc.lcs[control_index].ix_equal(colnames=['F'],val='o')
		self.control_coords.t.loc[control_index,'n_detec'] = len(lc.lcs[control_index].t)
		self.control_coords.t.loc[control_index,'n_detec_o'] = len(o_ix)
		self.control_coords.t.loc[control_index,'n_detec_c'] = len(AnotB(lc.lcs[control_index].getindices(),o_ix))


	# download a single light curve
	def download_lc(self, lc, ra, dec, control_index=0):	
		if self.verbose:
			print(f'Downloading forced photometry light curve at {RaInDeg(ra):0.8f}, {DecInDeg(dec):0.8f} from ATLAS')
		lc.lcs[control_index] = pdastrostatsclass()

		while(True):
			try:
				lc.lcs[control_index].t = self.get_result(RaInDeg(ra), DecInDeg(dec))
				break
			except Exception as e:
				print('Exception caught: '+str(e))
				print('Trying again in 20 seconds! Waiting...')
				time.sleep(20)
				continue

		# sort data by mjd
		lc.lcs[control_index].t = lc.lcs[control_index].t.sort_values(by=['MJD'],ignore_index=True)

		# remove rows with duJy=0 or uJy=Nan
		dflux_zero_ix = lc.lcs[control_index].ix_inrange(colnames='duJy',lowlim=0,uplim=0)
		flux_nan_ix =lc.lcs[control_index].ix_is_null(colnames='uJy')
		print('\nDeleting %d rows with "duJy"==0 or "uJy"==NaN...' % (len(dflux_zero_ix) + len(flux_nan_ix)))
		if len(AorB(dflux_zero_ix,flux_nan_ix)) > 0:
			lc.lcs[control_index].t = lc.lcs[control_index].t.drop(AorB(dflux_zero_ix,flux_nan_ix))
			 
		lc.lcs[control_index].flux2mag('uJy', 'duJy', 'm', 'dm', zpt=23.9, upperlim_Nsigma=self.flux2mag_sigmalimit)

		return lc

	# download SN light curve and, if necessary, control light curves, then save
	def download_lcs(self):
		lc = atlas_lc(ra = self.ra,dec = self.dec,discdate= self.discdate,name=self.name)
		print(f'\nCOMMENCING LOOP FOR SN {lc.name}\n')

		# only download light curve if overwriting existing files
		if not(self.overwrite) and lc.exists(self.output_dir, 'o') and lc.exists(self.output_dir, 'c'):
			print(f'SN light curve files already exist and overwrite is set to {self.overwrite}! Skipping download...')
		else:
			lc = self.download_lc(lc, lc.ra, lc.dec)
			lc._save_lc(self.output_dir, overwrite=self.overwrite)

		if self.controls:
			print('Control light curve downloading set to True')
			
			self.get_control_coords(lc)

			# download control light curves
			for control_index in range(1,len(self.control_coords.t)):
				# only download control light curve if overwriting existing files
				if not(self.overwrite) and lc.exists(self.output_dir, 'o', control_index=control_index) and lc.exists(self.output_dir, 'c', control_index=control_index):
					print(f'Control light curve {control_index:03d} files already exist and overwrite is set to {self.overwrite}! Skipping download...')
				else:
					print(f'\nDownloading control light curve {control_index:03d}...')
					lc = self.download_lc(lc, 
										  ra=self.control_coords.t.loc[control_index,'ra'], 
										  dec=self.control_coords.t.loc[control_index,'dec'], 
										  control_index=control_index)
					self.update_control_coords(lc, control_index)
					lc._save_lc(self.output_dir, control_index=control_index, overwrite=self.overwrite)

			# save control_coords table
			if self.overwrite:
				self.control_coords.write(filename=f'{self.output_dir}/{lc.name}/controls/{lc.name}_control_coords.txt', overwrite=self.overwrite)