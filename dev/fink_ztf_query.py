import pandas as pd
import io
import requests
import matplotlib.pyplot as plt
import astropy.time as time
import time as t
import numpy as np
 
TESS = pd.read_csv('mastercat_new.csv')
 
for i in TESS.loc.index:
  print(i, end="\r")
  try:
    r = requests.post(
      "https://api.fink-portal.org/api/v1/conesearch",
      json={
        "ra": str(TESS.ra[i]),
        "dec": str(TESS.dec[i]),
        "radius": "50",
        "startdate": time.Time(TESS.mjd[i]-2, format='mjd').iso,
        "window": 10 # in days
      }
    )
  except:
    t.sleep(30)
    r = requests.post(
      "https://api.fink-portal.org/api/v1/conesearch",
      json={
        "ra": str(TESS.ra[i]),
        "dec": str(TESS.dec[i]),
        "radius": "50",
        "startdate": time.Time(TESS.mjd[i]-2, format='mjd').iso,
        "window": 10 # in days
      }
    )
 
  pdf = pd.read_json(io.BytesIO(r.content))
  if len(pdf) > 0:
    print('\n',i,np.array(pdf['i:objectId']),np.array(pdf['d:classification']))
    TESS.loc[i, 'ZTF_xmatch'] = pdf['i:objectId'][0]
    TESS.loc[i, 'classification'] = pdf['d:classification'][0]
 
TESS[TESS.ZTF_xmatch != ''].reset_index(drop=True).to_csv("sector48_mastercat_xmatch.csv", index=False)