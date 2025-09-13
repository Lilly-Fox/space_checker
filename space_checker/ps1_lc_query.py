import pandas as pd

def isolate_stars(cat,only_stars=False,Qf_lim=0.85,psfkron_diff=0.05):
    qf_ind = ((cat.gQfPerfect.values > Qf_lim) & (cat.rQfPerfect.values > Qf_lim) & 
              (cat.iQfPerfect.values > Qf_lim) & (cat.zQfPerfect.values > Qf_lim))
    kron_ind = (cat.rMeanPSFMag.values - cat.rMeanKronMag.values) < psfkron_diff
    ind = qf_ind & kron_ind
    if only_stars:
        cat = cat.iloc[ind]
        cat.loc[:,'star'] = 1
    else:
        cat.loc[:,'star'] = 0
        cat.loc[ind,'star'] = 1
    return cat 

def query_ps1(ra,dec,radius,only_stars=False,version='dr2'):
    if (version.lower() != 'dr2') & (version.lower() != 'dr1'):
        m = 'Version must be dr2, or dr1'
        raise ValueError(m)
    
    url = f'https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/{version.lower()}/mean?ra={ra}&dec={dec}&radius={radius}&nDetections.gte=1&pagesize=-1&format=csv'
    # Changed to url as it was causing issues when doing str().
    try:
        cat = pd.read_csv(url)
    except pd.errors.EmptyDataError:
        print('No detections')
        cat = []
        return pd.DataFrame
    cat = isolate_stars(cat,only_stars=only_stars)
    return cat 

