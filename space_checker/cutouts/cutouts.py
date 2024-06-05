from astropy.table import Table
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt


def _Get_images(ra,dec,filters):
    
    """Query ps1filenames.py service to get a list of images"""
    
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    table = Table.read(url, format='ascii')
    return table

def _Get_url(ra, dec, size, filters, color=False):
    
    """Get URL for images in the table"""
    
    table = _Get_images(ra,dec,filters=filters)
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           f"ra={ra}&dec={dec}&size={size}&format=jpg")
   
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param,table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
    return url

def _Get_im(ra, dec, size,color):
    
    """Get color image at a sky position"""

    if color:
        url = _Get_url(ra,dec,size=size,filters='grz',color=True)
        r = requests.get(url)
    else:
        url = _Get_url(ra,dec,size=size,filters='i')
        r = requests.get(url[0])
    im = Image.open(BytesIO(r.content))
    return im

def _Panstarrs_phot(ra,dec,size):

    grey_im = _Get_im(ra,dec,size=size*4,color=False)
    colour_im = _Get_im(ra,dec,size=size*4,color=True)

    plt.rcParams.update({'font.size':12})
    plt.figure(1,(12,6))
    plt.subplot(121)
    plt.imshow(grey_im,origin="lower",cmap="gray")
    plt.title('PS1 i')
    plt.xlabel('px (0.25")')
    plt.ylabel('px (0.25")')
    plt.subplot(122)
    plt.title('PS1 grz')
    plt.imshow(colour_im,origin="lower")
    plt.xlabel('px (0.25")')
    plt.ylabel('px (0.25")')


def _Skymapper_phot(ra,dec,size):
    """
    Gets g,r,i from skymapper.
    """

    size /= 3600

    url = f"https://api.skymapper.nci.org.au/public/siap/dr2/query?POS={ra},{dec}&SIZE={size}&BAND=g,r,i&FORMAT=GRAPHIC&VERB=3"
    table = Table.read(url, format='ascii')

    # sort filters from red to blue
    flist = ["irg".find(x) for x in table['col3']]
    table = table[np.argsort(flist)]

    if len(table) > 3:
        # pick 3 filters
        table = table[[0,len(table)//2,len(table)-1]]

    plt.rcParams.update({'font.size':12})
    plt.figure(1,(12,6))

    plt.subplot(131)
    url = table[2][3]
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    plt.imshow(im,origin="upper",cmap="gray")
    plt.title('SkyMapper g')
    plt.xlabel('px (1.1")')

    plt.subplot(132)
    url = table[1][3]
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    plt.title('SkyMapper r')
    plt.imshow(im,origin="upper",cmap="gray")
    plt.xlabel('px (1.1")')

    plt.subplot(133)
    url = table[0][3]
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    plt.title('SkyMapper i')
    plt.imshow(im,origin="upper",cmap="gray")
    plt.xlabel('px (1.1")')

def event_cutout(coords,size=50,phot=None):

    if phot is None:
        if coords[1] > -10:
            phot = 'PS1'
        else:
            phot = 'SkyMapper'
        
    if phot == 'PS1':
        _Panstarrs_phot(coords[0],coords[1],size)

    elif phot.lower() == 'skymapper':
        _Skymapper_phot(coords[0],coords[1],size)

    else:
        print('Photometry name invalid.')
