import numpy as np
from astropy.table import Table
from gcwork import starset
from gcwork import orbits
import os
import json
import hashlib

def get_cache_path(align_root, filename):
    """Generate cache path based on input path"""
    # Create cache directory in the app directory
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(app_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create hash of align_root to use as filename
    path_hash = hashlib.md5(align_root.encode()).hexdigest()
    return os.path.join(cache_dir, f"{path_hash}_{filename}.json")

def load_data(align_root, center_star=None, range=0.4, xcenter=0, ycenter=0):
    """Load star position data from alignment files or cache"""
    cache_path = get_cache_path(align_root, 'star_data')
    from_cache = False
    
    # Try to load from cache first
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            data = json.load(f)
            from_cache = True
            return data, from_cache
    
    # If no cache, load from original source
    # Create starset object
    s = starset.StarSet(align_root)
    
    # Get position and velocity data
    name = np.array(s.getArray('name')).tolist()
    x = np.array(s.getArray('x')).tolist()
    y = np.array(s.getArray('y')).tolist()
    xe = np.array(s.getArray('xerr')).tolist()
    ye = np.array(s.getArray('yerr')).tolist()
    vx = np.array(s.getArray('vx')).tolist()
    vy = np.array(s.getArray('vy')).tolist()
    vxe = np.array(s.getArray('vxerr')).tolist()
    vye = np.array(s.getArray('vyerr')).tolist()
    mag = np.array(s.getArray('mag')* 1.0).tolist()
    # Get number of epochs
    nEpochs = np.array(s.getArray('velCnt')).tolist()
    
    data = {
        'name': name,
        'x': x,
        'y': y,
        'xe': xe,
        'ye': ye,
        'vx': vx,
        'vy': vy,
        'vxe': vxe,
        'vye': vye,
        'mag': mag,
        'nEpochs': nEpochs
    }
    
    # Save to cache
    with open(cache_path, 'w') as f:
        json.dump(data, f)
    
    return data, from_cache

def find_star_coordinates(data, star_name):
    """Find star coordinates by name"""
    try:
        idx = data['name'].index(star_name)
        return {
            'found': True,
            'x': data['x'][idx],
            'y': data['y'][idx]
        }
    except ValueError:
        return {'found': False}

def load_orbits(orbits_file, tStart=1994., tEnd=2020., dt=0.01):
    """Load orbit data from file or cache"""
    if not orbits_file:
        return None, None, False
        
    cache_path = get_cache_path(orbits_file, 'orbit_data')
    from_cache = False
    
    # Try to load from cache first
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cached_data = json.load(f)
            return cached_data['orbits'], cached_data['names'], True
    
    # If no cache, load from original source
    t = np.linspace(tStart, tEnd, int(np.ceil((tEnd - tStart) / dt)))

    if orbits_file is None:
        tab = np.genfromtxt(orbits_file, dtype=str)
    else:
        tab = np.genfromtxt(orbits_file, dtype=str)

    res = []
    names = []
    for star in tab:
        orb = orbits.Orbit()
        orb.p = float(star[1])
        orb.t0 = float(star[3])
        orb.e = float(star[4])
        orb.i = float(star[5])
        orb.o = float(star[6])
        orb.w = float(star[7])

        (x, v, a) = orb.kep2xyz(epochs=t, mass=4.e6, dist=8000.)  # possible to change mass and R0 here if needed
        names.append(star[0])
        res.append(x[:, 0:2].tolist())  # Convert numpy array to list

    # Save to cache
    with open(cache_path, 'w') as f:
        json.dump({'orbits': res, 'names': names}, f)
    
    return res, names, from_cache