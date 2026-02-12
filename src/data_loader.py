import numpy as np

from gcwork import starset
from gcwork import orbits

def load_starset_data(data_filepath: str) -> dict[str, list[float]]:
    """
    Load star position data from alignment files
    """

    s = starset.StarSet(data_filepath)

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

    return data

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

def load_orbits(
    orbits_file: str, tStart=1994., tEnd=2020., dt=0.01
) -> tuple[list[any], list[any]]:
    """
    Load orbit data from file
    """
    if not orbits_file:
        return None, None, False

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

    return res, names
