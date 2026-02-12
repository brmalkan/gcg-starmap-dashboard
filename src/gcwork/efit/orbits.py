import numpy as np
import efit5_results as ef
from math import pi


def input(orb,GM,R0,pos=(0,0,0,0,0),Redshift=None):
    ''' 
        Create an object "star" from efit5_results and insert the parameters as a 1 element long chain with the parameter we want to use
    '''
    if Redshift==None:
        BH = ef.BH('Kepler')
    else:
        BH = ef.BH('Kepler_Redshift')
        
    BH.set_all_param(GM,R0,pos[0],pos[1],pos[2],pos[3],pos[4],Redshift)
        
    star = ef.star('my_star',BH)
    star.set_chain_from_param(np.array([orb[0]]),'P')
    star.set_chain_from_param(np.array([orb[1]]),'e')
    star.set_chain_from_param(np.array([orb[2]]),'T0')
    star.set_chain_from_param(np.array([orb[3]*pi/180.]),'i')
    star.set_chain_from_param(np.array([orb[4]*pi/180.]),'w')
    star.set_chain_from_param(np.array([orb[5]*pi/180.]),'O')
    return star
    
def astro(epochs,orb,GM,R0,pos=(0,0,0,0,0),Redshift=None):
    '''
    Input:
    ------
        epochs: numpy array of the epochs in Julian year
        orb: (P,e,T0,inc,w,Om) in (yr,-,yr,deg,deg,deg)
        GM: in units of GM sun
        R0: in pc
        pos: (x0,y0,vx0,vy0,vz0) in (as,as,as/yr,as/yr,km/s)
        Redshift: if =None: kepler model used. If it's a double, kepler + Romer time delay model used.
        
    return
    ------
    (x,y): 2 numpy array of the x and y position (in camera coord.) in arcsecond
    '''
    return input(orb,GM,R0,pos,Redshift).get_astro(epochs)
    
    

def RV(epochs,orb,GM,R0,pos=(0,0,0,0,0),Redshift=None):
    '''
    Input:
    ------
        epochs: numpy array of the epochs in Julian year
        orb: (P,e,T0,inc,w,Om) in (yr,-,yr,deg,deg,deg)
        GM: in units of GM sun
        R0: in pc
        pos: (x0,y0,vx0,vy0,vz0) in (as,as,as/yr,as/yr,km/s)
        Redshift: if =None: kepler model used. If it's a double, kepler + Romer time delay and redshift model used and the redshift parameter is given by Redshift
        
    return
    ------
    RV: 1 numpy array of the RV in km/s
    '''
    return input(orb,GM,R0,pos,Redshift).get_RV(epochs)
     

def orbit(epochs,orb,GM,R0,Romer=False):
    '''
    Input:
    ------
        epochs: numpy array of the epochs in Julian year
        orb: (P,e,T0,inc,w,Om) in (yr,-,yr,deg,deg,deg)
        GM: in units of GM sun
        R0: in pc
        Romer: Use the Romer time delay in the model (the redshift is never included here. It's not a physical effect on the star's velocity.)
        The BH is assumed to be at the center.
        
    return
    ------
        (x,y,z): 3D position of the star in AU.
        (vx,vy,vz): 3D velocity of the star in AU/yr.
        The Romer time delay is included 
    '''
    return input(orb,GM,R0,Redshift= None if Romer==False else 0.).orbit(epochs)