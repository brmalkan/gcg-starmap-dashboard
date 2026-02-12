import numpy as np
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp

J2000=51544.5 # in MJD
AU2KM=149597870.700
YR2DAY=365.25
YR2SEC=YR2DAY*86400.
AUYR2KMS=AU2KM/YR2SEC
GMSUN=365.25*365.25*0.2959122082855911e-3 #AU^3 / yr^2
GMSUN_4pi2=GMSUN/(4.*math.pi*math.pi)
CMKS=299792.0 #speed of light in km/s
CAUYR=CMKS/AUYR2KMS #speed of light in AU/yr
TORIGIN = 2000.
#MAGSGRA = 16.3 #magnitude of SgrA*
#FLUXSGRA = 10.**(-0.4*MAGSGRA)
#BIASSIGMA= 0.0267536367         #assuming that FWHM is 0.063 [as] (from PSF structure)

GLOB_PAR = ['GM','x0','y0','vx0','vy0','vz0','R0','EM','GR','Redshift']
STARS_PAR = ['O','w','i','P','T0','e']

def fromMJD2yr(m):
    """
        Transform from MJD to Julian years
        Parameters
        ----------
        m: numpy array of epochs in MJD
        
        Return
        ------
        numpyn array of epochs in Julian Years
        """
    return np.array(2000. + (m - J2000)/YR2DAY)



def fromyr2MJD(y):
    """
        Transform from Julian years to MJD
        Parameters
        ----------
        y: numpy array of epochs in Julian Years
        
        Return
        ------
        numpyn array of epochs in MJD
        """
    return np.array(  J2000 + (y-2000.)*YR2DAY)



def weighted_percentile(data, percents, weights=None):
    ''' percents
        weights specifies the frequency (count) of data.
        '''
    if weights is None:
        return np.percentile(data, 100*percents)
    ind=np.argsort(data)
    d=data[ind]
    w=weights[ind]
    p=1.*w.cumsum()/w.sum()
    y=np.interp(percents, p, d)
    
    return y

def weighted_avg(values, weights):
    """
        Returns the weighted average and standard deviation.
        
        values, weights -- Numpy ndarrays with the same shape.
        """
    
    average = np.average(values, weights=weights)
    variance = np.dot(weights, (values-average)**2)/weights.sum()
    
    return [average, math.sqrt(variance)]

def getContourLevels(probDist,percents = np.array([0.6827, .95, .997])):
    """
        If we want to overlay countours, we need to figure out the
        appropriate levels. The algorithim is:
        1. Sort all pixels in the 2D histogram (largest to smallest)
        2. Make a cumulative distribution function
        3. Find the level at which 68% of trials are enclosed.
        """
    # Get indices for sorted pixel values (smallest to largest)
    sid0 = probDist.flatten().argsort()
    # Reverse indices, now largest to smallest
    sid = sid0[::-1]
    # Sort the actual pixel values
    pixSort = probDist.flatten()[sid]
    
    # Make a cumulative distribution function starting from the
    # highest pixel value. This way we can find the level above
    # which 68% of the trials will fall.
    cdf = np.cumsum(pixSort)
    cdf = cdf/max(cdf)
    
    # Determine point at which we reach 68% level
    
    levels = np.zeros(len(percents), dtype=float)
    for ii in range(len(levels)):
        # Get the index of the pixel at which the CDF
        # reaches this percentage (the first one found)
        idx = (np.where(cdf < percents[ii]))[0]
        
        # Now get the level of that pixel
        levels[ii] = pixSort[idx[-1]]
    return np.array(levels)


def eccentric_anamoly(e, M):
    """
        Solve the Kepler equation

        Parameters
        ----------
        e: eccentricity
        M: mean anomaly

        Output
        ------
        the eccentric anomaly E, solution of  E-e sin(E)=M
        """
    NMAX=100

    E = M + e * math.sin(M) + e*e*math.sin(2.*M)/2.
    error = E - e * math.sin(E) - M
    i=0
    while(math.fabs(error) > 1.e-10):
        i=i+1
        E -= error / (1 - e * math.cos(E))
        error = E - e * math.sin(E) - M
        if i==NMAX:
            msg = 'eccentric_anomaly from efit5_results: Could not converge for e = %f and M =%f' % (e,M)
            raise EccAnomalyError(msg)
    return E



def compute_astro_contours(star,t,contour):
    # helper function
    (xch,ych) = star.get_astro(t)
    return [weighted_percentile(xch, contour, weights=star.weights),weighted_percentile(ych, contour, weights=star.weights)]

def compute_RV_contours(star,t,contour):
    # helper function
    RVch = star.get_RV(t)
    return weighted_percentile(RVch, contour, weights=star.weights)
    
def compute_dist2Sgra_contours(star,t,contour):
    #helper function
    return weighted_percentile(star.dist2Sgra_model(t), contour, weights=star.weights)
    
def compute_xy2Sgra_contours(star,t,contour):
    # helper function
    (xch,ych) = star.xy2Sgra_model(t)
    return [weighted_percentile(xch, contour, weights=star.weights),weighted_percentile(ych, contour, weights=star.weights)]
    
class star:
    def __init__(self,name,BH,astro=None,RV=None):
        '''
            Constructor for the class star

            Paramters:
            ----------
            name: string, name of the star
            BH: a BH object which contains information about global parameters
            astro: string, path to the .points file. The point file is loaded. The first column is replaced by the Julian year computed from the last column
            RV: string, path to the .rv file. The rv file is loaded. The first column is replaced by the Julian year computed from the last column. If no RV, None should be provided.
        '''
        self.name  = name
        self.BH=BH
        self.astroInput=astro
        self.RVInput=RV
        self.nastro=0
        self.nRV=0
        if astro is not None:
            if os.path.isfile(astro):
                self.astro = np.loadtxt(astro)
                self.nastro=len(self.astro[:,0])
                if len(self.astro[0,:])==6:
                    self.astro[:,0] = fromMJD2yr(self.astro[:,5])
        else:
            self.astro = None
        if RV is None:
            self.RV = None
        else:
            self.RV  = np.loadtxt(RV)
            self.nRV = len(self.RV[:,0])
            if len(self.RV[0,:])==4:
                self.RV[:,0]  = fromMJD2yr(self.RV[:,3])

        self.idx    = {x: False for x in STARS_PAR}
        self.TI_computed = False
        self.path_res = './' + name
        self.idxMAP = -1
        self.MAP = None
        self.chain   = {}
        self.mean = {}
        return

    def set_idx(self, par_name,idx):
        '''
            Construct the dictionnary which contains the link to the indices in the chain output file from efit5.

            par_name: a string which refers to the orbital parameters, i.e.: P, e, T0, O, w, i
            idx: an integer that corresponds to the column of that paramter for the corresponding star in the chain output file

            Output: this pair of key, value is updated in the dictionnary idx.
        '''
        self.idx[par_name] = idx
        return

    def set_path_res(self,p):
        ''' 
            set the directory where the residuals and plots related will be saved.
            '''
        self.path_res = p

    
    def set_chain_from_param(self,ch,p,w=None):
        self.chain[p] = ch
        if p=='P':
            self.chain['a_AU'] =  (self.BH.chain_glob['GM']*self.chain['P']*self.chain['P']*GMSUN_4pi2)**(1.0/3.0)   # a in AU = (Mass*Period^2)^(1/3)
        elif p=='e':
            self.chain['se2'] = np.sqrt(1.-self.chain['e']**2)
        if w is not None:
            self.weights = w
            self.mean[p] = weighted_avg(ch,w)
    

    def set_chain(self,ch,idxMAP=-1):
        '''
            Load the chains corresponding to the paramters related to the star.

            Paramters:
            ----------
            ch: 2D numpy array that corresponds to the weighted chains


            Output:
            -------
            The following paramter for this object are created
            chain: dictionnary. Keys: string, orbital parmeters. Values: chains related to this orbital parmeeter.
        '''

        self.weights = ch[:,0]
        
        for p in STARS_PAR:
            self.chain[p] = ch[:,self.idx[p]]
            self.mean[p]  = weighted_avg(self.chain[p],self.weights)

        self.chain_length = len(ch[:,0])
        self.chain['a_AU'] =  (self.BH.chain_glob['GM']*self.chain['P']*self.chain['P']*GMSUN_4pi2)**(1.0/3.0)   # a in AU = (Mass*Period^2)^(1/3)

        self.chain['se2'] = np.sqrt(1.-self.chain['e']**2)
        self.idxMAP=idxMAP
        if idxMAP !=-1:# I am creating another "star" object with only 1 element in the chain (that element being the MAP)
            self.MAP=star(self.name,self.BH.MAP,self.astroInput,self.RVInput)
            self.MAP.idx=self.idx
            chIDX=ch[idxMAP,:]
            self.MAP.set_chain(np.reshape(chIDX,(1,len(chIDX))))


    def set_TI(self):
        '''
            Compute the Thiennes-Illes constant for the whole chain. Need to be called only before an "enveloppe" calculation

            Output:
            -------
            The chain dictionnary is updated with the following keys:  TI_A, TI_B, TI_C, TI_F, TI_G, TI_H
        '''
        self.chain['TI_A'] = self.chain['a_AU']*( np.cos(self.chain['w'])*np.cos(self.chain['O']) - np.sin(self.chain['w'])*np.sin(self.chain['O'])*np.cos(self.chain['i']))
        self.chain['TI_B'] = self.chain['a_AU']*( np.cos(self.chain['w'])*np.sin(self.chain['O']) + np.sin(self.chain['w'])*np.cos(self.chain['O'])*np.cos(self.chain['i']))
        self.chain['TI_C'] = self.chain['a_AU']*( np.sin(self.chain['w'])*np.sin(self.chain['i']))
        self.chain['TI_F'] = self.chain['a_AU']*(-np.sin(self.chain['w'])*np.cos(self.chain['O']) - np.cos(self.chain['w'])*np.sin(self.chain['O'])*np.cos(self.chain['i']))
        self.chain['TI_G'] = self.chain['a_AU']*(-np.sin(self.chain['w'])*np.sin(self.chain['O']) + np.cos(self.chain['w'])*np.cos(self.chain['O'])*np.cos(self.chain['i']))
        self.chain['TI_H'] = self.chain['a_AU']*( np.cos(self.chain['w'])*np.sin(self.chain['i']))
        self.TI_computed = True


    def compute_eccentric_anomaly(self,t):
        '''Compute the eccentric for the whole chain (including Romer time delay if model requires it) 
        
        Parameter:
        ----------
        t: numpy array whose length is either 1 or the length of the chain or a scalar
        
        Output:
        -------
        E: in rad
        '''
        if not self.TI_computed:
            self.set_TI()
        M = 2.*math.pi*(t-self.chain['T0'])/self.chain['P']
        e = self.chain['e']
        if np.size(e)==1:
            e=e[0]*np.ones(M.shape)
        E = np.zeros(len(M))
        for i in range(len(M)):
            E[i] = eccentric_anamoly(e[i], M[i])

        if self.BH.model != 'Kepler' : #Romer time delay
            z = self.chain['TI_C'] * (np.cos(E) - self.chain['e']) + self.chain['TI_H'] * np.sin(E)*self.chain['se2'] # in AU
            M -= 2.*math.pi*z/CAUYR /self.chain['P']
            for i in range(len(M)):
                E[i] = eccentric_anamoly(e[i], M[i])

        return E
            
    def get_astro(self,t):
        E = self.compute_eccentric_anomaly(t)
        X  = np.cos(E) - self.chain['e']
        Y  = np.sin(E)*self.chain['se2']
        x = self.chain['TI_B'] * X + self.chain['TI_G'] * Y
        y = self.chain['TI_A'] * X + self.chain['TI_F'] * Y
        x = -x/self.BH.chain_glob['R0'] + self.BH.chain_glob['x0'] + self.BH.chain_glob['vx0'] *(t-TORIGIN) #WORKING WITH CAMERA COORDINATE
        y =  y/self.BH.chain_glob['R0'] + self.BH.chain_glob['y0'] + self.BH.chain_glob['vy0'] *(t-TORIGIN)
        return (x,y)
    
    def get_astro_opt(self,t):
        return self.MAP.get_astro(t)
        
    
    def xy2Sgra_model(self,t):
        E = self.compute_eccentric_anomaly(t)
        X  = np.cos(E) - self.chain['e']
        Y  = np.sin(E)*self.chain['se2']
        x = (self.chain['TI_B'] * X + self.chain['TI_G'] * Y)/self.BH.chain_glob['R0'] 
        y = (self.chain['TI_A'] * X + self.chain['TI_F'] * Y)/self.BH.chain_glob['R0'] 
        return (x,y)# in arcsecond
        
    def dist2Sgra_model(self,t):
        (x,y) = self.xy2Sgra_model(t)
        return np.sqrt(x**2+y**2)#in as
        
    def astro_model(self,t,CL=[0.68],processes=4):
        # process the chains to get residuals for astrometry
        nt = np.size(t)
        n_CL = 1
        contour_list = [0.5 ]
        if CL:
            n_CL = 2*len(CL) +1
            for cl in CL:
                contour_list.append(0.5-cl*0.5)
                contour_list.append(0.5+cl*0.5)
                
        contour = np.array(contour_list)
        x = np.zeros((nt,n_CL))
        y = np.zeros((nt,n_CL))

        pool = mp.Pool(processes=processes)
        results = [pool.apply_async(compute_astro_contours, args=(self,tt,contour)) for tt in t]

        output = [p.get() for p in results]
        pool.close()
        for i in range(nt):
            pp = output[i]
            x[i,:] = pp[0]
            y[i,:] = pp[1]
        #for i in range(nt):
        #    x[i,:],y[i,:]=compute_astro_contours(self,t[i],contour)
        return np.array(x),np.array(y)
        
    def astro_residuals(self,dt=.05,CL=[0.68],processes=4):
        # Parallelize the determination of the astrometry residuals
        # 2018-03-26 - edited (T. Do)

        # the model for the epochs corresponding to observation
        x_model,y_model = self.astro_model(self.astro[:,0],CL=None,processes=processes)


        x_res = self.astro[:,1] - x_model[:,0]
        y_res = self.astro[:,2] - y_model[:,0]
        res = np.copy(self.astro)
        res[:,1] = x_res
        res[:,2] = y_res

        np.savetxt(self.path_res+'_res.points',res)

        #compute the envelope
        t_min = np.min(self.astro[:,0])-0.5
        t_max = np.max(self.astro[:,0])+2
        time = np.linspace(t_min,t_max,int((t_max-t_min)/dt+1),endpoint=True)
        x_env,y_env = self.astro_model(time,CL=CL,processes=processes)
        tosave = np.column_stack(( time,x_env,y_env))
        np.savetxt(self.path_res+'_mod.points',tosave) 
        
    def get_dist2Sgra(self,t,CL=[0.68],processes=4):
        nt = np.size(t)
        n_CL = 1
        contour_list = [0.5 ]
        if CL:
            n_CL = 2*len(CL) +1
            for cl in CL:
                contour_list.append(0.5-cl*0.5)
                contour_list.append(0.5+cl*0.5)
                
        contour = np.array(contour_list)

        pool = mp.Pool(processes=processes)
        results = [pool.apply_async(compute_dist2Sgra_contours, args=(self,tt,contour)) for tt in t]

        output = [p.get() for p in results]
        pool.close()
        return np.array(output)     
    
    def get_xy2Sgra(self,t,CL=[0.68],processes=4):
        # process the chains to get residuals for astrometry
        nt = np.size(t)
        n_CL = 1
        contour_list = [0.5 ]
        if CL:
            n_CL = 2*len(CL) +1
            for cl in CL:
                contour_list.append(0.5-cl*0.5)
                contour_list.append(0.5+cl*0.5)
                
        contour = np.array(contour_list)
        x = np.zeros((nt,n_CL))
        y = np.zeros((nt,n_CL))

        pool = mp.Pool(processes=processes)
        results = [pool.apply_async(compute_xy2Sgra_contours, args=(self,tt,contour)) for tt in t]

        output = [p.get() for p in results]
        pool.close()
        for i in range(nt):
            pp = output[i]
            x[i,:] = pp[0]
            y[i,:] = pp[1]
        #for i in range(nt):
        #    x[i,:],y[i,:]=compute_astro_contours(self,t[i],contour)
        return np.array(x),np.array(y)
    
    def get_pred_astro(self,t,col,CL=0.68,nbins=50):
        # t needs to be a scalar here
        (xch,ych)=self.get_astro(t)
        
        (hist, obins, ibins) = np.histogram2d(ych,xch, bins = nbins, weights = self.weights)
        levels = getContourLevels(hist,percents=[CL])
        c = plt.gca().contourf(hist, [levels,1e5],origin=None,extent = [ibins[0], ibins[-1], obins[0], obins[-1]],linestyles='-',colors=[col,col])
        p = c.collections[0].get_paths()[0]
        v = p.vertices
        return v
            
    def astro_chi2(self):
        if self.astro is not None:
            x_opt,y_opt = self.MAP.get_astro(self.astro[:,0])
            chi2 = np.sum((x_opt-self.astro[:,1])**2/self.astro[:,3]**2 + (y_opt-self.astro[:,2])**2/self.astro[:,4]**2)
        else:
            chi2=0.
        return chi2
        
    def get_RV(self,t):
        '''Compute the RV of the star (models implemented: kepler and kepler + redshift)

            Parameter:
            ----------
            t: numpy array whose length is either 1 or the length of the chain or a scalar
            '''
        E = self.compute_eccentric_anomaly(t)
        cE = np.cos(E)
        VX = -2.*math.pi*np.sin(E)/(1.-self.chain['e']*cE)/self.chain['P']
        VY = 2.*math.pi*cE*self.chain['se2']/(1.-self.chain['e']*cE)/self.chain['P']
        RV = self.chain['TI_C'] * VX + self.chain['TI_H'] * VY # in AU/yr
    

        if self.BH.model != 'Kepler': # redshift
            r  = self.chain['a_AU']*(1.-self.chain['e']*cE) #AU
            v2 = self.BH.chain_glob['GM']*GMSUN*(1.+self.chain['e']*cE)/r #AU^2/yr^2
            SR = 0.5*v2 #AU^2/yr^2
            GR = GMSUN*self.BH.chain_glob['GM']/r #AU^2/yr^2
            RV += self.BH.chain_glob['Redshift']*(SR+GR)/CAUYR # in AU/yr

        RV = RV * AUYR2KMS + self.BH.chain_glob['vz0'] #in km/s
        return RV
        
    def get_RV_opt(self,t):
        return self.MAP.get_RV(t)

    def RV_model(self,t,CL=[0.68],processes=4):
        # Parallelize the determination of the RV residuals
        # 2018-03-26 - edited (T. Do)
        nt = np.size(t)
        n_CL = 1
        contour_list = [0.5 ]
        if CL:
            n_CL = 2*len(CL) +1
            for cl in CL:
                contour_list.append(0.5-cl*0.5)
                contour_list.append(0.5+cl*0.5)

        contour = np.array(contour_list)
        RV = np.zeros((nt,n_CL))

        pool = mp.Pool(processes=processes)
        results = [pool.apply_async(compute_RV_contours, args=(self,tt,contour)) for tt in t]

        output = [p.get() for p in results]
        pool.close()
        return np.array(output)

    def RV_residuals(self,dt=.05,CL=[0.68],processes=4):
        if self.RV is not None:
            # the model for the epochs corresponding to observation
            RV_model = self.RV_model(self.RV[:,0],CL=None,processes=processes)

            res = np.copy(self.RV)
            res[:,1] = self.RV[:,1] - RV_model[:,0]

            np.savetxt(self.path_res+'_res.rv',res)
            t_min = 2000.#np.min(star.RV[:,0])-0.5
            t_max = 2020.#np.max(star.RV[:,0])+1
            #np.savetxt(star.path_res+'_model',np.column_stack(( star.RV[:,0],RV_model,.5*(RV_model[:,2]-RV_model[:,1]))))
        else:
            t_min = 1994.
            t_max = 2018.

        #compute the envelope
        time = np.linspace(t_min,t_max,int((t_max-t_min)/dt+1),endpoint=True)
        RV_env = self.RV_model(time,CL=CL,processes=processes)
        tosave = np.column_stack(( time,RV_env))
        np.savetxt(self.path_res+'_mod.rv',tosave)

    def RV_chi2(self):
        if self.RV is not None:
            RV_opt = self.MAP.get_RV(self.RV[:,0])
            chi2 = np.sum((RV_opt-self.RV[:,1])**2/self.RV[:,2]**2 )
        else:
            chi2=0.
        return chi2
            
    def orbit(self,t):
        '''
            Computed the 3D position and velocity of the star for the whole chain

            Parameter:
            ----------
            t: numpy array whose length is either 1 or the length of the chain.

            Output:
            -------
            x,y,z and vx vy vz: 2D numpy array of length of the chain with the position/velcoity of the star
                computed for the c=whole chain. The Romer time delay is included if the model requires it. The redshift is 
                never included here since it is an effect on the observable and not a physical effect on S0-2 velocity in itself
                The BH is assumed to be at the center.
                Units: AU and AU/yr
        '''
        E = self.compute_eccentric_anomaly(t)
        X  = np.cos(E) - self.chain['e']
        Y  = np.sin(E)*self.chain['se2']
        VX = -2.*math.pi*np.sin(E)/(1.-self.chain['e']*np.cos(E))/self.chain['P']
        VY = 2.*math.pi*np.cos(E)*self.chain['se2']/(1.-self.chain['e']*np.cos(E))/self.chain['P']

        x = self.chain['TI_B'] * X + self.chain['TI_G'] * Y
        y = self.chain['TI_A'] * X + self.chain['TI_F'] * Y
        z = self.chain['TI_C'] * X + self.chain['TI_H'] * Y
        vx = self.chain['TI_B'] * VX + self.chain['TI_G'] * VY
        vy = self.chain['TI_A'] * VX + self.chain['TI_F'] * VY
        vz = self.chain['TI_C'] * VX + self.chain['TI_H'] * VY
        return (x,y,z),(vx,vy,vz)
        
        
    def orbit_opt(self,t):
        return self.MAP.orbit(t)
        
    def orbit_Kepler(self,t):
        '''
            Computed the 3D position and velocity of the star for the whole chain

            Parameter:
            ----------
            t: numpy array whose length is either 1 or the length of the chain.

            Output:
            -------
            x,y,z and vx vy vz: 1D numpy array of length of the chain with the position/velcoity of the star
                computed for the c=whole chain. Units: AU and AU/yr
        '''
        if not self.TI_computed:
            self.set_TI()
        M = 2.*math.pi*(t-self.chain['T0'])/self.chain['P']
        E = np.zeros(len(M))
        for i in range(len(M)):
            E[i] = eccentric_anamoly(self.chain['e'][i], M[i])
        X  = np.cos(E) - self.chain['e']
        Y  = np.sin(E)*self.chain['se2']
        VX = -2.*math.pi*np.sin(E)/(1.-self.chain['e']*np.cos(E))/self.chain['P']
        VY = 2.*math.pi*np.cos(E)*self.chain['se2']/(1.-self.chain['e']*np.cos(E))/self.chain['P']

        x = self.chain['TI_B'] * X + self.chain['TI_G'] * Y
        y = self.chain['TI_A'] * X + self.chain['TI_F'] * Y
        z = self.chain['TI_C'] * X + self.chain['TI_H'] * Y
        vx = self.chain['TI_B'] * VX + self.chain['TI_G'] * VY
        vy = self.chain['TI_A'] * VX + self.chain['TI_F'] * VY
        vz = self.chain['TI_C'] * VX + self.chain['TI_H'] * VY
        return (x,y,z),(vx,vy,vz)



    def get_Jacobian_astro(self,t):
        '''
            Compute the sum of the absolute value of the Jacobian of the transformation between X,Y (position in the orbit) with
            respect to P and e (used for observables pripros)
            
            '''
        M = 2.*math.pi*(t-self.chain['T0'])/self.chain['P']
        E = np.zeros(len(M))
        for i in range(len(M)):
            
            E[i] = eccentric_anamoly(self.chain['e'][i], M[i])
        return np.fabs( 3.*M*(self.chain['e']+np.cos(E)) + 2.*(-2.+self.chain['e']**2+self.chain['e']*np.cos(E))*np.sin(E) )


    def get_Jacobian_RV(self,t):
        '''
            Compute the sum of the absolute value of the Jacobian of the transformation between VX,VY, (velocity in the orbit) with
            respect to P and e (used for observables pripros)
            
            '''
        M = 2.*math.pi*(t-self.chain['T0'])/self.chain['P']
        E = np.zeros(len(M))
        for i in range(len(M)):
            E[i] = eccentric_anamoly(self.chain['e'][i], M[i])
        return np.fabs(  (  np.sin(E)*(self.chain['e']**2*np.cos(2.*E) + 3.*self.chain['e']**2-2. ) + np.cos(E)*(6.*M - 2.*self.chain['e']*np.sin(E)) ) / ( self.chain['se2']*((-1.+self.chain['e']*np.cos(E))**3) )  )

    def get_observable_prior(self):
        '''
            Compute -2 log (Prior) for an observable prior based only on astro
            
            '''
        chi2=np.zeros(self.chain_length)
        if self.astro is not None:
            for t in self.astro[:,0]:
                chi2=chi2+self.get_Jacobian_astro(t)
        if self.RV is not None:
            for t in self.RV[:,0]:
                chi2=chi2+self.get_Jacobian_astro(t)
        chi2=chi2*(self.chain['P']**(1./3.))/self.chain['se2']
        return -2.*np.log(chi2)

    def get_observableXV_prior(self,R0):
        '''
            Compute -2 log (Prior) for an observable prior based on astro and RV
            Input:
            ------
            R0: in parsec.
            
            '''
        chi2=np.zeros(self.chain_length)
        if self.astro is not None:
            for i,t in enumerate(self.astro[:,0]):
                #print t,self.astro[i,:]
                #sigRA = self.astro[i,3]
                #sigDEC = self.astro[i,4]

                chi2=chi2+self.get_Jacobian_astro(t)/(self.astro[i,3]*self.astro[i,4]*R0**2)
            chi2=chi2*(math.pi**(-2./3.))*(self.chain['P']**(1./3.))/(2.*self.chain['se2'])

        if self.RV is not None:
            for i,t in enumerate(self.RV[:,0]):
                #print t,self.RV[i,:]
                #sigRV = self.RV[i,2]
                chi2=chi2+self.get_Jacobian_RV(t)*AUYR2KMS*AUYR2KMS/(self.RV[i,2]**2)*(self.chain['P']**(-5./3.))*(math.pi**(2./3.))
        return -2.*np.log(chi2)

    def compute_t_maxRV(self):
        '''
            For the chain, this routine computes the time of min and max RV (assuming a Keplerian orbit)
            
            '''

        E = 2*np.arctan(np.sqrt((1-self.chain['e'])/(1+self.chain['e']))*np.tan(-0.5*self.chain['w']))
        self.t_maxRV = (E-self.chain['e']*np.sin(E))*self.chain['P']/2./math.pi + self.chain['T0']
        E = 2*np.arctan(np.sqrt((1-self.chain['e'])/(1+self.chain['e']))*np.tan(0.5*(math.pi-self.chain['w'])))
        self.t_minRV = (E-self.chain['e']*np.sin(E))*self.chain['P']/2./math.pi + self.chain['T0']


    def plot_astro_residuals(self,outname=None):
        res = np.loadtxt(self.path_res+'_res.points')
        mod = np.loadtxt(self.path_res+'_mod.points')


        f, (ax1,ax2) = plt.subplots(2, sharex=True)
        ax1.errorbar(res[:,0], res[:,1]*1000.,yerr=res[:,3]*1000.,fmt='bo')
        ax1.plot(mod[:,0],(mod[:,2]-mod[:,1])*1000.,'g--')
        ax1.plot(mod[:,0],(mod[:,3]-mod[:,1])*1000.,'g--')
        ax1.fill_between(mod[:,0],(mod[:,3]-mod[:,1])*1000.,(mod[:,2]-mod[:,1])*1000.,facecolor='green',alpha=0.5)


        ax1.set_ylabel('\Delta x [mas]')
        ax1.set_xlim([np.min(mod[:,0]),np.max(mod[:,0])])

        ax2.errorbar(res[:,0], res[:,2]*1000.,yerr=res[:,4]*1000.,fmt='bo')
        ax2.plot(mod[:,0],(mod[:,5]-mod[:,4])*1000.,'g--')
        ax2.plot(mod[:,0],(mod[:,6]-mod[:,4])*1000.,'g--')
        ax2.fill_between(mod[:,0],(mod[:,6]-mod[:,4])*1000.,(mod[:,5]-mod[:,4])*1000.,facecolor='green',alpha=0.5)
        ax2.set_ylabel('\Delta y [mas]')
        ax2.set_xlabel('Time [yr]')
        ax2.set_xlim([np.min(mod[:,0]),np.max(mod[:,0])])

        chi2 = np.sum((res[:,1]/res[:,3])**2 +(res[:,2]/res[:,4])**2)
        n = len(res[:,1])
        #ax1.set_title('Astro for '+self.name+' '+ 'n='+str(n)+' chi2=%.2f  chi2/2n=%.2f' %(chi2,chi2*0.5/(n*1.) ))
        ax1.set_title('Astro for '+self.name+' '+ 'n='+str(n)+' chi2=%.2f  ' %(chi2 ))
        if outname==None:
            plt.savefig(self.path_res+'_astro_res.png')
        else:
            plt.savefig(outname+'.png')
        
        f, (ax1,ax2) = plt.subplots(2, sharex=True)
        ax1.plot(mod[:,0],mod[:,1]*1000.,'g')
        ax1.errorbar(self.astro[:,0], self.astro[:,1]*1000.,yerr=self.astro[:,3]*1000.,fmt='bo')
        #ax1.plot(mod[:,0],(mod[:,2])*1000.,'g--')
        #ax1.plot(mod[:,0],(mod[:,3])*1000.,'g--')
        #ax1.fill_between(mod[:,0],(mod[:,3])*1000.,(mod[:,2])*1000.,facecolor='green',alpha=0.5)
        ax1.set_title('Astro for '+self.name)
        ax1.set_ylabel('x [mas]')

        ax2.plot(mod[:,0],mod[:,4]*1000.,'g')
        ax2.errorbar(self.astro[:,0], self.astro[:,2]*1000.,yerr=self.astro[:,4]*1000.,fmt='bo')
        ax2.set_ylabel('y [mas]')
        ax2.set_xlabel('Time [yr]')
        ax2.set_xlim([np.min(mod[:,0]),np.max(mod[:,0])])

        if outname==None:
            plt.savefig(self.path_res+'_astro_mod.png')
        else:
            plt.savefig(outname+'_mod.png')
        # figure with the 2D orbits and residuals

        fig =plt.figure(figsize=(8,8))
            
        plt.errorbar(self.astro[:,1]*1000.,self.astro[:,2]*1000.,xerr=self.astro[:,3]*1000.,yerr=self.astro[:,4]*1000.,ls='None',lw=2,marker='o')
        plt.plot(mod[:,1]*1000.,mod[:,4]*1000.,lw=1,ls='-',color='g')
        plt.axis('equal')

        plt.xlabel('x [mas]')
        plt.ylabel('y [mas]')
        plt.savefig(self.path_res+'_orbit.png')
    

    def plot_RV_residuals(self,outname=None,ylim=None):
        fig = plt.figure()
        if self.RV  is not None:
            res = np.loadtxt(self.path_res+'_res.rv')
            plt.errorbar(res[:,0], res[:,1],yerr=res[:,2],fmt='bo')
            chi2 = np.sum((res[:,1]/res[:,2])**2 )
            n = len(res[:,1])

        mod = np.loadtxt(self.path_res+'_mod.rv')
        
        plt.plot(mod[:,0],(mod[:,2]-mod[:,1]),'g--')
        plt.plot(mod[:,0],(mod[:,3]-mod[:,1]),'g--')
        plt.fill_between(mod[:,0],(mod[:,3]-mod[:,1]),(mod[:,2]-mod[:,1]),facecolor='green',alpha=0.5)
        
        plt.title('RV for '+self.name)
        plt.ylabel('\Delta RV [km/s]')
        plt.xlabel('Time [yr]')
        plt.xlim([np.min(mod[:,0]),np.max(mod[:,0])])
        
        if ylim is not None:
            plt.ylim(ylim)
        
        if self.RV is not None:
            plt.title('RV for '+self.name+' '+ 'nRV='+str(n)+' chi2=%.2f ' %(chi2))
            #plt.title('RV for '+star.name+' '+ 'n='+str(n)+' chi2=%.2f  chi2/n=%.2f' %(chi2,chi2/(n*1.)))
        else:
            plt.title('RV for '+self.name)
        
        if outname==None:
            fig.set_size_inches(10.5, 7.5)
            plt.savefig(self.path_res+'_RV_res.png')
        else:
            plt.savefig(outname+'.png')

        f = plt.figure()
        if self.RV  is not None:
            plt.errorbar(self.RV[:,0], self.RV[:,1],yerr=self.RV[:,2],fmt='bo')

        plt.plot(mod[:,0],mod[:,1],'g')
        plt.title('RV for '+self.name)
        plt.xlabel('Time [yr]')
        plt.ylabel('RV [km/s]')
        plt.xlim([np.min(mod[:,0]),np.max(mod[:,0])])
        
        if outname==None:
            plt.savefig(self.path_res+'_RV_mod.png')
        else:
            plt.savefig(outname+'_mod.png')

            
class BH:
    '''
        Class that will deal with the global parameters
        '''

    def __init__(self,mod):
        self.model = mod
        self.chain_glob={'GM' : None , 'R0' : None, 'x0': 0. , 'y0':0., 'vx0':0., 'vy0':0., 'vz0':0. , 'Redshift':0. , 'GR':None  , 'EM':None   }
        self.MAP = None
        self.weights = None
        self.mean={}
        self.idxMAP=-1

    def set_chains(self,ch,par,idxMAP=-1,weights=None):
        '''
            Set the chain for one parameter
            '''
        self.chain_glob[par] = ch
        if idxMAP !=-1 and self.MAP==None:
            self.idxMAP = idxMAP
            self.MAP=BH(self.model)
        if self.idxMAP!=-1:
                self.MAP.set_chains(np.reshape(ch[self.idxMAP],(1)),par)

        if weights is not None:
                self.weights  = weights
        if self.weights is not None:
            self.mean[par] = weighted_avg(ch,self.weights)
    
    

    def set_all_param(self,GM,R0,x0=0.,y0=0.,vx0=0.,vy0=0.,vz0=0.,redshift=0.,weights=None):
        ''' 
            Set the chain for all parameters
            '''
        
        self.chain_glob['GM'] = GM
        self.chain_glob['R0'] = R0
        self.chain_glob['x0'] = x0
        self.chain_glob['y0'] = y0
        self.chain_glob['vx0'] = vx0
        self.chain_glob['vy0'] = vy0
        self.chain_glob['vz0'] = vz0
        self.chain_glob['Redshift'] = redshift
        if weights is not None:
            self.weights  = weights
        if self.weights is not None:
            for p in ['GM','R0','x0','y0','vx0','vy0','vz0','Redshift']:
                self.mean[p] = weighted_avg(self.chain_glob[p],self.weights)




class efit5:
    def __init__(self,input,withInt=True):
        '''
            Constructor for the class efit5.

            Input: string containing the path to the input file for en efit5 run
            withInt: if true, it writes a file with the confidence intervals

            Output: the input file is read and the efit5 output chain is loaded
            The following parameters are initialized for this object:

            input_file: the path to the input file
            model: string, name of the model. So far either Kepler and Kepler_Redshift is allowed
            chain_dir: string which refers to the directory where the output files from efit are stored
            n_par_fitted: number of parameters fitted in the efit5 run
            star_names: list of strings containing the names of the stars considered in the efit5 run
            idx_glob_par: dictionnary. Keys are the name of the global parameters related to the model
                          (i.e. for Kepler: GM, R0, x0, y0, vx0, vy0, vz0 and R0 - for Kepler_Redshift: Redshift, GR and EM)
                          The values of the dictionnary are integer that correspond to the index of that parameter in the
                          output chain file: efit_.txt.
            weights: numpy 1-D array with the weight of the chain
            idx_MAP: index in the chain that corresponds to the MAP
            chain_glob_par: dictionnary. Keys are the name of the global paramters, the values are 1D numpy array with the chain
                            corresponding to the key parameter
            glob_par: dictionnary. Keys are the name of the global paramters, values are list containing the mean and std
                           of this parameter and the values of the MAP

            stars: list containing object from the class star. These objects are initialized and contain the related chains.
        '''
        self.input_file = input
        #if not os.path.isfile(input):

        file_in = open(input,'r')
        lines_in = file_in.readlines()
        file_in.close()

        dir_path = os.path.dirname(os.path.realpath(input))
        model = 'Kepler'
        self.chain_dir = 'chains'
        self.nstars = 0
        self.n_par_fitted = 0

        dict_glob_par = {'GM' : False , 'R0' : False, 'x0':False , 'y0':False, 'vx0':False, 'vy0':False, 'vz0':False , 'Redshift':False , 'GR':False  , 'EM':False   }
        dict_stars_par = []
        astro_dir ='.'
        star_names = []
        self.priors = []
        self.prior_par = {}
        for l in lines_in:

            if '[' in l:
                self.n_par_fitted +=1

            if 'useKeplerRedshift' in l:
                model='Kepler_Redshift'
            #priors
            elif 'useXVObservablesPrior' in l:
                self.priors.append('ObservableXV')
            elif 'useObservablesPrior' in l:
                self.priors.append('Observable')
            elif 'useCosIPrior' in l:
                self.priors.append('cosi')
            elif 'useVzPrior' in l:
                self.priors.append('vz')
            elif 'useMPrior' in l:
                self.priors.append('M')
            elif 'useR0Prior' in l:
                self.priors.append('R0')

            elif 'meanVzPrior' in l:
                self.prior_par['mvz']=float(l.split('=')[1].split(';')[0])
            elif 'sigVzPrior' in l:
                self.prior_par['svz']=float(l.split('=')[1].split(';')[0])
            elif 'meanMPrior' in l:
                self.prior_par['mM']=float(l.split('=')[1].split(';')[0])
            elif 'sigMPrior' in l:
                self.prior_par['sM']=float(l.split('=')[1].split(';')[0])
            elif 'meanR0Prior' in l:
                self.prior_par['mR0']=float(l.split('=')[1].split(';')[0])
            elif 'sigR0Prior' in l:
                self.prior_par['sR0']=float(l.split('=')[1].split(';')[0])

            elif 'chainDir' in l:
                self.chain_dir = l.split('"')[1]
            elif 'dataSource' in l:
                astro_dir = l.split('"')[1]
            elif 'object' in l:
                self.nstars +=1
                star_names.append(l.split('"')[1])
                dict_stars_par.append({'P':False, 'e':False, 'T0':False, 'w':False, 'O':False, 'i':False,'name':star_names[-1],'RV':None})

            elif 'mass' in l:
                dict_glob_par['GM'] = '[' in l
            elif 'distance' in l:
                dict_glob_par['R0'] = '[' in l
            elif 'focusX_velocity' in l:
                dict_glob_par['vx0'] = '[' in l
            elif 'focusY_velocity' in l:
                dict_glob_par['vy0'] = '[' in l
            elif 'focusX' in l:
                dict_glob_par['x0'] = '[' in l
            elif 'focusY' in l:
                dict_glob_par['y0'] = '[' in l
            elif 'focus_radial' in l:
                dict_glob_par['vz0'] = '[' in l
            elif 'TestingEquivalence' in l:
                dict_glob_par['Redshift'] = '[' in l
            elif 'extendedMass' in l:
                dict_glob_par['EM'] = '[' in l
            elif 'TestingEinstein' in l:
                dict_glob_par['GR'] = '[' in l
            elif 'period' in l:
                dict_stars_par[-1]['P']= '[' in l
            elif 'periastronPassage' in l:
                dict_stars_par[-1]['T0']= '[' in l
            elif 'eccentricity' in l:
                dict_stars_par[-1]['e']= '[' in l
            elif 'bigOmega' in l:
                dict_stars_par[-1]['O']= '[' in l
            elif 'smallOmega' in l:
                dict_stars_par[-1]['w']= '[' in l
            elif 'inclination' in l:
                dict_stars_par[-1]['i']= '[' in l
            elif 'radialVelocityData' in l:
                if l.split('"')[1][0]=='/':
                    dict_stars_par[-1]['RV'] = l.split('"')[1]
                else:
                    dict_stars_par[-1]['RV'] = dir_path+'/'+l.split('"')[1]
        self.BH = BH(model)
        
        if self.chain_dir[0]!='/':
            self.chain_dir = dir_path+'/'+self.chain_dir

        if len(astro_dir)>0:
            if astro_dir[0] != '/':
                astro_dir =dir_path+'/'+astro_dir
        else:
            astro_dir = '.'
        # constructing dictionary that contains the index for which each parameters is save in "efit_.txt" chains file
        idx_fitted = 2
        idx_not_fitted = 2+self.n_par_fitted
        idx_glob_par = {}
        for p in  GLOB_PAR :
            if dict_glob_par[p]:
                idx_glob_par[p] = idx_fitted
                idx_fitted +=1
            else:
                idx_glob_par[p] = idx_not_fitted
                idx_not_fitted +=1
        self.nobs=0
        # loop on stars to put the indices as well
        self.stars=[]
        for s in dict_stars_par:
            self.stars.append(star(s['name'],self.BH,astro_dir+'/'+s['name']+'.points',s['RV']))
            self.nobs+= self.stars[-1].nastro*2 +self.stars[-1].nRV
            self.stars[-1].set_path_res(self.chain_dir+'/'+s['name'])
            for p in STARS_PAR:
                if s[p]:
                    self.stars[-1].set_idx(p,idx_fitted)
                    idx_fitted +=1
                else:
                    self.stars[-1].set_idx(p,idx_not_fitted)
                    idx_not_fitted +=1


        # loading the chains file
        chain = pd.read_csv(self.chain_dir+'/efit_.txt',delim_whitespace=True,header=None)
        self.weights       = chain.iloc[:,0]
        self.posterior     = chain.iloc[:,1]
        self.idx_MAP       = np.where(chain.iloc[:,1]==np.min(chain.iloc[:,1]))[0][0]
        self.chain_length  = len(chain.iloc[:,0])
        for p in GLOB_PAR:
            self.BH.set_chains(chain.iloc[:,idx_glob_par[p]],p,self.idx_MAP,self.weights)

        for s in self.stars:
            s.set_chain(chain.to_numpy(),self.idx_MAP)
        # creating the confidence intervals
        n= len(chain.iloc[0,:])-2
        cint = np.zeros((n,6))
        cint[:,0] = chain.iloc[self.idx_MAP,2:]
        for i in range(n):
            cint[i,1:4] = weighted_percentile(chain.iloc[:,i+2],[0.5,0.5-0.68/2.,0.5+0.68/2.],self.weights)
            cint[i,4:6] = cint[i,2:4]-cint[i,1]
        if withInt:
            np.savetxt(self.chain_dir+'/efit_int.dat',cint)

        self.ci=cint[:,1:6]
        
        
        # loading stat file to get the global evidence
        fstat = open(self.chain_dir+'/efit_stats.dat')
        line = fstat.readline()
        fstat.close()
        parts = line.split()

        # this the efit_stats.dat file is different between MultiNest v 3.10 and 2.18
        if parts[0] == 'Nested':
            self.log_evidence = float(parts[5])
        else:
            self.log_evidence = float(parts[2])

    def compute_prior(self):
        chi2 = np.zeros(self.chain_length)
        if 'Observable' in self.priors:
            print('Observable prior')
            for s in self.stars: #priors for each star
                chi2 = chi2 + s.get_observable_prior() - 2.*np.log(np.fabs(np.sin(s.chain['i'])))

            # now adding prior related to global parameters (GM, R0)
            chi2=chi2  + 2.*np.log(self.BH.chain_glob['GM']) + 2.*np.log(self.BH.chain_glob['R0'])
        elif 'ObservableXV' in self.priors:
            print('X-V Observable prior')
            for s in self.stars: #priors for each star
                chi2 = chi2 + s.get_observableXV_prior(self.BH.chain_glob['R0'])- 2.*np.log(np.fabs(np.sin(s.chain['i'])))
            # now adding prior related to global parameters (GM, i)
            chi2=chi2  + 2.*np.log(self.BH.chain_glob['GM'])
        elif 'cosi' in self.priors:
            print('Prior flat in cos (i)')
            for s in self.stars: #priors for each star
                chi2 = chi2 - 2.*np.log(np.fabs(np.sin(s.chain['i'])))

        if 'vz' in self.priors :
            chi2 = chi2 + (self.BH.chain_glob['vz0']-self.prior_par[mvz])**2/self.prior_par[svz]**2
        if 'M' in self.priors:
            chi2 = chi2 + (self.BH.chain_glob['GM']-self.prior_par[mM])**2/self.prior_par[sM]**2
        if 'R0' in self.priors:
            chi2 = chi2 + (self.BH.chain_glob['R0']-self.prior_par[mR0])**2/self.prior_par[sR0]**2

        self.prior_values = chi2
        self.likelihood = self.posterior-self.prior_values


    def compute_relative_entropy(self,input_priors_only=None):
        log_evidence_prior =0.
        if not hasattr(self, 'likelihood'):
            self.compute_prior()
        if input_priors_only is not None:

            file_in = open(input_priors_only,'r')

            dir_path = os.path.dirname(os.path.realpath(input_priors_only))
            isPrior = False
            ch_dir = 'chains'
            for l in file_in:
                if 'priorsOnly' in l:
                    isPrior=True
                elif  'chainDir' in l:
                    ch_dir = l.split('"')[1]
            file_in.close()
            if not isPrior:
                print("The input file ",input_priors_only," is not a priorOnly run")
                return

            if ch_dir[0]!='/':
                ch_dir = dir_path+'/'+ ch_dir

            fstat = open(ch_dir+'/efit_stats.dat')
            l = fstat.readline()
            fstat.close()
            log_evidence_prior = float(l.split()[2])
        self.relative_entropy = -np.sum(self.weights*self.likelihood)*.5 - self.log_evidence + log_evidence_prior
        return self.relative_entropy

    def get_chi2(self):
        chi2=0
        for s in self.stars:
            chi2+= s.astro_chi2() + s.RV_chi2()
        self.chi2 = chi2 
        return chi2

    def reduced_chi2(self):
        chi2 = self.get_chi2()
        return chi2/(1.*self.nobs-self.n_par_fitted)


    def plot_all_res(self,outname=None):
        plt.close("all")
        for s in self.stars:
            s.plot_astro_residuals(outname)
            s.plot_RV_residuals(outname)

    def all_residuals(self,processes=4):
        plt.close("all")
        for s in self.stars:
            s.astro_residuals(processes=processes)
            s.RV_residuals(processes=processes)
            
    def plot_2D_contour(self,p1,p2,fac=1,CL=[0.68],col=None,ax=None,linestyles=['-','-.','--'],nbins=50,withSave=False):
        if type(CL) is not np.array:
            CL=np.array(CL)

        if type(linestyles) is not list:
            linestyles = [linestyles]*np.size(CL)
        if len(linestyles) < np.size(CL):
            linestyles = linestyles*int(math.ceil(np.size(CL)/(1.*len(linestyles))))



        ls = np.array(linestyles[0:np.size(CL)])
        #the confidence levels need to be sorted from larger to smaller for contour
        #the corresponding linestyles are sorted as well.
        idx = np.argsort(-CL)

        CL  = CL[idx]
        ls = ls[idx]

        if col is not None:
            if type(col) is not list:
                col = [col]*np.size(CL)
            if len(linestyles) < np.size(CL):
                col = col*int(math.ceil(np.size(CL)/(1.*len(col))))
            print(idx)
            col =np.array(col)
            col=col[idx]


        if type(fac) is not list:
            fac=[fac,fac]
        elif len(fac)==1:
            fac=[fac[0]]*2

        if type(p1) is list:
            if p1[0] not in self.star_names:
                print("Error in plot_2D_contour: first argument of list for parameter p1: ",p1[0]," does not correspond to an existing star")
                print("Existing stars are: ",self.star_names)
                return
            if p1[1] not in STARS_PAR:
                print("Error in plot_2D_contour: second argument of list for parameter p1: ",p1[1]," does not correspond to an existing orbital parameter")
                print("Existing orbital parameters are: ",STARS_PAR)
                return
            idx_star = self.star_names.index(p1[0])
            print(' First parameter for the 2D contour plot is: ',p1[1],' from ',p1[0])
            ch1 = self.stars[idx_star].chain[p1[1]]
            xlab = p1[1]+' from '+p1[0]
            xname = p1[0]+'_'+p1[1]
        else:
            if p1 not in GLOB_PAR :
                print("Error in plot_2D_contour: first argument p1: ",p1," does not correspond to an existing global parameter for model ",self.BH.model)
                print("Existing parameters are: ",GLOB_PAR)
                return
            print(' First parameter for the 2D contour plot is: ',p1)
            ch1 = self.BH.chain_glob[p1]
            xlab = p1
            xname = p1

        if type(p2) is list:
            if p2[0] not in self.star_names:
                print("Error in plot_2D_contour: first argument of list for parameter p2: ",p2[0]," does not correspond to an existing star")
                print("Existing stars are: ",self.star_names)
                return
            if p2[1] not in STARS_PAR:
                print("Error in plot_2D_contour: second argument of list for parameter p2: ",p2[1]," does not correspond to an existing orbital parameter")
                print("Existing orbital parameters are: ",STARS_PAR)
                return

            idx_star = self.star_names.index(p2[0])#np.where(self.star_names==p2[0])[0][0]
            print(' Second parameter for the 2D contour plot is: ',p2[1],' from ',p2[0])
            ch2 = self.stars[idx_star].chain[p2[1]]
            ylab = p2[1]+' from '+p2[0]
            yname= p2[0]+'_'+p2[1]
        else:
            if p2 not in GLOB_PAR :
                print("Error in plot_2D_contour: first argument p2: ",p2," does not correspond to an existing global parameter for model ",self.BH.model)
                print("Existing parameters are: ",GLOB_PAR)
                return
            print(' Second parameter for the 2D contour plot is: ',p2)
            ch2 = self.BH.chain_glob[p2]
            ylab = p2
            yname = p2

        (hist, obins, ibins) = np.histogram2d(ch2*fac[1],ch1*fac[0], bins = nbins, weights = self.weights)
        levels = getContourLevels(hist,percents=CL)
        if ax is None:
            ax = plt.gca()
        if col is None:
            c = ax.contour(hist, levels,origin=None,extent = [ibins[0], ibins[-1], obins[0], obins[-1]],linestyles=ls)
        else:
            c = ax.contour(hist, levels,origin=None,extent = [ibins[0], ibins[-1], obins[0], obins[-1]],colors=col,linestyles=ls)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

        if withSave:
            for i,cl in enumerate(CL):
                x = c.collections[i].get_paths()[0].vertices
                np.savetxt(self.chain_dir+'/'+xname+'_'+yname+'_'+str(cl)+'.cont',x)
        return c
        
    #####################    #######    #######    #######    #######    #######    #######
    #   DEPRECEATED FUNCTIONS
    #####################    #######    #######    #######    #######    #######    #######
    def astro_residuals(self, star,dt=.05,CL=[0.68]):
        print('WARNING: use star.astro_residuals() instead of this function who is no longer working')
        star.astro_residuals(dt=dt,CL=CL)    

    def RV_residuals(self, star,dt=.05,CL=[0.68]):
        print('WARNING: use star.RV_residuals() instead of this function who is no longer working')
        star.RV_residuals(dt=dt,CL=CL)

    def plot_astro_residuals(self,star,outname=None):
        print('WARNING: use star.plot_astro_residuals() instead of this function who is no longer working')
        star.plot_astro_residuals(outname)

    def plot_RV_residuals(self,star,outname=None,ylim=None):
        print('WARNING: use star.plot_RV_residuals() instead of this function who is no longer working')
        star.plot_RV_residuals(outname,ylim=ylim)
    
    def get_pred_astro(self,star,t,col,CL=0.68,nbins=50):
        print('WARNING: use star.get_pred_astro(t,col,CL,nbins) instead of this function who is no longer working')
        return star.get_pred_astro(t,col,CL,nbins)
        
    def get_star_dist2Sgra_model(self,star,t,CL=[0.68]):
        print('WARNING: use star.get_dist2Sgra_model(t,CL) instead of this function who is no longer working')
        return star.get_star_dist2Sgra_model(t,CL)
        
    def get_star_xySgra_model(self,star,t,CL=[0.68]):
        print('WARNING: use star.get_xy2Sgra_model(t,CL) instead of this function who is no longer working')
        return star.get_star_xySgra_model(t,CL)
            

#    def get_star_biasfromSgra_model(self,star,t,mag,CL=[0.68]):
#        nt = np.size(t)
#        n_CL = 1
#        contour_list = [0.5 ]
#        if CL:
#            n_CL = 2*len(CL) +1
#            for cl in CL:
#                contour_list.append(0.5-cl*0.5)
#                contour_list.append(0.5+cl*0.5)
#
#        contour = np.array(contour_list)
#        xbias = np.zeros((nt,n_CL))
#        ybias = np.zeros((nt,n_CL))
#        dbias = np.zeros((nt,n_CL))
#
#        starflux=10**(-0.4*mag)
#
#        for i in range(nt):
#            X,V = star.orbit_Kepler(t[i])
#            if self.model != 'Kepler': #Romer time delay
#                X,V = star.orbit_Kepler(t[i]-X[2]/CAUYR)#
#
#            xch = -X[0]/self.chain_glob_par['R0']
#            ych = X[1]/self.chain_glob_par['R0']#
#
#            rch=np.sqrt(xch**2 + ych**2)
#            wf = np.multiply(np.exp(-(rch/BIASSIGMA)**2/2.) , FLUXSGRA/starflux)
#
#            xbiasch = wf *xch /(wf +1)
#            ybiasch = wf *ych /(wf +1)
#
#            dbiasch = np.sqrt(xbiasch**2+ybiasch**2)
#            xbias[i,:] = weighted_percentile(xbiasch, contour, weights=self.weights)
#            ybias[i,:] = weighted_percentile(ybiasch, contour, weights=self.weights)
#            dbias[i,:] = weighted_percentile(dbiasch, contour, weights=self.weights)
#        return (xbias,ybias,dbias)  


def multiprocess_astro_residuals(star,efit_obj,dt=.05,CL=[0.68],processes=4):
    print('WARNING: use star.astro_residuals(dt,CL,processes) instead of this function who is no longer working')
    star.astro_residuals(dt,CL,processes)

def multiprocess_RV_residuals(star,efit_obj,dt=.05,CL=[0.68],processes=4):
    print('WARNING: use star.RV_residuals(dt,CL,processes) instead of this function who is no longer working')
    star.RV_residuals(dt,CL,processes)
          

class EccAnomalyError(Exception):
    def __init__(self, message):
        self.message = message

