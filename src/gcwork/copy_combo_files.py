import numpy as np
import glob
import shutil
import os
def copy_combo_files(epochs_info,outdir):
    '''

    Copy the combo fits files referred to an epochInfo.txt file into
    the same directory, along with the starfiner residuals and .lis
    and .reg files

    Inputs
    ------
    epochs_info - epochsInfo.txt file location
    outdir - directory to put the files

    
    HISTORY
    -------
    2018-08-20 - T. Do
    '''

    t = np.loadtxt(epochs_info,dtype='str')
    epoch_name = t[:,0]
    directory = t[:,1]
    isAO = t[:,2]
    version = t[:,3]
    doAlign = t[:,4]
    
    dir_name = os.path.dirname(epochs_info)
    
    for i in np.arange(len(epoch_name)):
        if doAlign[i] == '1':

            
            if isAO[i] == '1':
                file_name = os.path.join(directory[i],'combo/mag'+epoch_name[i]+'.fits')
                res_file = os.path.join(directory[i],'combo/mag'+epoch_name[i]+'_res.fits')
                stf_lis = glob.glob(os.path.join(directory[i],'combo/starfinder/mag'+epoch_name[i]+'*rms*'))
<<<<<<< HEAD

=======
                stf_lis_all = glob.glob(os.path.join(directory[i],'combo/starfinder/mag'+epoch_name[i]+'*0.8_stf_cal.lis'))                
                psf_file = os.path.join(directory[i],'combo/mag'+epoch_name[i]+'_psf.fits')
>>>>>>> a9a8708d68233ca608c51f2bb67fba97e9386350
            else:
                file_name = os.path.join(directory[i],'final/'+epoch_name[i]+'_holo.fits')
                res_file = None
                stf_lis = glob.glob(os.path.join(dir_name,'../lis/mag'+epoch_name[i]+'_rms.lis'))
<<<<<<< HEAD

=======
                stf_lis_all = None
                psf_file = None
                
>>>>>>> a9a8708d68233ca608c51f2bb67fba97e9386350
            print('copying: '+file_name)
            shutil.copy(file_name,outdir)
            if res_file is not None:
                shutil.copy(res_file,outdir)
            print(stf_lis)
            for f in stf_lis:
                shutil.copy(f,outdir)
<<<<<<< HEAD
            
=======
            if stf_lis_all is not None:
                print(stf_lis_all)
                for f in stf_lis_all:
                    shutil.copy(f,outdir)
            if psf_file is not None:
                psf_dir = os.path.join(outdir,'psfs')
                if not(os.path.exists(psf_dir)):
                    os.makedirs(psf_dir)
                    
                shutil.copy(psf_file,psf_dir)
>>>>>>> a9a8708d68233ca608c51f2bb67fba97e9386350
