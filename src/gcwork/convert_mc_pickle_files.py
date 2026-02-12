import argparse
import numpy as np
import pandas as pd
import pickle
import os
parser = argparse.ArgumentParser(description="Convert Monte Carlo files from StarOrbitsMC into text files")
parser.add_argument('files', metavar='F',type=str,nargs='+',help='File(s) to convert')
parser.add_argument('--outdir',dest='outdir',default='./')


args = parser.parse_args()
print('Input files: ',args.files)
print('Output directory: ',args.outdir)


for tempfile in args.files:
    mc = pickle.load(open(tempfile))
    df = pd.DataFrame({'i':mc.i,
                       'e':mc.e,
                       'evec1':mc.evec[:,0],
                       'evec2':mc.evec[:,1],
                       'evec3':mc.evec[:,2],
                       'w':mc.w,
                       'o':mc.o,
                       'p':mc.p,
                       't0':mc.t0,
                       'ph':mc.ph,
                       'x':mc.x,
                       'y':mc.y,
                       'z':mc.z,
                       'vx':mc.vx,
                       'vy':mc.vy,
                       'vz':mc.vz,
                       'ar':mc.ar,
                       'm':mc.m,
                       'r0':mc.r0,
                       'x0':mc.x0,
                       'y0':mc.y0})


    filepart = os.path.split(tempfile)
    outfile = args.outdir+os.path.splitext(filepart[-1])[0]+'.csv'

    print('saving file: '+outfile)
    df.to_csv(outfile,index=False)
