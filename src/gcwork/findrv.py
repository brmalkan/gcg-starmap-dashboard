#!/usr/bin/env python

# reads sqlite database and outputs the S0-2 RVs
# prints RVs to the screen: python findrv.py S0-2
# prints RVs to S0-2.rv file: python findrv.py -o S0-2

try:
    import MySQLdb as mdb
except:
    try:
        import mysql.connector as mdb
    except:
        import pymysql as mdb

import argparse
import os
from datetime import datetime
import numpy as np

def run(star,pwd,output=False,source=False,snr=20.,gaussian=False,keck=False,vlt=False,Observed=False,debug=False):
    con = mdb.connect(host='galaxy1.astro.ucla.edu',user='dbread',passwd=pwd,db='gcg')
    cur = con.cursor()

    if Observed or vlt:
        vlsrstr = 'vz'
    else:
        vlsrstr = 'vlsr'

#query = 'SELECT name,date,ddate,'+vlsrstr+',vz_err,source,mjd FROM spectra WHERE name = %s AND '+vlsrstr+' != "" ORDER BY ddate'

#if args.dev:
#    dev = " AND (date not in ('2003-06-08','2003-06-09')) "
#else:
#    dev = ''
    dev = ''
   
    if gaussian:
        query = "SELECT ucla,date,ddate,vz,vz_err,source,mjd,vsys_err,vsys_offset from gillessen2017 where (vz_err > 0) AND (vz is not NULL) and (ucla=%s) and (source !='Keck') UNION ALL select name,date,ddate,vlsr,vz_err,source,mjd,vsys_err,vsys_offset from spectra WHERE (vz_err > 0) AND (name =%s) and (source !='VLT') and (vlsr is not NULL) AND (quality < 2. OR quality = 132 OR quality IS NULL) AND snr > %s  order by ddate"
    else:
        query = "SELECT ucla,date,ddate,vz,vz_err,source,mjd,vsys_err,vsys_offset from gillessen2017 where (vz_err > 0) AND (vz is not NULL) and (ucla=%s) and (source !='Keck') UNION ALL select name,date,ddate,vlsr,vz_err,source,mjd,vsys_err,vsys_offset from spectra WHERE (vz_err > 0) AND (name =%s) and (source !='VLT') and (vlsr is not NULL) AND (source in ('NIRSPEC')) "+dev+" UNION ALL select name,date,ddate,vlsr,vz_err,source,mjd,vsys_err,vsys_offset from starkit WHERE (name =%s) AND (quality < 2. OR quality = 132 OR quality IS NULL) AND SNR > %s ORDER by ddate"

    if keck:
        if gaussian:
            query = 'SELECT name,date,ddate,'+vlsrstr+',vz_err,source,mjd,vsys_err,vsys_offset FROM starkit WHERE (vz_err > 0) AND name = %s AND source != "VLT" and (vlsr is not NULL) AND '+vlsrstr+' != "" AND (quality < 2. OR quality = 132 OR quality IS NULL) AND snr > %s ORDER BY ddate'
        else:
            query = "select name,date,ddate,"+vlsrstr+",vz_err,source,mjd,vsys_err,vsys_offset from spectra WHERE (vz_err > 0) AND (name =%s) and (source !='VLT') and (vlsr is not NULL) AND (source in ('NIRSPEC')) UNION ALL select name,date,ddate,"+vlsrstr+",vz_err,source,mjd,vsys_err,vsys_offset from starkit WHERE (name =%s) AND (quality < 2. OR quality = 132 OR quality IS NULL) AND SNR > %s ORDER by ddate"


    if vlt:
        query = 'SELECT name,date,ddate,vz,vz_err,source,mjd,vsys_err,vsys_offset FROM gillessen2017 WHERE (vz_err > 0) AND ucla = %s AND source = "VLT" AND '+vlsrstr+' != "" ORDER BY ddate'    

    if keck & gaussian:
        cur.execute(query,(star,snr))
    elif keck:
        cur.execute(query,(star,star,snr))    
    elif vlt:
        cur.execute(query,(star,))
    elif gaussian:
        cur.execute(query,(star,star,snr))    
    else:
        cur.execute(query,(star,star,star,snr))

    r = cur.fetchall()
    if output:
        filename = star+'.rv'
        outputF = open(filename,'w')
        outputF.write("# Created on "+str(datetime.now())+"\n")
        outputF.write("# SQL QUERY: "+query+"\n")
    for a in r:
        if debug:
            print(a)
    #date = Time(a[1])
    #d_days = date.mjd+0.5 - 51179.00  # add half a day to be at 12:00:00 UT
    #year = 1999.0000 + d_days/365.242

    # add systematic errors in quadrature if available
        if a[7] is not None:
            err = np.sqrt(a[4]**2+a[7]**2)
        else:
            err = a[4]

        # add systematic offset if available
        if a[8] is not None:
            vel = a[3] + a[8]
        else:
            vel = a[3]
        if output:
            if source:
                outputF.write('%9.6f %8.2f %4.2f %13.6f %s\n' % (a[2],vel,err,a[6],a[5]))
            else:
                outputF.write('%9.6f %8.2f %4.2f %13.6f\n' % (a[2],vel,err,a[6]))
        
        else:
            if source:
                print('%9.6f %8.2f %4.2f %13.6f %s' % (a[2],vel,err,a[6],a[5]))
            else:
                print('%9.6f %8.2f %4.2f %13.6f' % (a[2],vel,err,a[6]))

    if output:
        outputF.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find the radial velocity measurements for a particular star in the GCG database. By default, this will use the fits from the starkit table for OSIRIS data and will return also all literature values for that star.')
    parser.add_argument("star",help="Name of star to look up RV (example: S0-2)",type=str)
    parser.add_argument("-d","--debug",help="Print out debug info",action='store_true')
    parser.add_argument("-o","--output",help="Output RVs into a file called star.rv",action='store_true')
    parser.add_argument("-p","--password",help="Password for mysql database (galaxy1.astro.ulca.edu)")
    parser.add_argument("-g","--gaussian",help="Use previous RVs derived from a Gaussian fit instead of from StarKit",action='store_true')
    parser.add_argument("-k","--keck",help="Use only RVs measured at Keck",action='store_true')
    parser.add_argument("-v","--vlt",help="Use only RVs measured at VLT",action='store_true')
#    parser.add_argument("-n","--nomjd",help="Don't return the MJD column (old behavior)",action='store_true')
    parser.add_argument("-O","--Observed",help="Don't apply VLSR correction, return observed velocity. This option is only available for Keck-only data.",action='store_true')
    parser.add_argument("--snr",help="Only return SNR above this value (default: 20)",type=float,default=20.0)
    parser.add_argument("--source",help="Include a column with the source of the RV",action='store_true')
    #parser.add_argument("--dev",help="Use development version of NIRC2 RVs",action='store_true')
    args = parser.parse_args()
    run(args.star,pwd=args.password,output=args.output,source=args.source,snr=args.snr,gaussian=args.gaussian,keck=args.keck,vlt=args.vlt,Observed=args.Observed,debug=args.debug)


