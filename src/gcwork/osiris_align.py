from astropy.table import Table
from flystar import starlists, align
import numpy as np
import pymysql as mdb
import pandas as pd
import ast
import matplotlib.pyplot as plt
from astropy.io import fits
import pdb
from matplotlib.colors import LogNorm

def get_fields(inBerkeley=False):
    if inBerkeley:
        pwFile='/g/lu/scratch/siyao/other/pw.txt'
        pw = open(pwFile).read().split()[0]

    else:
        in1 = open('/u/tdo/osiris/dblogin')
        pw = in1.readline().strip()
        in1.close()

    con = mdb.connect(host='galaxy1.astro.ucla.edu',user='dbread',passwd=pw,db='gcg')
    df_fields = pd.read_sql_query("SELECT short_name,stfdir,posfile,mosaic_flatten FROM fields", con)
    con.close()
    return df_fields
 

def name_osiris_stf(field='S4-3', 
        input_file = 'gcows_S4-3_stf_named.lis',
        f_label='/u/ghezgroup/data/gc/source_list/label.dat'):

    # readin stf directory from database
    df_fields = get_fields()
    df_fields = df_fields[df_fields.short_name==field]
    workDir = df_fields['stfdir'].iloc[0]

    # read in starlist and label.dat (use label.dat as reference)
    t1 = Table.read(workDir+input_file, format='ascii')
    t1.rename_column('col1', 'name')
    t1.rename_column('col2', 'm')
    t1.rename_column('col4', 'x')
    t1.rename_column('col5', 'y')
    t1['t'] = 0
    
        
    t2 = Table.read(f_label, format='ascii')
    t2.rename_column('col1', 'name')
    t2.rename_column('col2', 'm')
    t2.rename_column('col3', 'x')
    t2.rename_column('col4', 'y')
    t2.rename_column('col11', 't')
    
    # convert to starlists object
    sl = starlists.StarList()
    sl = sl.from_table(t1)
    label = starlists.StarList()
    label = label.from_table(t2)
    
    # match the two starlists
    mos = align.MosaicToRef(label, [sl], init_guess_mode='name', dr_tol=0.1, dm_tol=0.5)
    mos.fit()
    
    # write to pos file
    test = Table()
    test['name'] = mos.ref_table['name']
    test['x_orig'] = [i[0] for i in mos.ref_table['x_orig']]
    test['y_orig'] = [i[0] for i in mos.ref_table['y_orig']]
    test.write(workDir+'gcows_{0}_pos.txt'.format(field), format='ascii.no_header')

    # write to mag file
    test = Table()
    test['name'] = mos.ref_table['name']
    test['m_label'] = mos.ref_table['m0']
    test['x_orig'] = [i[0] for i in mos.ref_table['x_orig']]
    test['y_orig'] = [i[0] for i in mos.ref_table['y_orig']]
    test['m_label'].format = '.1f'
    test.write(workDir+'gcows_{0}_mag.txt'.format(field), format='ascii.no_header')
    

def get_vertices(vertices_orig='[[3.2,63.9], [65.1, 63.1], [64.6,8.4], [3.9, 7.7]]', 
        field='S4-3', 
        f_label='/u/ghezgroup/data/gc/source_list/label.dat',
        inBerkeley=False):
    """
    to work in Berkeley machien, inBerkeley=True"""

    df_fields = get_fields(inBerkeley=inBerkeley)
    df_fields = df_fields[df_fields.short_name==field]
    f_starlist = df_fields['posfile'].iloc[0]
    workDir = df_fields['stfdir'].iloc[0]
    img = df_fields['mosaic_flatten'].iloc[0]
    if inBerkeley:
        f_starlist = f_starlist.replace('/u/ghezgroup/', '/g/lu/')
        workDir = workDir.replace('/u/ghezgroup/', '/g/lu/')
        img = img.replace('/u/ghezgroup/', '/g/lu/')
        f_label = f_label.replace('/u/ghezgroup/', '/g/lu/')

    vertices_orig = ast.literal_eval(vertices_orig)

    t1 = Table.read(f_starlist, format='ascii')
    t1.rename_column('col1', 'name')
    t1.rename_column('col2', 'm')
    t1.rename_column('col3', 'x')
    t1.rename_column('col4', 'y')
    t1['t'] = 0

    t2 = Table.read(f_label, format='ascii')
    t2.rename_column('col1', 'name')
    t2.rename_column('col2', 'm')
    t2.rename_column('col3', 'x')
    t2.rename_column('col4', 'y')
    t2.rename_column('col11', 't')
 
    # convert to starlists object
    sl = starlists.StarList()
    sl = sl.from_table(t1)
    label = starlists.StarList()
    label = label.from_table(t2)

    # match the two starlists
    mos = align.MosaicToRef(label, [sl], init_guess_mode='name', dr_tol=0.1, dm_tol=0.5)
    mos.fit()

    # transfer the vertices
    x_vertices = []
    y_vertices = []

    for vertice in vertices_orig:
        x,y = mos.trans_list[0].evaluate(vertice[0]-1, vertice[1]-1)
        x_vertices.append(round(x,2))
        y_vertices.append(round(y,2))

    print('x_vertices', x_vertices)
    print('y_vertices', y_vertices)

    # make the plot
    plt.figure()
    im = fits.open(img)
    plt.imshow(im[0].data, origin='lower',norm=LogNorm())
    plt.plot([vertices_orig[i][0]-1 for i in [0,1,2,3,0]], [vertices_orig[i][1]-1 for i in [0,1,2,3,0]], 'g-')
    for i in range(4):
        plt.annotate('({0},{1})'.format(str(x_vertices[i]), str(y_vertices[i])), xy=(vertices_orig[i][0], vertices_orig[i][1]), color='r')
    plt.title(field)
    plt.xlabel('x (pixel)')
    plt.ylabel('y (pixel)')
    plt.savefig(workDir+'field_{0}.png'.format(field))
    plt.close()

    with open(workDir+'field_{0}_vertices.txt'.format(field),'w') as f:
        f.write(str(vertices_orig)+'\n')
        f.write(str(x_vertices)+'\n')
        f.write(str(y_vertices))

    return

