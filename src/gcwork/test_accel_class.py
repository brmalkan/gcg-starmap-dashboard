import accel_class


def test_confused_epochs():
    # test the manual removal of confused epochs
    testpath = '/g/ghez/align/lucy_16_06_17_holo4_abhimat_S0-38_test/'
    manualepochsfile = '/u/ghezgroup/data/gc/source_list/confused_epochs_astrometry.txt'
    
    a = accel_class.accelClass(rootDir=testpath,poly='polyfit_s/fit',
                               points='points_s/',align='align/align_d_rms_1000_abs_t')
    a.findNearestNeighbors()
    a.removeConfusedEpochs(mkPointsFiles=True,runPolyfit=False,debug=True,manualepochsfile=manualepochsfile)
