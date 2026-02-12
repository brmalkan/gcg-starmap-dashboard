 To use:
1. Copy the python files in this directory to your own directly. \
2. Edit the USER INPUT settings in _setup_iterative_fits.py. In particular, make sure to specify the correct directories from which you want to pull data for points and rv files. \
3. Run _setup_iterative_fits.py (this will produce the output in the \'91run1\'92 subdirectory of this example directory)\
4. Set run_num = 2 in _setup_iterative_fits.py and run it again. This will re-run orbits for all stars that had outliers flagged. \
5. Continue iterating by increasing run_num until no more outliers are flagged for any stars. The orbits.dat file produced from that iteration is the \'93final\'94 orbits.dat. \
6. After all orbits are finished running, run _iterativeFits_summary.py to get a spreadsheet summarizing the fits for all stars in the sample. }



Notes
- Recommend using systematic error parameter for fits of all stars other than S0-2
- Standard is to keep BH parameters fixed and fit astrometry only for creating orbits.dat. To include RV data, just set ‘no’ to ‘RV’ for stars that do have RV files when defining star_list in _setup_iterative_fits.py 
- This example in the repo just runs orbits for stars other than S0-2 since it uses a kepler model. Can either include S0-2 using Kepler model as an approximation, or just manually add in S0-2’s parameters to the top of the orbits.dat file after. 


Technical note:
https://docs.google.com/document/d/1LMK7eFi69s7oU8JaGtfYG5Blm1rXUpoabGPodP2X-k4/edit#heading=h.xkhkgwwpiq3u