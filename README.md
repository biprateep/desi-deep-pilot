# Deep Spectroscopy with DESI for Photometric Redshift Training and Calibration

Documentation, codes and catalogs accompanying our paper, Deep Spectroscopy with DESI for Photometric Redshift Training and Calibration by Biprateep Dey, Jeffrey A. Newman and the DESI Collaboration members.

Website: biprateep.github.io/desi-deep-pilot

Contact: biprateep@pitt.edu | janewman@pitt.edu

### Python Code
Jupyter notebooks and supporting code used to analyse data and prepare results can be found in the `notebooks` directory. The code has also been archived on Zenodo at https://doi.org/10.5281/zenodo.20311880. Following is a brief desciption of the files. 
1. `select_targets_pilot.ipynb`:  Defines the LSST Year 1–equivalent target sample for photometric redshift training. Reads HSC photometry for the COSMOS field, applies quality cuts, and resamples ~100,000 targets to a uniform *i*-band magnitude distribution. Also shows overlap with external samples (LOWZ, Merian, HSC weak lensing). 
2. `explore_LSSTY1_pilot_data.ipynb`: Core data assembly notebook. Reads DESI spectra and redshift catalogs from three pilot fields (COSMOS, XMM-LSS, HERCULES), merges fiber assignment tables, HSC photometry, and Redrock outputs into a single master catalog.
3. `merge_VI_*.ipynb`: Three field-specific notebooks (identical workflow) that merge VI submissions from multiple human inspectors, detect and resolve conflicts, and write a final redshift quality catalog for each field. These ground-truth catalogs are used to validate automated Redrock redshifts.
4. `create_spectra_VI.ipynb`: Generates interactive HTML visual inspection pages using the DESI Prospect viewer. Groups spectra by exposure characteristics (single visit, bright/faint objects, conflicted cases) so inspectors can efficiently review and flag them.
5. `paper_general_stat.ipynb`: Produces the survey overview figures: sky coverage maps, magnitude distributions, and exposure time distributions across the three fields. 
6. `paper_measure_snr.ipynb`: Measures empirical signal-to-noise ratios from spectra by computing noise in spectral residuals (data minus best-fit Redrock template). Validates SNR scaling with exposure time and magnitude, providing the empirical basis for the time-to-redshift predictions.
7. `paper_plot_spectra.ipynb`: Selects four representative targets (passive/star-forming at high/low SNR) and plots smoothed example spectra with labeled emission and absorption features. Generates the example-spectra figure in the paper.
paper_redshift_efficiency.ipynb: Computes redshift success rates as a function of *i*-band magnitude and exposure time. Fits logistic regression models (binomial GLM) to the success probability, and compares DESI-II performance against DEEP2/3 and zCOSMOS. Also applies corrections for the *z* > 1.6 population.
8. `paper_time_to_redshift.ipynb`: Links redshift success rates to observing time and magnitude using power-law scaling, then connects those rates to the cosmological forecasts from forecast_w0wa.ipynb.
paper_telescope_time.ipynb: Estimates the total survey time required to build a deep photo-*z* training sample on a range of current and planned facilities (DESI, 4MOST, WEAVE, PFS, MOONS, DEIMOS, WST, MANIFEST-GMACS). Reads facility parameters from `telescope_params.yaml` and outputs a comparison table.
9. `forecast_w0wa.ipynb`: Runs Fisher matrix forecasts for dark energy constraints  using FisherA2Z. Varies the effective number density of photo-z training galaxies per redshift bin to quantify how photo-z degradation propagates to LSST Year 1 and Year 10 cosmological constraints.
10. `utils.py`: Utility functions shared across multiple notebooks.
11. `PowerLawScale.py`: A custom matplotlib scale class implementing a power-law axis transformation.
12. `forecasts_FoM.yam`: Pre-computed Fisher matrix Figure of Merit values for LSST Year 1 and Year 10 under various photo-z degradation scenarios (different n_eff fractions by redshift bin). Loaded by forecast_w0wa.ipynb to avoid recomputing Fisher matrices at runtime.
13. `telescope_params.yaml`: Instrument parameters (collecting area, field of view, multiplexing) for current and planned spectroscopic facilities. Loaded by paper_telescope_time.ipynb.



**Key Python Dependencies**
- numpy
- scipy
- matplotlib
- pandas
- astropy
- fitsio
- tqdm
- yaml
- scikit-learn
- statsmodels
- desispec
- redrock
- prospect
- pyccl
- fisherA2Z
- mpl_scatter_density
- prettytable


### Data Sets 
The data sets used for this work are available on this Zenodo repo: https://doi.org/10.5281/zenodo.19260375. Following is a brief description of the files.

 1. `merged_cat_LSST_WL_Y1.fits`: This file contains crossmatched HSC photometry, DESI redshift pipeline outputs and visual inspection labels for all the objects described in the article. 
 2. `HSC_COSMOS_I_mag_lim_24.8.fits`: Parent catalog of all objects in the DESI-COSMOS field which was used to select targets from.
 3. `HSC_XMM_I_mag_lim_24.8.fits`:  Parent catalog of all objects in the DESI-XMMLSS field which was used to select targets from.
4. `COSMOS_LSSTY1_target_list.fits` : Catalog of objects in the DESI-COSMOS field selected from the parent catalog after quality and magnitude cuts and downsampling. The DESI fiber assignment algorithm was run on this catalog.  
5. `XMM_LSSTY1_target_list.fits` : Catalog of objects in the DESI-COSMOS field selected from the parent catalog after quality and magnitude cuts and downsampling. The DESI fiber assignment algorithm was run on this catalog. 
6. `FIGURE-XX.zip`: Data points plotted in Figure-XX in the paper. 


