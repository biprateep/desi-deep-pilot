# VI_merge_functions_v1
# File for VI merging functions for both notebook and command-line use.
#


import os, sys, glob
import fnmatch
import re
from astropy.table import Table, join, vstack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d


"""
The followig three functions are a javascript to python translation 
of the utility employed by prospect. This works adequately for visualization
needs additional vetting if used for measurements
"""
import numpy as np

def piecewise_linear_interpolation(x_known, y_known):
    """
    Creates a piecewise linear interpolation function using numpy.interp
    with added linear extrapolation.

    This function handles x-axis points that are monotonically increasing or
    decreasing. For extrapolation, it calculates and uses the slope of the
    first and last line segments.

    Args:
        x_known (np.ndarray): The x-coordinates of the known data points.
                              Must be 1D and monotonic.
        y_known (np.ndarray): The y-coordinates of the known data points.

    Returns:
        function: A function that can be called with new x-coordinates to get
                  the interpolated/extrapolated y-coordinates.
    """
    # Ensure inputs are numpy arrays
    x_known = np.asarray(x_known)
    y_known = np.asarray(y_known)

    # 1. Check for monotonicity
    diff_x = np.diff(x_known)
    if not (np.all(diff_x > 0) or np.all(diff_x < 0)):
        raise ValueError("x_known must be monotonically increasing or decreasing.")
    if x_known.ndim != 1 or y_known.ndim != 1 or len(x_known) != len(y_known):
        raise ValueError("Inputs must be 1D arrays of the same length.")
    if len(x_known) < 2:
        raise ValueError("Need at least two points to interpolate.")

    # 2. Pre-calculate slopes and identify endpoints
    # The slope of the first segment
    slope_start = (y_known[1] - y_known[0]) / (x_known[1] - x_known[0])
    # The slope of the last segment
    slope_end = (y_known[-1] - y_known[-2]) / (x_known[-1] - x_known[-2])

    # Identify the min/max points for extrapolation anchoring
    if x_known[0] < x_known[-1]:  # Monotonically increasing
        x_min, y_min = x_known[0], y_known[0]
        x_max, y_max = x_known[-1], y_known[-1]
        slope_min, slope_max = slope_start, slope_end
    else:  # Monotonically decreasing
        x_min, y_min = x_known[-1], y_known[-1]
        x_max, y_max = x_known[0], y_known[0]
        # Slopes are swapped as well
        slope_min, slope_max = slope_end, slope_start


    # 3. Return the callable interpolator function
    def interpolator(x_new):
        """
        Calculates y-values for x_new points using interpolation
        or linear extrapolation.
        """
        x_new = np.asarray(x_new)
        y_new = np.empty_like(x_new, dtype=float)

        # Identify points for interpolation and extrapolation
        is_below = x_new < x_min
        is_above = x_new > x_max
        is_between = ~(is_below | is_above)

        # --- Perform Calculations ---
        # Interpolate using numpy.interp for points within the range
        y_new[is_between] = np.interp(x_new[is_between], x_known, y_known)

        # Extrapolate for points below the minimum
        # Formula: y = y_start + slope * (x_new - x_start)
        y_new[is_below] = y_min + slope_min * (x_new[is_below] - x_min)

        # Extrapolate for points above the maximum
        y_new[is_above] = y_max + slope_max * (x_new[is_above] - x_max)

        # If original input was a single value, return a single value
        return y_new if y_new.ndim > 0 else y_new.item()

    return interpolator
def get_kernel(nsmooth: int) -> np.ndarray:
    """
    Generates a Gaussian kernel for smoothing.

    This is a Python/NumPy equivalent of the get_kernel JavaScript function.

    Args:
        nsmooth: The standard deviation (sigma) of the Gaussian kernel in pixels.

    Returns:
        A 1D NumPy array containing the kernel values.
    """
    if nsmooth <= 0:
        return np.array([])
    # The kernel extends to 2*nsmooth on each side of the center, matching the JS implementation.
    x = np.arange(-2 * nsmooth, 2 * nsmooth + 1)
    kernel = np.exp(-x**2 / (2 * nsmooth**2))
    return kernel

def smooth_data(
    data_in: np.ndarray,
    kernel: np.ndarray,
    ivar_in: np.ndarray = None,
    ivar_weight: bool = False
) -> np.ndarray:
    """
    Smooths data using a provided kernel, with optional inverse variance weighting.

    This function vectorizes the logic from the original JavaScript `smooth_data`
    using `scipy.ndimage.convolve1d` for performance and accuracy, especially
    at the boundaries.

    Args:
        data_in: The input data array (e.g., flux).
        kernel: The smoothing kernel.
        ivar_in: The inverse variance array for weighting. Required if ivar_weight is True.
        ivar_weight: If True, apply inverse variance weighting.

    Returns:
        The smoothed data array.
    """
    if kernel.size == 0 or data_in.size == 0:
        return np.copy(data_in)

    # The JS code checks for finite values inside the loop. We can do this upfront
    # by creating a mask and zeroing out non-finite values.
    finite_mask = np.isfinite(data_in)
    if ivar_weight:
        if ivar_in is None:
            raise ValueError("ivar_in must be provided when ivar_weight is True.")
        if ivar_in.shape != data_in.shape:
             raise ValueError("ivar_in must have the same shape as data_in.")
        finite_mask &= np.isfinite(ivar_in)

    # Use a convolution operation, which is equivalent to the nested loops in JS.
    # The `convolve1d` function from SciPy handles boundary conditions gracefully.
    # `mode='constant'` with `cval=0` mimics the JS behavior of ignoring out-of-bounds pixels.
    
    if ivar_weight:
        # Equivalent to smooth(data*ivar) / smooth(ivar)
        # We multiply by the finite_mask to zero out non-finite values before convolution.
        numerator = convolve1d(
            (data_in * ivar_in) * finite_mask,
            kernel,
            mode='constant',
            cval=0.0
        )
        denominator = convolve1d(
            ivar_in * finite_mask,
            kernel,
            mode='constant',
            cval=0.0
        )
    else:
        # Equivalent to smooth(data) / smooth(ones)
        # The denominator correctly calculates the sum of kernel weights at each point,
        # accounting for edge effects, just like the JS version.
        numerator = convolve1d(
            data_in * finite_mask,
            kernel,
            mode='constant',
            cval=0.0
        )
        denominator = convolve1d(
            finite_mask.astype(float), # convolve with the mask to get the correct weights
            kernel,
            mode='constant',
            cval=0.0
        )

    # To avoid division by zero, set result to 0 where denominator is 0
    smoothed_data = np.zeros_like(data_in)
    np.divide(numerator, denominator, out=smoothed_data, where=denominator!=0)

    return smoothed_data

def smooth_noise(
    noise_in: np.ndarray,
    kernel: np.ndarray,
    ivar_weight: bool = False
) -> np.ndarray:
    """
    Smooths noise or ivar using a provided kernel, propagating errors correctly.

    This function vectorizes the logic from the original JavaScript `smooth_noise`.

    Args:
        noise_in: The input noise (stddev) or inverse variance (ivar) array.
        kernel: The smoothing kernel.
        ivar_weight: If True, `noise_in` is treated as ivar, and the error
                     propagation for a weighted mean is used. Otherwise, it's
                     treated as noise to be added in quadrature.

    Returns:
        The smoothed noise or ivar array.
    """
    if kernel.size == 0 or noise_in.size == 0:
        return np.copy(noise_in)

    finite_mask = np.isfinite(noise_in)
    
    if ivar_weight:
        # Propagating error for a weighted mean:
        # sigma_smooth^2 = sum(K_i^2 * ivar_i) / (sum(K_i * ivar_i))^2
        # We are calculating sigma_smooth.
        numerator_sq = convolve1d(
            noise_in * finite_mask, # noise_in is ivar here
            kernel**2,
            mode='constant',
            cval=0.0
        )
        denominator = convolve1d(
            noise_in * finite_mask,
            kernel,
            mode='constant',
            cval=0.0
        )
        
        # Calculate sqrt(numerator_sq) / denominator
        numerator = np.sqrt(numerator_sq)
    else:
        # Adding noise in quadrature:
        # sigma_smooth^2 = sum(K_i^2 * sigma_i^2) / (sum(K_i))^2
        # We are calculating sigma_smooth.
        numerator_sq = convolve1d(
            (noise_in**2) * finite_mask, # noise_in is sigma here
            kernel**2,
            mode='constant',
            cval=0.0
        )
        denominator = convolve1d(
            finite_mask.astype(float),
            kernel,
            mode='constant',
            cval=0.0
        )
        # Calculate sqrt(numerator_sq) / denominator
        numerator = np.sqrt(numerator_sq)

    smoothed_noise = np.zeros_like(noise_in)
    np.divide(numerator, denominator, out=smoothed_noise, where=denominator!=0)
    
    return smoothed_noise



def airtovac(w):
    """Convert air wavelengths to vacuum wavelengths. Don't convert less than 2000 Å.

    Parameters
    ----------
    w : :class:`float`
        Wavelength [Å] of the line in air.

    Returns
    -------
    :class:`float`
        Wavelength [Å] of the line in vacuum.
    """
    if w < 2000.0:
        return w
    vac = w
    for iter in range(2):
        sigma2 = (1.0e4/vac)*(1.0e4/vac)
        fact = 1.0 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)
        vac = w*fact
    return vac
def better_step(bin_edges, y, yerr=None, ax=None, **kwargs):
    """A 'better' version of matplotlib's step function
    Given a set of bin edges and bin heights, this plots the thing
    that I wish matplotlib's ``step`` command plotted. All extra
    arguments are passed directly to matplotlib's ``plot`` command.
    Args:
        bin_edges: The bin edges. This should be one element longer than
            the bin heights array ``y``.
        y: The bin heights.
        yerr: asymmetric error on y.
        ax (Optional): The axis where this should be plotted.
    """
    new_x = [a for row in zip(bin_edges[:-1], bin_edges[1:]) for a in row]
    new_y = [a for row in zip(y, y) for a in row]
    if ax is None:
        ax = plt.gca()
    p = ax.plot(new_x, new_y, **kwargs)
    if yerr is not None:
        new_yerr_lo = np.array([a for row in zip(yerr[0], yerr[0]) for a in row])
        new_yerr_up = np.array([a for row in zip(yerr[1], yerr[1]) for a in row])
        new_yerr = np.array([a for row in zip(yerr[0], yerr[1]) for a in row])
        ax.fill_between(
            new_x,  new_yerr_up, new_yerr_lo, alpha=0.1, color=p[0].get_color()
        )
    return ax

def choose_best_z(vi):
    # make new column with best redshift estimate for each VI - take VI redshift if available, else take Redrock redshift.
    vi["best_z"] = vi["VI_z"]
    # vi.loc[vi['best_z']=='--', 'best_z'] = vi.loc[vi['best_z']=='--', 'Redrock_z']
    vi.loc[vi["best_z"] == "", "best_z"] = vi.loc[vi["best_z"] == "", "Redrock_z"]
    vi["best_z"] = vi["best_z"].astype(float)

    # add new column to find how much deviation there is in the redshift estimates
    vi["dz"] = vi.groupby("TARGETID")["best_z"].transform(
        lambda x: ((x.max() - x.min()) / (1 + x.min()))
    )

    # if the deviation is small, fill best redshift with the mean redshift of the VI_z (which may be a VI and a Redrock mean if only one VI changed the z)
    vi.loc[vi["dz"] < 0.0033, ["best_z"]] = (
        vi.loc[vi["dz"] < 0.0033].groupby("TARGETID")["best_z"].transform("mean")
    )


def choose_best_spectype(vi):
    # make new column with best_spectype estimate for each VI - take VI_spectype if available, else take Redrock_spectype
    vi["best_spectype"] = vi["VI_spectype"]
    # vi.loc[vi['best_spectype']=='--', 'best_spectype'] = vi.loc[vi['best_spectype']=='--', 'Redrock_spectype']
    vi.loc[vi["best_spectype"] == "", "best_spectype"] = vi.loc[
        vi["best_spectype"] == "", "Redrock_spectype"
    ]


def choose_best_quality(vi):
    # add new columns, holding the mean of the flags and the maximum difference in flag classification
    vi["best_quality"] = vi.groupby("TARGETID")["VI_quality"].transform("mean")
    # If the classification difference is too big, give it 999 so it is clear the merger has to assess it
    vi["vi_class_diff"] = vi.groupby("TARGETID")["VI_quality"].transform(
        lambda x: (x.max() - x.min())
    )
    vi.loc[vi["vi_class_diff"] > 1, ["best_quality"]] = 999


def concatenate_all_issues(vi):
    ##make new column with issue flags - concatenate all issue flags from any VI.  We don't check these, we only check these if the VIs conflict for some other reason.
    vi["all_VI_issues"] = vi.groupby("TARGETID")["VI_issue"].transform(
        lambda x: "".join(set(list(x)))
    )
    # vi.loc[vi['all_VI_issues']!='--', 'all_VI_issues'] = vi.loc[vi['all_VI_issues']!='--', 'all_VI_issues'].transform(lambda x: ''.join(set(list(x))-set('-')))
    vi.loc[vi["all_VI_issues"] != "", "all_VI_issues"] = vi.loc[
        vi["all_VI_issues"] != "", "all_VI_issues"
    ].transform(lambda x: "".join(set(list(x))))


def concatenate_all_comments(vi):
    # add new column, with all comments concatenated
    vi["all_VI_comments"] = vi.groupby("TARGETID")["VI_comment"].transform(
        lambda x: " ".join(set(list(x)))
    )
    # vi.loc[vi["all_VI_comments"] != "", "all_VI_comments"] = vi.loc[
    #     vi["all_VI_comments"] != "--", "all_VI_comments"
    # ].transform(lambda x: x.replace("--", ""))
    vi["all_VI_comments"] = vi["all_VI_comments"].transform(lambda x: x.strip())


def add_extra_details(vi):
    # add new column, with the number of VI inspections for each object
    vi["N_VI"] = vi.groupby("TARGETID")["TARGETID"].transform("count")

    # add new column to hold comments from merger if needed
    vi["merger_comment"] = "none"


def read_in_data(VI_dir, tile, subset):
    # We will read all the *.csv files in this directory. Change as needed.
    all_files = os.listdir(VI_dir)
    vi_files = []

    # Choose a subset
    pattern = "desi*" + tile + "*" + subset + "*.csv"
    for entry in all_files:
        if fnmatch.fnmatch(entry, pattern):
            vi_files.append(entry)

    # Make sure the path exists to write the output file
    if not os.path.exists(VI_dir + "output"):
        os.makedirs(VI_dir + "output")

    # Read the first file in to vi to set up vi
    print("VI Files:")
    print(vi_files[0])
    vi = pd.read_csv(
        VI_dir + vi_files[0], delimiter=",", engine="python", keep_default_na=False
    )

    # Read in the rest of the files and append them to vi
    for i in range(1, len(vi_files)):
        print(vi_files[i])
        vi2 = pd.read_csv(
            VI_dir + vi_files[i], delimiter=",", engine="python", keep_default_na=False
        )
        vi = vi.append(vi2, ignore_index=True)
    # vi['TILEID']=tile
    # Change the column name to TARGETID to match standards elsewhere in DESI.
    # vi = vi.rename(columns={"TargetID": "TARGETID"})
    return vi


def read_in_data_cascades(VI_dir, tile, subset):
    # We will read all the *.csv files in this directory. Change as needed.
    all_files = os.listdir(VI_dir)
    vi_files = []

    # Choose a subset
    pattern = "desi*" + tile + "*" + "_" + subset + "_" + "*.csv"
    for entry in all_files:
        if fnmatch.fnmatch(entry, pattern):
            vi_files.append(entry)

    # Make sure the path exists to write the output file
    if not os.path.exists(VI_dir + "output"):
        os.makedirs(VI_dir + "output")

    # Read the first file in to vi to set up vi
    print("VI Files:")
    print(vi_files[0])
    vi = pd.read_csv(
        VI_dir + vi_files[0], delimiter=",", engine="python", keep_default_na=False
    )

    # Read in the rest of the files and append them to vi
    for i in range(1, len(vi_files)):
        print(vi_files[i])
        vi2 = pd.read_csv(
            VI_dir + vi_files[i], delimiter=",", engine="python", keep_default_na=False
        )
        vi = vi.append(vi2, ignore_index=True)
    # vi['TILEID']=tile
    # Change the column name to TARGETID to match standards elsewhere in DESI.
    # vi = vi.rename(columns={"TargetID": "TARGETID"})
    return vi


def read_in_data_fuji(VI_dir, tile, subset):
    # We will read all the *.csv files in this directory. Change as needed.
    all_files = os.listdir(VI_dir)
    vi_files = []

    # Choose a subset
    pattern = "desi*" + tile + "*" + "_" + subset + "_" + "*.csv"
    for entry in all_files:
        if fnmatch.fnmatch(entry, pattern):
            vi_files.append(entry)

    # Make sure the path exists to write the output file
    if not os.path.exists(VI_dir + "output"):
        os.makedirs(VI_dir + "output")

    # Read the first file in to vi to set up vi
    print("VI Files:")
    print(vi_files[0])
    vi = pd.read_csv(
        VI_dir + vi_files[0], delimiter=",", engine="python", keep_default_na=False
    )

    # Read in the rest of the files and append them to vi
    for i in range(1, len(vi_files)):
        print(vi_files[i])
        vi2 = pd.read_csv(
            VI_dir + vi_files[i], delimiter=",", engine="python", keep_default_na=False
        )
        vi = vi.append(vi2, ignore_index=True)
    # vi['TILEID']=tile
    # Change the column name to TARGETID to match standards elsewhere in DESI.
    # vi = vi.rename(columns={"TargetID": "TARGETID"})
    return vi


def add_auxiliary_data(vi, tiledir, tiles, nights, petals):
    # tf = Table.read(tiledir+'/'+tiles[0] + '/'+nights[0]+'/zbest-'+str(petals[0])+'-'+str(tiles[0])+'-'+nights[0]+'.fits',hdu='FIBERMAP')
    # tspec = Table.read(tiledir+'/'+tiles[0] + '/'+nights[0]+'/zbest-'+str(petals[0])+'-'+str(tiles[0])+'-'+nights[0]+'.fits',hdu='ZBEST')
    # for i in range(1,len(petals)):
    #    tfn = Table.read(tiledir+'/'+tiles[0] + '/'+nights[0]+'/zbest-'+str(petals[i])+'-'+str(tiles[0])+'-'+nights[0]+'.fits',hdu='FIBERMAP')
    #    tf = vstack([tf,tfn])
    #    tspecn = Table.read(tiledir+'/'+tiles[0] + '/'+nights[0]+'/zbest-'+str(petals[i])+'-'+str(tiles[0])+'-'+nights[0]+'.fits',hdu='ZBEST')
    #    tspec = vstack([tspec,tspecn])
    dataname = (
        tiledir
        + "/"
        + tiles[0]
        + "/deep/zbest-"
        + str(petals[0])
        + "-"
        + str(tiles[0])
        + "-deep.fits"
    )
    tf = Table.read(dataname, hdu="FIBERMAP")
    tspec = Table.read(dataname, hdu="ZBEST")
    for i in range(1, len(petals)):
        tfn = Table.read(
            tiledir
            + "/"
            + tiles[0]
            + "/deep/zbest-"
            + str(petals[i])
            + "-"
            + str(tiles[0])
            + "-deep.fits",
            hdu="FIBERMAP",
        )
        tf = vstack([tf, tfn])
        tspecn = Table.read(
            tiledir
            + "/"
            + tiles[0]
            + "/deep/zbest-"
            + str(petals[i])
            + "-"
            + str(tiles[0])
            + "-deep.fits",
            hdu="ZBEST",
        )
        tspec = vstack([tspec, tspecn])

    EXPID = list(set(tf["EXPID"]))[-1]
    """This was 0 before but  """
    tf = tf[tf["EXPID"] == EXPID]
    # tf_df = tf['TARGETID','TARGET_RA','TARGET_DEC','FIBER','FLUX_G','FLUX_R','FLUX_Z','FIBERFLUX_G','FIBERFLUX_R','FIBERFLUX_Z','EBV'].to_pandas()
    tf_df = tf.to_pandas()
    tspec_df = tspec[
        "TARGETID", "DELTACHI2", "ZWARN", "ZERR", "CHI2", "NPIXELS"
    ].to_pandas()
    for i_coeff in range(0, 10):
        tspec_df["COEFF_" + str(i_coeff)] = tspec["COEFF"].T[i_coeff]
    vi = vi.merge(tf_df, how="left", on="TARGETID", suffixes=("", "_y"))
    vi = vi.merge(tspec_df, how="left", on="TARGETID", suffixes=("", "_y"))
    print(len(vi.columns))
    vi.drop(vi.filter(regex="_y$").columns.tolist(), axis=1, inplace=True)
    print(len(vi.columns))
    return vi


def find_conflicts(vi_gp):
    # Choose spectra where VI has disagreed
    vi_conflict = vi_gp.filter(
        lambda x: (  # x is a group by TARGETID
            (
                (x["VI_quality"].max() - x["VI_quality"].min()) >= 2
            )  # Quality differs by 2 or more.
            | (
                x["dz"].max() >= 0.0033
            )  # Redshift difference is >=0.0033 (approx 1000km/s at low-z).
            | (
                not all(i == x["best_spectype"].iloc[0] for i in x["best_spectype"])
            )  # best_spectype differs.
        )
        & (len(x) >= 2)
    )  # And there is more than one VI
    return vi_conflict


def print_conflict(vi, unique_targets, conflict_id):
    # function to display the conflict in table format and open a prospect window
    print(
        vi[vi.TARGETID == unique_targets[conflict_id]][
            [
                "TARGETID",
                "Redrock_spectype",
                "VI_spectype",
                "best_spectype",
                "Redrock_z",
                "VI_z",
                "best_z",
                "VI_quality",
                "best_quality",
                "VI_issue",
                "all_VI_issues",
                "VI_comment",
                "merger_comment",
                "VI_scanner",
            ]
        ]
    )


def choose_spectype(argument):
    switcher = {"s": "STAR", "g": "GALAXY", "q": "QSO"}
    return switcher.get(argument, "Invalid_switch")


def issue_match(strg, search=re.compile(r"[^RCSN]").search):
    # Clean up issues
    return not bool(search(strg))


def print_conflicts_for_prospect(unique_targets):
    unique_target_csv = str(unique_targets[0])
    for target in unique_targets[1:]:
        unique_target_csv = unique_target_csv + ", " + str(target)
    print(
        'Copy the following list of problematic targets in to the "targets" list in Prospect_targetid.ipynb'
    )
    # On the wiki start from Computing/JupyterAtNERSC
    print("Targets with problematic VI: ", unique_target_csv)
    print("Total number of conflicts to resolve: ", len(unique_targets))


def print_merged_file(vi_gp, output_file):
    # 	vi_gp['Redrock_z', 'best_z', 'best_quality', 'Redrock_spectype', 'best_spectype', 'all_VI_issues', 'all_VI_comments', 'merger_comment','N_VI','DELTACHI2', 'ZWARN', 'ZERR','TARGET_RA','TARGET_DEC','FIBER','FLUX_G', 'FLUX_R', 'FLUX_Z','FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z', 'EBV','TILEID'].first().to_csv(output_file)
    vi_gp[
        "Redrock_z",
        "best_z",
        "best_quality",
        "Redrock_spectype",
        "best_spectype",
        "all_VI_issues",
        "all_VI_comments",
        "merger_comment",
        "N_VI",
        "Redrock_deltachi2",
        "TILEID",
    ].first().to_csv(output_file)


if __name__ == "__main__":
    print("What a cool program.")
