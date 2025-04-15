"""
Identifier:     chili_wcs/coord_data.py
Name:           coord_data.py
Description:    Class for Handling Coordinates data,pixel coordinates and sky coordinates.
Author:         Yifei Xiong
Created:        2023-11-30
Modified-History:
    2023-11-30: add module header
    2024-12-4: add spaxel coords detection for ifu data
    2025-04-14: add background fitting and subtraction for ifu data
"""

from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord, Distance
from astropy.wcs import WCS as aWCS
import astropy.units as u
import numpy as np
import sep
from .load_data import LoadData
from astropy.time import Time
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.optimize import curve_fit

class CoordData():
    """
    Class for handling coordinate data.

    pixel coordinates from sep.sky coordinates from Gaia DR3.

    Parameters
    ----------
    data_class : object
        Class from csst_ifs_wcs.load_data.
    rad : float
        The radius for the Gaia DR3 query. unit: arcsec
    radec_point : tuple
        Tuple containing the RA and Dec values. If not provided, the RA and Dec
        values are taken from `data_class`.
    backparam : dict, optional
        Dictionary containing background subtraction parameters:
        - 'background': bool, whether to fit and subtract background
        - 'k': float, threshold factor for mask calculation
        - 'ndex': int, polynomial degree for background fitting

    Attributes
    ----------
    xy : numpy.ndarray
        Pixel coordinates extracted from the image in `data_class`.
    ra_point : float
        Right Ascension coordinate of the data center
    dec_point : float
        Declination coordinate of the data center
    radec_table : astropy.table.Table
        Table containing Gaia DR3 data for the specified RA and Dec within the
        given radius.
    radec : astropy.coordinates.SkyCoord
        SkyCoord object representing the celestial coordinates from `radec_table`.

    Methods
    -------
    pixel_coord(image)
        Determine the pixel coordinates of the star, with the bottom-left corner
        of the image as the origin (0,0).

    gaiadr3_query(ra, dec, rad=1.0, maxmag=25, maxsources=1000000)
        Acquire the Gaia DR3 catalog.

    sky_coord(radec_table)
        Extract Gaia RA and Dec coordinates from the Gaia DR3 catalog.
    """

    def __init__(self, data_class: LoadData, mode: str = "Guider", rad: float = None,
                 radec_point: np.ndarray = None, observe_time = Time('2024-11-11'), 
                 plot_result = False, backparam: dict = None):
        self.data_class = data_class
        if radec_point is not None:
            self.ra_point = radec_point[0]
            self.dec_point = radec_point[1]
        self.observe_time = observe_time
        self.mode = mode
        
        # Set default background parameters
        self.backparam = {
            'background': True,
            'k': 1.5,
            'ndex': 12
        }
        # Update default values if background parameters are provided
        if backparam is not None:
            self.backparam.update(backparam)
    
        if mode == "Guider":
            if rad is not None:
                self.rad = rad
            else:
                self.rad = 240
            self.img = data_class.img
            self.xy = self.pixel_coord(self.img)
            self.radec_table = self.gaiadr3_query([self.ra_point],
                                              [self.dec_point], rad = self.rad)
            self.radec = self.sky_coord(self.radec_table)        
            if plot_result == True:
                self.plot_pixel()
                self.plot_radec()
        
        if mode == "IFU":
            if rad is not None:
                self.rad = rad
            else:
                self.rad = 80
            self.xy, self.dbscan_labels = self.spaxel_coord(
                self.data_class.sum_fluxes,
                self.data_class.x_fiber, 
                self.data_class.y_fiber
            )
            self.radec_table = self.gaiadr3_query([self.ra_point],
                                              [self.dec_point], rad = self.rad)
            self.radec = self.sky_coord(self.radec_table)  
            if plot_result == True:
                self.plot_spaxel(self.data_class.sum_fluxes, self.data_class.x_fiber, self.data_class.y_fiber, self.xy, self.dbscan_labels)
                self.plot_radec()
        if mode == "Normal":
            if rad is not None:
                self.rad = rad
            else:
                self.rad = 240
            self.img = data_class
            self.xy = self.pixel_coord(self.img)
            self.radec_table = self.gaiadr3_query([self.ra_point],
                                              [self.dec_point], rad = self.rad)
            self.radec = self.sky_coord(self.radec_table)
            if plot_result == True:
                self.plot_pixel()
                self.plot_radec()

    def pixel_coord(self, image: np.ndarray):
        """
        Determine the pixel coordinates of the star,
        with the bottom-left corner of the image as the origin (0,0)

        Parameters
        ----------
        image : numpy 2d array
            An astronomical image with many star on it

        Returns
        -------
        xy : numpy 2d array ,shape is (N, 2)
            pixel coordinates of the stars
        """
        sep.set_extract_pixstack(1000000) 
        bkg = sep.Background(image, bw=5, bh=5, fw=3, fh=3, fthresh=0.0)  # extract background
        image_sub = image - np.median(bkg)  # substract background
        objects = sep.extract(image_sub, thresh = 5, err=bkg.globalrms)  # extract stars
        sort = np.argsort(-objects["flux"])
        a = objects['a'][sort]
        b = objects['b'][sort]
        e = a/b
        ecut = (e < 1.5)
        x = objects['x'][sort][ecut]
        y = objects['y'][sort][ecut]
        xycut = (x > 1) & (x < (image.shape[1]-1)) & (y > 1) & (y < (image.shape[0]-1))
        xy = np.column_stack([x, y])[xycut][:20]
        return xy

    def fit_background(self, x_fiber, y_fiber, flux):
        """
        Fit background surface and subtract it from the original data
        
        Parameters:
            x_fiber: x coordinates of fibers
            y_fiber: y coordinates of fibers
            flux: flux values for each fiber
        
        Returns:
            flux_bg_sub: flux values after background subtraction
            background: fitted background values
        """
        # Get polynomial order
        ndex = self.backparam['ndex']
        
        # Define background surface function (polynomial)
        def background_model(xy, *params):
            x, y = xy
            result = 0
            idx = 0
           
            # Build polynomial: sum_{i,j} p_{i,j} * x^i * y^j, where i+j <= ndex
            for i in range(ndex):
                for j in range(ndex-i):
                    result += params[idx] * (x**i) * (y**j)
                    idx += 1
            return result
        
        # Calculate number of polynomial parameters
        n_params = sum(ndex-i for i in range(ndex))
        
        # Filter out outliers (using median and standard deviation)
        median_flux = np.nanmedian(flux)
        std_flux = np.nanstd(flux)
        mask_normal = (flux < median_flux + 0.1*std_flux)
        
        # Check if filtered data is empty
        if np.sum(mask_normal) == 0:
            print("Warning: Filtered data is empty, adjusting filter conditions")
            # Use more relaxed conditions
            mask_normal = (flux < median_flux + 5*std_flux)
        
        x_fit = x_fiber[mask_normal]
        y_fit = y_fiber[mask_normal]
        z_fit = flux[mask_normal]
        
        # Fit background surface
        try:
            # Initialize all parameters to 0
            initial_params = np.zeros(n_params)
            popt, pcov = curve_fit(background_model, (x_fit, y_fit), z_fit, p0=initial_params)
            
            # Calculate fitted background values
            background = background_model((x_fiber, y_fiber), *popt)
            
            # Subtract background
            flux_bg_sub = flux - background
        except Exception as e:
            print(f"Background fitting failed: {e}")
            flux_bg_sub = flux
            background = np.zeros_like(flux)
            
        return flux_bg_sub, background

    def spaxel_coord(self, flux: np.ndarray, x_fiber: np.ndarray, y_fiber: np.ndarray, eps=1.4, min_samples=7):
        """
        Use DBSCAN algorithm and corrected second moments to find the centroid of star images

        Parameters:
            flux: Array of flux values for each IFU unit
            x_fiber, y_fiber: Coordinates of IFU units
            eps: DBSCAN neighborhood radius parameter
            min_samples: DBSCAN minimum sample number parameter
        Returns:
            List of centroid coordinates and labels for each star region
        """
        # Get background parameters
        background = self.backparam['background']
        k = self.backparam['k']
        
        # Decide whether to subtract background based on background parameter
        if background:
            # Subtract background
            flux_bg_sub, bg_values = self.fit_background(x_fiber, y_fiber, flux)
            # Calculate mask using background values as threshold
            self.mask = flux_bg_sub > k * np.nanstd(bg_values)
        else:
            # Use original data
            flux_bg_sub = flux
            # Calculate mask using median as threshold
            self.mask = (flux_bg_sub - np.nanmin(flux_bg_sub)) > k * np.nanmedian(flux_bg_sub - np.nanmin(flux_bg_sub))

        # Add flux values as the third dimension for clustering
        coords = np.column_stack((x_fiber[self.mask], y_fiber[self.mask], flux_bg_sub[self.mask] / np.max(flux_bg_sub[self.mask])))
        flux_values = flux_bg_sub[self.mask]

        # Use DBSCAN for clustering, considering distances in 3D space
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = clustering.labels_

        # Calculate the centroid for each cluster
        xy = []
        flux_list = [] 
        for label in set(labels):
            if label == -1:
                continue  # Ignore noise points
            region_mask = labels == label
            region_coords = coords[region_mask]
            region_flux = flux_values[region_mask]

            # Calculate initial centroid estimate
            total_flux = np.sum(region_flux)
            flux_list.append(total_flux)
            x_c = np.sum(region_coords[:, 0] * region_flux) / total_flux
            y_c = np.sum(region_coords[:, 1] * region_flux) / total_flux
            # Store results
            xy.append([x_c, y_c])
        # Sort xy by flux from high to low
        xy = np.array(xy)
        flux_list = np.array(flux_list)
        sorted_indices = np.argsort(flux_list)[::-1]
        xy = xy[sorted_indices]
        return np.array(xy), labels

    def gaiadr3_query(self,
                      ra: list,
                      dec: list,
                      rad: float = 1.0,
                      maxmag: float = 25,
                      maxsources: float = 1000000):
        """
        Acquire the Gaia DR3, from work of zhang tianmeng

        This function uses astroquery.vizier to query Gaia DR3 catalog.

        Parameters
        ----------
        ra : list
            RA of center in degrees.
        dec : list
            Dec of center in degrees.
        rad : float
            Field radius in degrees.
        maxmag : float
            Upper limit magnitude.
        maxsources : float
            Maximum number of sources.

        Returns
        -------
        astropy.table.Table
            Catalog of gaiadr3.

        Examples
        --------
        >>> catalog = gaiadr3_query([180.0], [30.0], 2.0)
        """

        vquery = Vizier(columns=[
            'RA_ICRS', 'DE_ICRS', 'pmRA', 'pmDE', 'Plx', 'RVDR2', 'Gmag'
        ],
                        row_limit=maxsources,
                        column_filters={
                            "Gmag": ("<%f" % maxmag),
                            "Plx": ">0"
                        })
        coord = SkyCoord(ra=ra, dec=dec, unit=u.deg, frame="icrs")
        r = vquery.query_region(coord,
                                radius=rad * u.arcsec,
                                catalog='I/355/gaiadr3')

        return r[0]

    def sky_coord(self, radectable):
        """

        Parameters
        ----------
        radec_table : astropy.table.Table
            Catalog of gaiadr3.

        Returns
        -------
        numpy 2d array
            gaia ra、dec
        """
        ra_s = radectable["RA_ICRS"]
        dec_s = radectable["DE_ICRS"]
        pmra_s = radectable["pmRA"]
        pmdec_s = radectable["pmDE"]
        paral_s = radectable["Plx"]
        Gmag = radectable["Gmag"]

        if self.observe_time is None:
            ra = np.array(ra_s)
            dec = np.array(dec_s)
            radec = np.column_stack([ra, dec])
            return radec
        else:
            # Calculate apply_space_motion for all coordinates
            c_all = SkyCoord(ra=ra_s, dec=dec_s,
                         distance=Distance(parallax=np.abs(paral_s) * u.mas),
                         pm_ra_cosdec=pmra_s,
                         pm_dec=pmdec_s,
                         obstime=Time(2016.0, format='jyear',
                                      scale='tcb'), frame="icrs")
            epoch_observe = self.observe_time
            c_plx_all = c_all.apply_space_motion(epoch_observe)
            ra = c_plx_all.ra.degree
            dec = c_plx_all.dec.degree

            # Filter results with Gmag < 18
            mag_filter_indices = Gmag < 18
            ra = ra[mag_filter_indices]
            dec = dec[mag_filter_indices]
            Gmag_filtered = Gmag[mag_filter_indices]

            # Sort by Gmag
            sorted_indices = np.argsort(Gmag_filtered)
            ra = ra[sorted_indices]
            dec = dec[sorted_indices]
            radec = np.column_stack([ra, dec])
            self.Gmag = Gmag_filtered[sorted_indices]
            return radec
        
    def plot_pixel(self):
        xys = self.xy 
        # img = image_sub
        fig = plt.figure(figsize=(5, 5))
        ax = plt.subplot()
        img = self.img
        m = np.mean(img)
        s = np.std(img)
        ax.imshow(img, cmap='gray', vmin=m-s, vmax=m+3*s, origin='lower')
        ax.scatter(xys[:, 0], xys[:, 1], s=20, marker='+', color="b", alpha=0.5, label="Detected Stars")
        ax.set_title("Guider Image", fontsize=15)
        ax.legend()

    def plot_spaxel(self,fluxes, x_fiber, y_fiber, xy, labels):
        plt.figure()
        self.data_class.to_2dmap()
        # Plot centroid positions
        if len(xy) > 0:  # Ensure there are centroid points
            plt.scatter(xy[:, 0], xy[:, 1], c='red', marker='+', s=100)

        # Add labels for different regions
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # Ignore noise points
            # Use the same mask as labels to select corresponding points
            region_x_fiber = x_fiber[self.mask][labels == label]
            region_y_fiber = y_fiber[self.mask][labels == label]
            plt.scatter(region_x_fiber, region_y_fiber, alpha=0.3)

        #plt.legend()
        plt.show()

    def plot_radec(self):
        ra = self.radec[:, 0]
        dec = self.radec[:, 1] 
        wcs = aWCS(naxis=2)
        wcs.wcs.crpix = [100, 100]
        wcs.wcs.crval = [self.ra_point, self.dec_point]
        wcs.wcs.cd = [[-2/3600,0],[0, 2/3600]]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        sizes = 40000 / (self.Gmag)**2  
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection=wcs)
        ax.scatter(ra, dec, s=sizes, facecolor="none", edgecolors="red", alpha=0.6, transform=ax.get_transform('world'))
        ax.set_xlabel("RA (degrees)")
        ax.set_ylabel("DEC (degrees)")
        ax.set_title("RA DEC of star catalog ")
        ax.grid(True)
        plt.show()