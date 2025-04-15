import numpy as np
from collections import OrderedDict
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MatplotlibPolygon, FancyArrowPatch
from astropy import wcs as astropy_wcs
from astroquery.hips2fits import hips2fits
import os
from requests.exceptions import ReadTimeout
import time
from .load_data import LoadIWCS
from .wcs_solver import WCS

class ChiliPlanTool:
    """
    Tool for predicting WCS parameters and pointing information for CHILI guider cameras
    
    This tool predicts WCS parameters, pointing and PA angle for both Guider and 
    Nasmyth Guider based on IFU pointing and PA angle, and generates sky images 
    showing the field of view of all three instruments
    """
    
    def __init__(self, ra_IFU, dec_IFU, PA_IFU, iwcs_path, save_path=None, plot=True):
        """
        Initialize ChiliPlanTool
        
        Parameters
        ----------
        ra_IFU : float
            Right ascension of IFU pointing (degrees)
        dec_IFU : float
            Declination of IFU pointing (degrees)
        PA_IFU : float
            Position angle of IFU (degrees)
        iwcs_path : str
            Path to IWCS reference file
        save_path : str, optional
            Path to save generated images, default is None
        plot : bool, optional
            Whether to automatically plot all images, default is True
        """
        # Save IFU parameters
        self.ra_IFU = ra_IFU
        self.dec_IFU = dec_IFU
        self.PA_IFU = PA_IFU
        
        # Set save path
        self.save_path = save_path
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Load IWCS parameters
        self.iwcs = LoadIWCS(iwcs_path)
        self.guider_iwcspara = self.iwcs.guider_iwcspara
        self.nasmyth_iwcspara = self.iwcs.guider_nasmyth_iwcspara
        
        # Set camera parameters
        self.guiderimg_shape = (700, 934)
        self.nasmythimg_shape = (2822/4, 4144/4)
        self.ifu_shape = (710, 650)
        self.pixel_size_guider = 9.55992827e-05
        self.pixel_size_nasmyth_guider = 4.32230639e-05 * 4
        self.pixel_size_ifu = 2.76923077e-05
        
        # Set HIPS URL
        self.hips_url = 'CDS/P/DSS2/color'
        
        # Calculate guider camera parameters
        self.calculate_guider_params()
        print(self.ra_dec_IFU_text)
        print(self.ra_dec_Guider_text)
        print(self.ra_dec_Nasmyth_text)
        # Create WCS and get images
        self.create_wcs()
        

        self.get_hips_images()
        
        # Automatically plot if requested
        if plot:
            self.plot_all()
        
    def calculate_guider_params(self):
        """
        Calculate WCS parameters, pointing and PA angle for Guider and Nasmyth Guider
        """
        # Calculate Guider parameters
        guider_pointing = self._calculate_pointing(
            self.guider_iwcspara["ICRVAL1"],
            self.guider_iwcspara["ICRVAL2"],
            self.guider_iwcspara["ILONPOLE"]
        )
        self.guider_ra, self.guider_dec, self.guider_PA = guider_pointing
        
        # Calculate Nasmyth Guider parameters
        nasmyth_pointing = self._calculate_pointing(
            self.nasmyth_iwcspara["ICRVAL1"],
            self.nasmyth_iwcspara["ICRVAL2"],
            self.nasmyth_iwcspara["ILONPOLE"]
        )
        self.nasmyth_ra, self.nasmyth_dec, self.nasmyth_PA = nasmyth_pointing
        
        # Calculate WCS parameters for Guider and Nasmyth Guider
        self.guider_wcspara = self._calculate_wcs_params(
            self.guider_ra, self.guider_dec, self.guider_PA,
            self.guiderimg_shape, self.pixel_size_guider
        )
        
        self.nasmyth_wcspara = self._calculate_wcs_params(
            self.nasmyth_ra, self.nasmyth_dec, self.nasmyth_PA,
            self.nasmythimg_shape, self.pixel_size_nasmyth_guider
        )
        
        # Calculate distance between IFU and Guider (angular)
        self.distance_ifu_guider = 90 - self.guider_iwcspara["ICRVAL2"]
        
        # Calculate distance between IFU and Nasmyth Guider (angular)
        self.distance_ifu_nasmyth = 90 - self.nasmyth_iwcspara["ICRVAL2"]
        
        # Format coordinate display
        self.ifu_coord = SkyCoord(ra=self.ra_IFU*u.deg, dec=self.dec_IFU*u.deg, frame='icrs')
        self.ra_dec_IFU_text = f"IFU Center: RA={self.ifu_coord.ra.to_string(unit=u.hourangle, sep=':', precision=2)}, DEC={self.ifu_coord.dec.to_string(unit=u.deg, sep=':', precision=2)}"
        
        self.guider_coord = SkyCoord(ra=self.guider_ra*u.deg, dec=self.guider_dec*u.deg, frame='icrs')
        self.ra_dec_Guider_text = f"Guider Center: RA={self.guider_coord.ra.to_string(unit=u.hourangle, sep=':', precision=2)}, DEC={self.guider_coord.dec.to_string(unit=u.deg, sep=':', precision=2)}"
        
        self.nasmyth_coord = SkyCoord(ra=self.nasmyth_ra*u.deg, dec=self.nasmyth_dec*u.deg, frame='icrs')
        self.ra_dec_Nasmyth_text = f"Nasmyth Guider Center: RA={self.nasmyth_coord.ra.to_string(unit=u.hourangle, sep=':', precision=2)}, DEC={self.nasmyth_coord.dec.to_string(unit=u.deg, sep=':', precision=2)}"
    
    def _calculate_pointing(self, ICRVAL1, ICRVAL2, ILONPOLE):
        """
        Calculate pointing and PA angle for guider camera
        
        Parameters
        ----------
        ICRVAL1 : float
            Longitude of the "north pole" of IFS local celestial coordinates in Guider local celestial coordinates
        ICRVAL2 : float
            Latitude of the "north pole" of IFS local celestial coordinates in Guider local celestial coordinates
        ILONPOLE : float
            Longitude of the "north pole" of Guider local celestial coordinates in IFS local celestial coordinates
            
        Returns
        ----------
        tuple
            (ra, dec, PA): Right ascension, declination and position angle of guider camera
        """
        # Calculate guider camera pointing
        # Input:
        lon_gui_i = ILONPOLE  # guider center longitude in IFS native sky
        lat_gui_i = ICRVAL2   # guider center latitude in IFS native sky
        
        # Parameter
        lon_ifu_p = self.ra_IFU
        lat_ifu_p = self.dec_IFU
        lon_pole_i = 180 + self.PA_IFU  # Polar longitude in IFS native sky
        
        
        # Output
        # Calculate guider camera center pointing
        ra0_guider, dec0_guider = WCS.sphere_rotate(
            phi=lon_gui_i,
            theta=lat_gui_i, 
            ra0= lon_ifu_p, 
            dec0= lat_ifu_p, 
            phi_p=lon_pole_i
        )
        
        # Calculate guider camera PA
        # Input:
        lon_pole_i = 180 + self.PA_IFU  # Polar longitude in IFS native sky
        lat_pole_i = self.dec_IFU   # Polar latitude in IFS native sky
        # Parameter:
        lon_ifu_g  = ICRVAL1
        lat_ifu_g  = ICRVAL2
        lon_gui_i = ILONPOLE

        lon_pole_g, lat_pole_g = WCS.sphere_rotate(
            phi=lon_pole_i,
            theta=lat_pole_i, 
            ra0=lon_ifu_g, 
            dec0=lat_ifu_g, 
            phi_p=lon_gui_i
        )
        PA_guider = lon_pole_g - 180
        
        return ra0_guider, dec0_guider, PA_guider
    
    def _calculate_wcs_params(self, ra, dec, PA, img_shape, pixel_size):
        """
        Calculate WCS parameters
        
        Parameters
        ----------
        ra : float
            Right ascension (degrees)
        dec : float
            Declination (degrees)
        PA : float
            Position angle (degrees)
        img_shape : tuple
            Image dimensions (height, width)
        pixel_size : float
            Pixel size (degrees/pixel)
            
        Returns
        ----------
        OrderedDict
            Dictionary of WCS parameters
        """
        theta = np.deg2rad(PA)  # rotation angle
        
        wcspara = OrderedDict({
            "CRPIX1": float((img_shape[1] + 1) / 2),
            "CRPIX2": float((img_shape[0] + 1) / 2), 
            "CRVAL1": float(ra),
            "CRVAL2": float(dec),
            "CD1_1": float(-np.cos(theta) * pixel_size),
            "CD1_2": float(np.sin(theta) * pixel_size),
            "CD2_1": float(np.sin(theta) * pixel_size),
            "CD2_2": float(np.cos(theta) * pixel_size)
        })
        return wcspara
    
    def create_wcs(self):
        """
        Create overall WCS and individual instrument WCS objects
        """
        # Create overall WCS
        self.wcs = astropy_wcs.WCS(header={
            'NAXIS1': 1200,
            'NAXIS2': 1200,
            'WCSAXES': 2,
            'CRPIX1': 600.5,
            'CRPIX2': 600.5,
            'CUNIT1': 'deg',
            'CUNIT2': 'deg',
            'CTYPE1': 'RA---TAN',
            'CTYPE2': 'DEC--TAN',
            'CRVAL1': self.ra_IFU,
            'CRVAL2': self.dec_IFU,
            'CD1_1': -2/3600,
            'CD1_2': 0,
            'CD2_1': 0,
            'CD2_2': 2/3600
        })
        
        # Create IFU WCS directly using header
        self.ifu_wcs = astropy_wcs.WCS(header={
            'NAXIS1': self.ifu_shape[1],
            'NAXIS2': self.ifu_shape[0],
            'WCSAXES': 2,
            'CRPIX1': (self.ifu_shape[1] + 1) / 2,
            'CRPIX2': (self.ifu_shape[0] + 1) / 2,
            'CUNIT1': 'deg',
            'CUNIT2': 'deg',
            'CTYPE1': 'RA---TAN',
            'CTYPE2': 'DEC--TAN',
            'CRVAL1': self.ra_IFU,
            'CRVAL2': self.dec_IFU,
            'CD1_1': -np.cos(np.deg2rad(self.PA_IFU)) * self.pixel_size_ifu,
            'CD1_2': np.sin(np.deg2rad(self.PA_IFU)) * self.pixel_size_ifu,
            'CD2_1': np.sin(np.deg2rad(self.PA_IFU)) * self.pixel_size_ifu,
            'CD2_2': np.cos(np.deg2rad(self.PA_IFU)) * self.pixel_size_ifu
        })
        
        # Create Guider WCS
        self.guider_wcs = astropy_wcs.WCS(header={
            'NAXIS1': self.guiderimg_shape[1],
            'NAXIS2': self.guiderimg_shape[0],
            'WCSAXES': 2,
            'CRPIX1': self.guider_wcspara["CRPIX1"],
            'CRPIX2': self.guider_wcspara["CRPIX2"],
            'CUNIT1': 'deg',
            'CUNIT2': 'deg',
            'CTYPE1': 'RA---TAN',
            'CTYPE2': 'DEC--TAN',
            'CRVAL1': self.guider_wcspara["CRVAL1"],
            'CRVAL2': self.guider_wcspara["CRVAL2"],
            'CD1_1': self.guider_wcspara["CD1_1"],
            'CD1_2': self.guider_wcspara["CD1_2"],
            'CD2_1': self.guider_wcspara["CD2_1"],
            'CD2_2': self.guider_wcspara["CD2_2"]
        })
        
        # Create Nasmyth Guider WCS
        self.nasmyth_wcs = astropy_wcs.WCS(header={
            'NAXIS1': self.nasmythimg_shape[1],
            'NAXIS2': self.nasmythimg_shape[0],
            'WCSAXES': 2,
            'CRPIX1': self.nasmyth_wcspara["CRPIX1"],
            'CRPIX2': self.nasmyth_wcspara["CRPIX2"],
            'CUNIT1': 'deg',
            'CUNIT2': 'deg',
            'CTYPE1': 'RA---TAN',
            'CTYPE2': 'DEC--TAN',
            'CRVAL1': self.nasmyth_wcspara["CRVAL1"],
            'CRVAL2': self.nasmyth_wcspara["CRVAL2"],
            'CD1_1': self.nasmyth_wcspara["CD1_1"],
            'CD1_2': self.nasmyth_wcspara["CD1_2"],
            'CD2_1': self.nasmyth_wcspara["CD2_1"],
            'CD2_2': self.nasmyth_wcspara["CD2_2"]
        })
    
    def get_hips_images(self):
        """
        Retrieve sky images from HIPS service
        """
        max_retries = 5
        
        # Get overall sky image
        for attempt in range(max_retries):
            try:
                self.hipsimg = np.flipud(hips2fits.query_with_wcs(
                    hips=self.hips_url,
                    wcs=self.wcs,
                    get_query_payload=False,
                    format='jpg',
                    min_cut=0,
                    max_cut=99.9
                ))
                break
            except ReadTimeout:
                print(f"Attempt {attempt + 1} failed, retrying to get overall sky image...")
                time.sleep(2)
        else:
            raise Exception("Failed to get overall sky image after multiple attempts")
        
        # Get IFU sky image
        for attempt in range(max_retries):
            try:
                self.ifu_hips = np.flipud(hips2fits.query_with_wcs(
                    hips=self.hips_url,
                    wcs=self.ifu_wcs,
                    get_query_payload=False,
                    format='jpg',
                    min_cut=0,
                    max_cut=99.9
                ))
                break
            except ReadTimeout:
                print(f"Attempt {attempt + 1} failed, retrying to get IFU sky image...")
                time.sleep(2)
        else:
            raise Exception("Failed to get IFU sky image after multiple attempts")
        
        # Get Guider sky image
        for attempt in range(max_retries):
            try:
                self.guider_hips = np.flipud(hips2fits.query_with_wcs(
                    hips=self.hips_url,
                    wcs=self.guider_wcs,
                    get_query_payload=False,
                    format='jpg',
                    min_cut=0,
                    max_cut=99.9
                ))
                break
            except ReadTimeout:
                print(f"Attempt {attempt + 1} failed, retrying to get Guider sky image...")
                time.sleep(2)
        else:
            raise Exception("Failed to get Guider sky image after multiple attempts")
        
        # Get Nasmyth Guider sky image
        for attempt in range(max_retries):
            try:
                self.nasmyth_hips = np.flipud(hips2fits.query_with_wcs(
                    hips=self.hips_url,
                    wcs=self.nasmyth_wcs,
                    get_query_payload=False,
                    format='jpg',
                    min_cut=0,
                    max_cut=99.9
                ))
                break
            except ReadTimeout:
                print(f"Attempt {attempt + 1} failed, retrying to get Nasmyth Guider sky image...")
                time.sleep(2)
        else:
            raise Exception("Failed to get Nasmyth Guider sky image after multiple attempts")
    
    def create_instrument_polygon(self, wcs, shape):
        """
        Create instrument field of view polygon
        
        Parameters
        ----------
        wcs : astropy.wcs.WCS
            WCS object
        shape : tuple
            Image shape (height, width)
            
        Returns
        ----------
        list
            Polygon vertex coordinates list (celestial coordinates)
        """
        # Create pixel coordinates for the four corners of the image
        corners = np.array([
            [0, 0],
            [shape[1]-1, 0],
            [shape[1]-1, shape[0]-1],
            [0, shape[0]-1]
        ])
        
        # Convert to celestial coordinates
        sky_corners = wcs.pixel_to_world(corners[:, 0], corners[:, 1])
        
        # Return coordinates list
        return np.array([[c.ra.deg, c.dec.deg] for c in sky_corners])
    
    def plot_sky(self):
        """
        Plot overall sky image showing all three instrument fields of view
        """
        fig = plt.figure(figsize=(12, 12))
        ax = plt.subplot(projection=self.wcs)
        
        # Display sky image
        ax.imshow(self.hipsimg, origin="lower")
        
        # Create instrument field of view polygons
        ifu_polygon = self.create_instrument_polygon(self.ifu_wcs, self.ifu_shape)
        guider_polygon = self.create_instrument_polygon(self.guider_wcs, self.guiderimg_shape)
        nasmyth_polygon = self.create_instrument_polygon(self.nasmyth_wcs, self.nasmythimg_shape)
        
        # Plot IFU field of view
        ax.add_patch(MatplotlibPolygon(ifu_polygon, facecolor='none', edgecolor='#8B0000', 
                                      alpha=0.9, transform=ax.get_transform('world')))
        ax.text(ifu_polygon[0][0], ifu_polygon[0][1], 'IFU', color='#8B0000', 
                transform=ax.get_transform('world'), fontsize=14)
        
        # Plot Guider field of view
        ax.add_patch(MatplotlibPolygon(guider_polygon, facecolor='none', edgecolor='#2ECC40', 
                                      alpha=0.9, transform=ax.get_transform('world')))
        ax.text(guider_polygon[0][0], guider_polygon[0][1], 'Guider', color='#2ECC40', 
                transform=ax.get_transform('world'), fontsize=14)
        
        # Plot Nasmyth Guider field of view
        ax.add_patch(MatplotlibPolygon(nasmyth_polygon, facecolor='none', edgecolor='#0074D9', 
                                      alpha=0.9, transform=ax.get_transform('world')))
        ax.text(nasmyth_polygon[0][0], nasmyth_polygon[0][1], 'Nasmyth Guider', color='#0074D9', 
                transform=ax.get_transform('world'), fontsize=14)
        
        # Draw connection line between IFU and Guider
        arrow1 = FancyArrowPatch(
            (self.ra_IFU, self.dec_IFU),
            (self.guider_ra, self.guider_dec),
            transform=ax.get_transform('world'),
            arrowstyle='<->', color='yellow', linewidth=2
        )
        ax.add_patch(arrow1)
        
        # Label distance between IFU and Guider
        mid_ra1 = (self.ra_IFU + self.guider_ra) / 2
        mid_dec1 = (self.dec_IFU + self.guider_dec) / 2
        ax.text(mid_ra1, mid_dec1, f'{self.distance_ifu_guider*60:.1f}′', color='yellow', 
                transform=ax.get_transform('world'), fontsize=12, 
                bbox=dict(facecolor='black', alpha=0.5))
        
        # Draw connection line between IFU and Nasmyth Guider
        arrow2 = FancyArrowPatch(
            (self.ra_IFU, self.dec_IFU),
            (self.nasmyth_ra, self.nasmyth_dec),
            transform=ax.get_transform('world'),
            arrowstyle='<->', color='magenta', linewidth=2
        )
        ax.add_patch(arrow2)
        
        # Label distance between IFU and Nasmyth Guider
        mid_ra2 = (self.ra_IFU + self.nasmyth_ra) / 2
        mid_dec2 = (self.dec_IFU + self.nasmyth_dec) / 2
        ax.text(mid_ra2, mid_dec2, f'{self.distance_ifu_nasmyth*60:.1f}′', color='magenta', 
                transform=ax.get_transform('world'), fontsize=12, 
                bbox=dict(facecolor='black', alpha=0.5))
        
        # Display coordinate information
        ax.text(0.95, 0.95, self.ra_dec_IFU_text, transform=ax.transAxes, 
                fontsize=12, ha='right', va='top', color='white', 
                bbox=dict(facecolor='black', alpha=0.5))
        ax.text(0.95, 0.90, self.ra_dec_Guider_text, transform=ax.transAxes, 
                fontsize=12, ha='right', va='top', color='white', 
                bbox=dict(facecolor='black', alpha=0.5))
        ax.text(0.95, 0.85, self.ra_dec_Nasmyth_text, transform=ax.transAxes, 
                fontsize=12, ha='right', va='top', color='white', 
                bbox=dict(facecolor='black', alpha=0.5))
        
        # Set grid and labels
        ax.grid(color='white', ls='solid', alpha=0.3)
        ax.set_xlabel('RA', fontsize=14)
        ax.set_ylabel('DEC', fontsize=14)
        plt.tight_layout()
        
        # Save image
        if self.save_path is not None:
            savename = os.path.join(self.save_path, "ChiliSky.jpg")
            plt.savefig(savename, dpi=200)
        
        plt.show()
    
    def plot_instrument_sky(self, name):
        """
        Plot individual instrument sky image
        
        Parameters
        ----------
        name : str
            Instrument name ('IFU', 'Guider', or 'Nasmyth')
        """
        if name == 'IFU':
            wcs = self.ifu_wcs
            hips = self.ifu_hips
            title = self.ra_dec_IFU_text
        elif name == 'Guider':
            wcs = self.guider_wcs
            hips = self.guider_hips
            title = self.ra_dec_Guider_text
        elif name == 'Nasmyth':
            wcs = self.nasmyth_wcs
            hips = self.nasmyth_hips
            title = self.ra_dec_Nasmyth_text
        else:
            raise ValueError("Instrument name must be 'IFU', 'Guider', or 'Nasmyth'")
        
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(projection=wcs)
        ax.imshow(hips, origin="lower")
        ax.grid(color='white', ls='solid')
        ax.set_xlabel('RA', fontsize=14)
        ax.set_ylabel('DEC', fontsize=14)
        plt.title(title, fontsize=14, color='white', bbox=dict(facecolor='black', alpha=0.5))
        plt.tight_layout()
        
        if self.save_path is not None:
            savename = os.path.join(self.save_path, f"{name}Sky.jpg")
            plt.savefig(savename, dpi=200)
        
        plt.show()
    
    def plot_all(self):
        """
        Plot all images
        """
        # Plot overall sky image
        self.plot_sky()
        
        # Plot individual instrument sky images
        self.plot_instrument_sky('IFU')
        self.plot_instrument_sky('Guider')
        self.plot_instrument_sky('Nasmyth')
        
        # Output instrument center pointings
    