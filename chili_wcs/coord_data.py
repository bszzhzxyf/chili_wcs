"""
Identifier:     csst_ifs_wcs/coord_data.py
Name:           coord_data.py
Description:    Class for Handling Coordinates data,pixel coordinates and sky coordinates.
Author:         Yifei Xiong
Created:        2023-11-30
Modified-History:
    2023-11-30: add module header
"""

from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy
import astropy.units as units
import astropy.units as u
import numpy as np
import sep
from .load_data import LoadData


class CoordData():
    """
    Class for handling coordinate data.

    pixel coordinates from sep.sky coordinates from Gaia DR3.

    Parameters
    ----------
    data_class : object
        Class from csst_ifs_wcs.load_data.
    rad : float
        The radius for the Gaia DR3 query.
    radec_point : tuple
        Tuple containing the RA and Dec values. If not provided, the RA and Dec
        values are taken from `data_class`.

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

    def __init__(self, data_class: object, rad: float,
                 radec_point: np.ndarray = None):
        self.xy = self.pixel_coord(data_class.image)
        if radec_point is None:
            self.ra_point = data_class.ra
            self.dec_point = data_class.dec
        if radec_point is not None:
            self.ra_point = radec_point[0]
            self.dec_point = radec_point[1]
        self.radec_table = self.gaiadr3_query([self.ra_point],
                                              [self.dec_point], rad)
        self.radec = self.sky_coord(self.radec_table)

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
        bkg = sep.Background(image, bw=10, bh=10, fw=3, fh=3, fthresh=0.0)  # extract background
        image_sub = image - np.median(bkg)  # substract background
        objects = sep.extract(image_sub, thresh=10, err=bkg.globalrms)  # extract stars
        sort = np.argsort(-objects["flux"])
        a = objects['a'][sort]
        b = objects['b'][sort]
        e = a/b
        ecut = (e < 1.3)
        x = objects['x'][sort][ecut]
        y = objects['y'][sort][ecut]
        xycut = (x > 1) & (x < (image.shape[0]-1)) & (y > 1) & (y < (image.shape[1]-1))
        xy = np.column_stack([x, y])[xycut]
        return xy

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

    def sky_coord(self, radec_table: astropy.table.Table):
        """_summary_

        Parameters
        ----------
        radec_table : astropy.table.Table
            Catalog of gaiadr3.

        Returns
        -------
        numpy 2d array
            gaia ra、dec
        """
        radec = np.column_stack([
            np.array(radec_table["RA_ICRS"]),
            np.array(radec_table["DE_ICRS"])
        ])
        return radec
