"""
Identifier:     chili_wcs/load_data.py
Name:           load_data.py
Description:    Load RSS、FGS and relative WCS (IWCS) reference file data.
Author:         Yifei Xiong
Created:        2023-11-30
Modified-History:
    2023-11-30: add module header
"""

import numpy as np
from astropy.io import fits
from .wcs import WCS
from scipy.interpolate import interp1d, griddata
from astropy.wcs import WCS as aWCS
import time
from collections import OrderedDict


class LoadData():
    pass


class LoadRSS(LoadData):
    """
    Class for handling Reduced Spectrum Slicer (RSS) data.

    This class provides methods to process and analyze RSS data.

    Parameters
    ----------
    path_rss : str
        Path to the RSS FITS file.
    path_filter : str, optional
        Path to the filter response curve file. Default is None.
    x0 : float, optional
        X-coordinate of the reference point. Default is 31 / 2.
    y0 : float, optional
        Y-coordinate of the reference point. Default is 31 / 2.

    Attributes
    ----------
    hdu : astropy.io.fits.HDUList
        FITS file header and data units.
    head : astropy.io.fits.Header
        Header of the RSS data.
    flux : numpy.ndarray
        Flux data of the RSS.
    error : numpy.ndarray
        Error data of the RSS.
    mask : numpy.ndarray
        Mask data of the RSS.
    wave : numpy.ndarray
        Wavelength data of the RSS.
    image : numpy.ndarray
        2-D image of the RSS.
    imageflat : numpy.ndarray
        Flattened 1-D RSS image.
    pixelXY : numpy.ndarray
        Flattened pixel coordinates of RSS.
    x0 : float
        X-coordinate of the reference point.
    y0 : float
        Y-coordinate of the reference point.
    centerXY : numpy.ndarray
        Flattened coordinates relative to the reference point.
    wcs : None or astropy.wcs.WCS
        Astropy WCS class.

    Methods
    -------
    RSS_Image()
        Process the RSS data and return a 2-D image and a flattened 1-D image.
    RSS_PixelCoor()
        Generate RSS Pixel Coordinates.
    RSS_ReferPoint(x0, y0)
        Set the reference point coordinates.
    RSS_Cgrid()
        Generate coordinates relative to the reference point.
    Create_Fits(newpath)
        Create a new FITS file with WCS information.
    """

    def __init__(self, path_rss: str, path_filter: str = None, x0: float = 31 / 2, y0: float = 31 / 2):
        # Data
        self.path = path_rss
        self.path_filter = path_filter
        self.hdu = fits.open(self.path)
        self.head = self.hdu[1].header
        self.oldname = self.hdu[0].header["FILENAME"]
        self.newname = self.oldname.split(r".")[0].strip("_") + "g" + ".fits"
        self.flux = self.hdu[1].data
        self.error = self.hdu[2].data
        self.mask = self.hdu[3].data
        # self.flux_masked = self.hdu[4].data
        self.wave = self.head['CRVAL3'] + np.arange(
            0., self.head['CD3_3'] * (self.head['NAXIS3']), self.head['CD3_3'])
        # Image
        self.image = self.RSS_Image()[0]  # 2-D image of RSS
        self.imageflat = self.RSS_Image()[1]  # Flattened 1-D RSS image
        # Pixel Coordinates
        self.pixelXY = self.RSS_PixelCoor()
        self.x0 = self.RSS_ReferPoint(x0, y0)[0]  # refer point of rss
        self.y0 = self.RSS_ReferPoint(x0, y0)[1]  # refer point of rss
        self.centerXY = self.RSS_Cgrid()  # [ND array] refer grid of rss
        self.wcs = None  # astropy wcs class

    def RSS_Image(self):
        # Convolve filter response
        # 1.load response curve
        if self.path_filter is not None:
            filt = np.loadtxt(self.path_filter)
            wave_filt = filt[:, 0]
            respons = filt[:, 1]

            # 2.interpolate response curve into wave_rss
            Interp_re = interp1d(wave_filt,
                                 respons,
                                 kind='linear',
                                 fill_value='extrapolate')
            range_filt = (self.wave > wave_filt[0]) & (self.wave < 9700)
            respons_interped = Interp_re(
                self.wave[range_filt]
            )  # response curve interpolated into wave_rss
            # 3.convolved with filter, make image#
            image_rss = np.ones([32, 32])
            for i in range(32):
                for j in range(32):
                    good = self.mask[:, i,
                                     j] == 0  # choose good locations(where mask==0)
                    wave_int = self.wave[good]  # wave at good locations
                    flux_int = self.flux[:, i,
                                         j][good]  # flux at good locations
                    Interp = interp1d(wave_int,
                                      flux_int,
                                      kind='nearest',
                                      fill_value='extrapolate')
                    flux_rss_i = Interp(
                        self.wave)  # interpolated flux at wave_rss
                    image_rss[i, j] = np.median(
                        (flux_rss_i[range_filt] * respons_interped))
        if self.path_filter is None:
            flux_int = self.flux.copy()
            flux_int[self.mask == 1] = np.nan
            image_rss = np.nanmedian(flux_int, axis=0)
        image_rss_flat = image_rss.flatten()
        return image_rss, image_rss_flat

    def RSS_PixelCoor(self):
        # RSS Pixel Coordinates
        x_rss = np.arange(0, 32, 1)
        y_rss = np.arange(0, 32, 1)
        X_rss, Y_rss = np.meshgrid(x_rss, y_rss)
        Xf_rss = X_rss.flatten()
        Yf_rss = Y_rss.flatten()
        XYf_rss = np.vstack((Xf_rss, Yf_rss))
        return XYf_rss

    def RSS_ReferPoint(self, x0, y0):
        x0r = x0
        y0r = y0
        return x0r, y0r

    def RSS_Cgrid(self):
        # Coordinates relative to reference point
        nx = self.image.shape[0]
        ny = self.image.shape[1]
        x_rss = np.arange(0, nx, 1)
        y_rss = np.arange(0, ny, 1)
        X_rss, Y_rss = np.meshgrid(x_rss, y_rss)  # X means coordinates gird
        Xf_rss = X_rss.flatten()
        Yf_rss = Y_rss.flatten()
        Xfc_rss = (Xf_rss - self.x0)
        Yfc_rss = (Yf_rss - self.y0)
        XYfc_rss = np.vstack((Xfc_rss, Yfc_rss))
        return XYfc_rss

    def Create_Fits(self, newpath):
        self.createtime = f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(time.time()))}.{int((time.time()% 1) * 10)}"
        WCS_List = [
            ("WCSCAL_S", 1, "WCS calibration completed?"),
            ("WCS_VER", "A0101", "version of WCS calibration code"),
            ("WCSCAL_P", "csst_ifs_wcs.toml", "WCS calibration configuration file name"),
            ("WCS_TLM", self.createtime, "last WCS calibration time"),
            ("EQUINOX", 2000.0, "epoch of the mean equator and equinox"),
            ("RADESYS", "ICRS", "frame of reference of coordinates"),
            ("CUNIT1", "deg", "units of coordinate increment and value"),
            ("CUNIT2", "deg", "units of coordinate increment and value"),
            ("CTYPE1", "RA---TAN", "right ascension, gnomonic projection"),
            ("CTYPE2", "DEC--TAN", "declination, gnomonic projection"),
            ("CRPIX1", self.wcs.CRPIX[0], "pixel coordinate of reference point"),
            ("CRPIX2", self.wcs.CRPIX[1], "pixel coordinate of reference point"),
            ("CRVAL1", self.wcs.CRVAL[0], "RA of reference point"),
            ("CRVAL2 ", self.wcs.CRVAL[1], "DEC of reference point"),
            ("CD1_1", self.wcs.CD[0, 0], "coordinate transformation matrix element"),
            ("CD1_2", self.wcs.CD[0, 1], "coordinate transformation matrix element"),
            ("CD2_1", self.wcs.CD[1, 0], "coordinate transformation matrix element"),
            ("CD2_2", self.wcs.CD[1, 1], "coordinate transformation matrix element"),
            ("CTYPE3", "AWAV", "wavelength direction"),
            ("CUNIT3", "Angstrom", "unit of wavelength")]
        """
            ("IGUIFILE",
             "CSST_IFS_slicer_IGUI_ASTM_yyyymmddHHMMSS _obsid _L1_VER_process.fit",
             "Image data file of the guide camera accompanying observation"),
            ("GGUIFILE",
             "CSST_IFS_slicer_GGUI_ASTM_yyyymmddHHMMSS _obsid _L1_VER_process.fit",
             "Geometric data file of the guide camera accompanying observation"),
            ("IMCIFILE",
             "CSST_IFS_slicer_IMCI_ASTM_yyyymmddHHMMSS _obsid _L1_VER_process.fit",
             "Image data file of the MCI  accompanying observation"),
            ("GMCIFILE",
             "CSST_IFS_slicer_GMCI_ASTM_yyyymmddHHMMSS _obsid _L1_VER_process.fit",
             "Geometric data file of the MCI accompanying observation")
        """
        self.hdu[1].header.extend(WCS_List, unique=True)
        self.hdu[0].header.set("FILENAME", self.newname)
        self.hdu[1].header.set("PROCESS", 'bewnrfcg')
        New_FitsName = newpath + self.newname
        self.hdu.writeto(New_FitsName, overwrite=True)


class LoadFGS(LoadData):
    """
    Class for handling Guider image data.

    This class provides methods to process and analyze Guider image data.

    Parameters
    ----------
    path_gui : str
        Path to the Guider FITS file.

    Attributes
    ----------
    hdu : astropy.io.fits.HDUList
        FITS file header and data units.
    image : numpy.ndarray
        Guider image data.
    head : astropy.io.fits.Header
        Header of the Guider data.
    mwcspara : collections.OrderedDict
        Ordered dictionary containing MWCS parameters.
    """

    def __init__(self, path_gui: str):
        # Data
        self.path = path_gui
        self.hdu = fits.open(self.path)
        self.image = self.hdu[1].data.byteswap().newbyteorder()
        self.head = self.hdu[1].header
        self.mwcspara = OrderedDict({
            "MCRPIX1":
            self.head["CRPIX1"],
            "MCRPIX2":
            self.head["CRPIX2"],
            "MCRVAL1":
            self.head["CRVAL1"],
            "MCRVAL2":
            self.head["CRVAL2"],
            "MCD1_1":
            -np.sqrt(self.head["CD1_1"]**2 + self.head["CD2_1"]**2),
            "MCD2_2":
            np.sqrt(self.head["CD1_2"]**2 + self.head["CD2_2"]**2),
            "MLONPOLE":
            180 + np.arctan2(self.head["CD2_1"], self.head["CD2_2"])
        })


class LoadIWCS(LoadData):
    """
    Class for handling relative WCS (IWCS) reference file data.

    This class provides methods to process and analyze IWCS reference file data.

    Parameters
    ----------
    path_iwcs_ref : str
        Path to the IWCS FITS reference file.

    Attributes
    ----------
    hdu : astropy.io.fits.HDUList
        FITS file header and data units.
    data : numpy.ndarray
        IWCS data.
    head : astropy.io.fits.Header
        Header of the IWCS data.
    iwcspara : collections.OrderedDict
        Ordered dictionary containing IWCS parameters.
    """
    def __init__(self, path_iwcs_ref: str):
        # Data
        self.path = path_iwcs_ref
        self.hdu = fits.open(self.path)
        self.data = self.hdu[1].data
        self.head = self.hdu[1].header
        self.iwcspara = OrderedDict({
            "ICRPIX1": self.data["ICRPIX1"][0],
            "ICRPIX2": self.data["ICRPIX2"][0],
            "ICRVAL1": self.data["ICRVAL1"][0],
            "ICRVAL2": self.data["ICRVAL2"][0],
            "ICD1_1": self.data["ICD1_1"][0],
            "ICD2_2": self.data["ICD2_2"][0],
            "ILONPOLE": self.data["ILONPOLE"][0]
        })
