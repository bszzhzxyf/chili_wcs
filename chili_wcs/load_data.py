"""
Identifier:     chili_wcs/load_data.py
Name:           load_data.py
Description:    Load RSS、Guider and relative WCS (IWCS) reference file data.
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
import pandas as pd
import matplotlib.pyplot as plt
import os


class LoadData():
    pass


class LoadRSS(LoadData):
    """
    The LoadRSS class is used for processing RSS data.

    Parameters
    ----------
    path_rss : str
        Path to the RSS FITS file.
    plot_map : bool, optional
        Whether to plot the fiber position map, default is False.
    save_path : str, optional
        Path to save the plotted map, default is None.

    Attributes
    ----------
    path_rss : str
        Path to the RSS FITS file.
    hdu : astropy.io.fits.HDUList
        FITS file header and data units.
    rss : numpy.ndarray
        RSS data.
    save_path : str
        Path to save the plotted map.
    x_fiber : numpy.ndarray
        X coordinates of the fibers.
    y_fiber : numpy.ndarray
        Y coordinates of the fibers.
    sum_fluxes : numpy.ndarray
        Total flux for each fiber.

    Methods
    ----------
    get_fiber_coord()
        Get the offset coordinates of the fibers.
    sum_flux()
        Calculate the total flux for each fiber.
    to_2dmap()
        Plot the fiber data into a 2D map.
    """
    def __init__(self, path_rss: str, plot_map: bool = False, save_path: str = None):
        self.path = path_rss
        self.filename = os.path.basename(self.path)
        self.hdu = fits.open(self.path)
        self.rss = self.hdu[0].data
        self.header = self.hdu[0].header
        self.save_path = save_path
        ## maps：
        self.x_fiber, self.y_fiber = self.get_fiber_coord()
        self.xy_fiber = np.column_stack((self.x_fiber[:, np.newaxis], self.y_fiber[:, np.newaxis]))
        self.sum_fluxes = self.sum_flux()
        ## plot map
        if plot_map:
            self.to_2dmap()   
    def get_fiber_coord(self):
        n = 494
        # 竖直方向到边的距离为0.5， 则六边形边长为0.5/np.cos(np.pi/6) 
        delta_x = np.cos(np.pi/6) # 1.5 * 0.5/np.cos(np.pi/6)
        xoff =  xoff = np.arange(11, -12, -1) * delta_x
        ixoff = np.zeros(shape=n, dtype=float)
        iyoff = np.zeros(shape=n, dtype=float)
        for i in range(n):
        # for i in range(43, 64):
            a = i // 43     # 整除
            b = i % 43      # 余数        
            if b >= 21:
                a = a*2 + 1
            else:
                a = a*2
            ixoff[i] = xoff[a]
            
            if a % 2 == 0:      # 偶数列，第0,2,4列
                iyoff[i] = b - 10
            else:                   # 奇数列，第1,3,5列
                iyoff[i] = b - 21 - 10.5

            if i >= 236 and i <= 246:
                iyoff[i] = iyoff[i] + 11
            if i >= 247 and i <= 257:
                iyoff[i] = iyoff[i] - 11
                

        return ixoff, iyoff

    def sum_flux(self):
        
        nspec = 494
        pos = np.zeros(shape=nspec, dtype=float)
        sum_fluxes = np.zeros(shape=nspec, dtype=float)
        for i in range(0, 494):
            pos[i] = i
            sum_fluxes[i] = np.nanmean(self.rss[i, 1020:1300])
        df = pd.DataFrame({'pos': pos, 'sum_flux': sum_fluxes, 'x_fiber': self.x_fiber, 'y_fiber': self.y_fiber})
        if self.save_path is not None:
            save_file = os.path.join(self.save_path, 'sum_flux_map.dat')
            df.to_csv(save_file, index=False)
        return sum_fluxes

    def to_2dmap(self):
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        
        import matplotlib.cm as cm
        from matplotlib.patches import RegularPolygon
        from astropy.visualization import AsinhStretch
        from astropy.visualization.mpl_normalize import ImageNormalize

        cmap = cm.get_cmap('gray')
        stretch = AsinhStretch(0.01)
        norm = ImageNormalize(vmin=min(self.sum_fluxes), vmax=max(self.sum_fluxes), stretch=stretch)
        zcolor = cmap(norm(self.sum_fluxes))
        
        for i in range(len(self.x_fiber)):
            hexagon = RegularPolygon((self.x_fiber[i], self.y_fiber[i]), 
                                    numVertices=6, radius=0.5/np.cos(np.pi/6), 
                                    orientation=np.pi/2, color=zcolor[i])
            ax.add_patch(hexagon)
        plt.autoscale(enable = True)
        if self.save_path is not None:
            save_file = os.path.join(self.save_path, 'sum_flux_map.png')
            plt.savefig(save_file)

    def Create_Fits(self,save_path: str , wcspara = None , gwcspara = None, iwcspara = None):
        self.createtime = f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(time.time()))}.{int((time.time()% 1) * 10)}"
        WCS_List = [
            ("WCS_TLM", self.createtime, "last WCS calibration time"),
            ("EQUINOX", 2015.0, "epoch of the mean equator and equinox"),
            ("RADESYS", "ICRS", "frame of reference of coordinates"),
            ("CUNIT1", "deg", "units of coordinate increment and value"),
            ("CUNIT2", "deg", "units of coordinate increment and value"),
            ("CTYPE1", "RA---TAN", "right ascension, gnomonic projection"),
            ("CTYPE2", "DEC--TAN", "declination, gnomonic projection"),
            ("CRPIX1", wcspara["CRPIX1"], "pixel coordinate of reference point"),
            ("CRPIX2", wcspara["CRPIX2"], "pixel coordinate of reference point"),
            ("CRVAL1", wcspara["CRVAL1"], "RA of reference point"),
            ("CRVAL2", wcspara["CRVAL2"], "DEC of reference point"),
            ("CD1_1", wcspara["CD1_1"], "coordinate transformation matrix element"),
            ("CD1_2", wcspara["CD1_2"], "coordinate transformation matrix element"),
            ("CD2_1", wcspara["CD2_1"], "coordinate transformation matrix element"),
            ("CD2_2", wcspara["CD2_2"], "coordinate transformation matrix element")
        ]
        
        # calculate RA and DEC of each fiber
        mi_wcs = WCS(gwcspara=gwcspara, iwcspara=iwcspara)
        iradec_fiber = mi_wcs.ifs_xy2sky(self.xy_fiber)
        self.ra_fiber = iradec_fiber[:, 0]
        self.dec_fiber = iradec_fiber[:, 1]

        # create fits file
        fiber_pos = np.arange(0, 494, 1)
        col1 = fits.Column(name='FIBER_POS', format='K', array=fiber_pos)
        col2 = fits.Column(name='SUM_FLUXES', format='D', array=self.sum_fluxes)
        col3 = fits.Column(name='X_FIBER', format='D', array=self.x_fiber)
        col4 = fits.Column(name='Y_FIBER', format='D', array=self.y_fiber)
        col5 = fits.Column(name='RA_FIBER', format='D', array=self.ra_fiber)
        col6 = fits.Column(name='DEC_FIBER', format='D', array=self.dec_fiber)
        hdu2 = fits.BinTableHDU.from_columns([col1, col2, col3, col4, col5, col6], name='RADEC')
        
        hdu1 = fits.ImageHDU(data=self.rss, name='RSS')
        
        # 将WCS_List写入头文件，包含comment
        for key, value, comment in WCS_List:
            hdu1.header[key] = (value, comment)
            hdu2.header[key] = (value, comment)
        
        hdu0 = fits.PrimaryHDU()
        hdul = fits.HDUList([hdu0, hdu1, hdu2])
        filename_radec = os.path.join(save_path, self.filename.replace(".fits", "_radec.fits"))
        hdul.writeto(filename_radec, overwrite=True)
        print("\nData has been saved to:", filename_radec)


class LoadGuider(LoadData):
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
        self.filename = self.path.split("/")[-1]
        self.hdu = fits.open(self.path)
        self.img = self.hdu[0].data.astype(np.float32)  # 将数据转换为float32类型以支持sep处理
        self.header = self.hdu[0].header
        try:
            self.gwcspara = OrderedDict({
            "GCRPIX1":
            self.header["CRPIX1"],
            "GCRPIX2":
            self.header["CRPIX2"],
            "GCRVAL1":
            self.header["CRVAL1"],
            "GCRVAL2":
            self.header["CRVAL2"],
            "GCD1_1":
            -np.sqrt(self.header["CD1_1"]**2 + self.header["CD2_1"]**2),
            "GCD2_2":
            np.sqrt(self.header["CD1_2"]**2 + self.header["CD2_2"]**2),
            "GLONPOLE":
            180 + np.arctan2(self.header["CD2_1"], self.header["CD2_2"])
            })
        except:
            pass
    @staticmethod
    def Create_Fits(self, newpath: str , mode = "Normal", wcspara = None):
        self.createtime = f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(time.time()))}.{int((time.time()% 1) * 10)}"
        if mode == "Normal":
            WCS_List = [
            ("WCS_TLM", self.createtime, "last WCS calibration time"),
            ("EQUINOX", 2015.0, "epoch of the mean equator and equinox"),
            ("RADESYS", "ICRS", "frame of reference of coordinates"),
            ("CUNIT1", "deg", "units of coordinate increment and value"),
            ("CUNIT2", "deg", "units of coordinate increment and value"),
            ("CTYPE1", "RA---TAN", "right ascension, gnomonic projection"),
            ("CTYPE2", "DEC--TAN", "declination, gnomonic projection"),
            ("CRPIX1", wcspara["CRPIX1"], "pixel coordinate of reference point"),
            ("CRPIX2", wcspara["CRPIX2"], "pixel coordinate of reference point"),
            ("CRVAL1", wcspara["CRVAL1"], "RA of reference point"),
            ("CRVAL2 ", wcspara["CRVAL2"], "DEC of reference point"),
            ("CD1_1", wcspara["CD1_1"], "coordinate transformation matrix element"),
            ("CD1_2", wcspara["CD1_2"], "coordinate transformation matrix element"),
            ("CD2_1", wcspara["CD2_1"], "coordinate transformation matrix element"),
                ("CD2_2", wcspara["CD2_2"], "coordinate transformation matrix element")]
            self.newname = self.filename.split(".")[0] + "_with_wcs.fits"
        if mode == "Guider":
            WCS_List = [
            ("WCS_TLM", self.createtime, "last WCS calibration time"),
            ("EQUINOX", 2015.0, "epoch of the mean equator and equinox"),
            ("RADESYS", "ICRS", "frame of reference of coordinates"),
            ("CUNIT1", "deg", "units of coordinate increment and value"),
            ("CUNIT2", "deg", "units of coordinate increment and value"),
            ("CTYPE1", "RA---TAN", "right ascension, gnomonic projection"),
            ("CTYPE2", "DEC--TAN", "declination, gnomonic projection"),
            ("GCRPIX1", wcspara["GCRPIX1"], "pixel coordinate of reference point"),
            ("GCRPIX2", wcspara["CRPIX2"], "pixel coordinate of reference point"),
            ("GCRVAL1", wcspara["GCRVAL1"], "RA of reference point"),
            ("GCRVAL2 ", wcspara["GCRVAL2"], "DEC of reference point"),
            ("GCD1_1", wcspara["GCD1_1"], "coordinate transformation matrix element"),
            ("GCD2_2", wcspara["GCD2_2"], "coordinate transformation matrix element"),
            ("GLONPOLE", wcspara["GLONPOLE"], "longitude of the pole")]
            self.newname = self.filename.split(".")[0] + "_with_gwcs.fits"
        self.hdu[0].header.extend(WCS_List, unique=True)
        New_FitsName = os.path.join(newpath, self.newname)
        self.hdu.writeto(New_FitsName, overwrite=True)



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
        # 获取导星相机的IWCS参数
        self.guider_data = self.hdu[1].data
        self.guider_head = self.hdu[1].header
        # 获取IFU的IWCS参数
        self.guider_nasmyth_data = self.hdu[2].data
        self.guider_nasmyth_head = self.hdu[2].header
        
        # 导星相机IWCS参数
        self.guider_iwcspara = OrderedDict({
            "ICRPIX1": self.guider_data["ICRPIX1"][0],
            "ICRPIX2": self.guider_data["ICRPIX2"][0],
            "ICRVAL1": self.guider_data["ICRVAL1"][0],
            "ICRVAL2": self.guider_data["ICRVAL2"][0],
            "ICD1_1": self.guider_data["ICD1_1"][0],
            "ICD2_2": self.guider_data["ICD2_2"][0],
            "ILONPOLE": self.guider_data["ILONPOLE"][0]
        })
        
        # Guider Nasmyth焦点IWCS参数
        self.guider_nasmyth_iwcspara = OrderedDict({
            "ICRPIX1": self.guider_nasmyth_data["ICRPIX1"][0],
            "ICRPIX2": self.guider_nasmyth_data["ICRPIX2"][0],
            "ICRVAL1": self.guider_nasmyth_data["ICRVAL1"][0],
            "ICRVAL2": self.guider_nasmyth_data["ICRVAL2"][0],
            "ICD1_1": self.guider_nasmyth_data["ICD1_1"][0],
            "ICD2_2": self.guider_nasmyth_data["ICD2_2"][0],
            "ILONPOLE": self.guider_nasmyth_data["ILONPOLE"][0]
        })
        
    @staticmethod
    def Create_Fits(save_path: str, iwcspara = None, guider_nasmyth_iwcspara = None):
        createtime = f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(time.time()))}.{int((time.time()% 1) * 10)}"
        current_date = time.strftime('%Y%m%d', time.gmtime(time.time()))
        
        # 创建导星相机IWCS HDU
        guider_header = [
            ("WCSRTIME", createtime, "Create Time for the reference file"),
            ("WCSRTYPE", "GUIDER", "relative WCS parameter for guider camera")
        ]
        
        # 如果只有iwcspara，则hdu1保存数据，hdu2数据为0
        if iwcspara is not None and guider_nasmyth_iwcspara is None:
            guider_hdu = fits.BinTableHDU.from_columns(
                [fits.Column(name=k, array=[v], format='D') for k, v in iwcspara.items()],
                header=fits.Header(guider_header),
                name='GUIDER_IWCS'
            )
            
            # 创建全为0的Guider Nasmyth焦点IWCS HDU
            guider_nasmyth_header = [
                ("WCSRTIME", createtime, "Create Time for the reference file"),
                ("WCSRTYPE", "GUIDER_NASMYTH", "relative WCS parameter for Guider Nasmyth focus")
            ]
            zero_data = {k: 0.0 for k in iwcspara.keys()}
            guider_nasmyth_hdu = fits.BinTableHDU.from_columns(
                [fits.Column(name=k, array=[v], format='D') for k, v in zero_data.items()],
                header=fits.Header(guider_nasmyth_header),
                name='GUIDER_NASMYTH_IWCS'
            )
        # 如果只有guider_nasmyth_iwcspara，则hdu2保存数据，hdu1数据为0
        elif guider_nasmyth_iwcspara is not None and iwcspara is None:
            # 创建全为0的导星相机IWCS HDU
            zero_data = {k: 0.0 for k in guider_nasmyth_iwcspara.keys()}
            guider_hdu = fits.BinTableHDU.from_columns(
                [fits.Column(name=k, array=[v], format='D') for k, v in zero_data.items()],
                header=fits.Header(guider_header),
                name='GUIDER_IWCS'
            )
            
            # 创建Guider Nasmyth焦点IWCS HDU
            guider_nasmyth_header = [
                ("WCSRTIME", createtime, "Create Time for the reference file"),
                ("WCSRTYPE", "GUIDER_NASMYTH", "relative WCS parameter for Guider Nasmyth focus")
            ]
            guider_nasmyth_hdu = fits.BinTableHDU.from_columns(
                [fits.Column(name=k, array=[v], format='D') for k, v in guider_nasmyth_iwcspara.items()],
                header=fits.Header(guider_nasmyth_header),
                name='GUIDER_NASMYTH_IWCS'
            )
        # 如果两者都有，则分别保存
        else:
            guider_hdu = fits.BinTableHDU.from_columns(
                [fits.Column(name=k, array=[v], format='D') for k, v in (iwcspara or {}).items()],
                header=fits.Header(guider_header),
                name='GUIDER_IWCS'
            )
            
            guider_nasmyth_header = [
                ("WCSRTIME", createtime, "Create Time for the reference file"),
                ("WCSRTYPE", "GUIDER_NASMYTH", "relative WCS parameter for Guider Nasmyth focus")
            ]
            guider_nasmyth_hdu = fits.BinTableHDU.from_columns(
                [fits.Column(name=k, array=[v], format='D') for k, v in (guider_nasmyth_iwcspara or {}).items()],
                header=fits.Header(guider_nasmyth_header),
                name='GUIDER_NASMYTH_IWCS'
            )
        
        # 创建FITS文件
        hdul = fits.HDUList([fits.PrimaryHDU(), guider_hdu, guider_nasmyth_hdu])
        
        # 判断save_path是文件夹路径还是文件路径
        if save_path.endswith('.fits'):
            # 如果save_path已经包含文件名，直接使用
            ref_file = save_path
        else:
            # 如果save_path是文件夹路径，生成默认文件名
            ref_file = os.path.join(save_path, f"IWCS_{current_date}.fits")
            
        hdul.writeto(ref_file, overwrite=True)
        print("\n IWCS reference file has been saved to:", ref_file)