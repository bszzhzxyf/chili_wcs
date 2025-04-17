"""
Identifier:     chili_wcs/wcs_solver.py
Name:           wcs_solver.py
Description:    Solve the wcs from the image and save the fits file.
Author:         Yifei Xiong
Created:        2024-11-21
Last Modified:  2025-04-17
Modified-History:
    2025-04-17:  change the algorithm of IFU WCS solver.
"""
import numpy as np
from collections import OrderedDict
from .load_data import LoadRSS,LoadGuider,LoadIWCS  # Load RSS module
from .coord_data import CoordData  # Star pixel coordinates and celestial coordinates
from .fit_wcs import TriMatch, FitParam  # Triangle matching and parameter fitting
from .wcs import WCS  # WCS plate model
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt

class WCSSolver:
    @staticmethod
    def normal_solver(img: np.ndarray = None, 
                 ra_guess: float = None,
                 dec_guess: float  = None, 
                 search_radius: float = None,
                 n_pixs: int = 8,
                 n_stars: int = 8,
                 match_radius: float = 0.05,
                 target_err: float = 0.2,
                 min_data_points: int = 3,
                 plot: bool = False
                 ):
        """
        Solve WCS parameters for a single image using normal method.

        Parameters
        ----------
        img : ndarray, shape (M, N)
            Input image array
        ra_guess : float
            Initial guess for right ascension in degrees
        dec_guess : float  
            Initial guess for declination in degrees
        search_radius : float
            Search radius for star matching in arcsec
        n_pixs : int, optional
            Maximum number of star pixel positions to use, default 8
        n_stars : int, optional
            Maximum number of catalog stars to use, default 8
        match_radius : float, optional
            Radius for triangle matching, default 0.05
        target_err : float, optional
            Target error threshold in arcsec, default 0.2
        min_data_points : int, optional
            Minimum number of matched points required, default 3
        plot : bool, optional
            Whether to plot results, default False

        Returns
        -------
        dict
            Dictionary containing fitted WCS parameters
        """
        print("Step 1: Starting star centering and catalog preparation...")
        coord = CoordData(img, mode="Normal",rad = search_radius ,radec_point = [ra_guess ,dec_guess],plot_result=True)
        xy = coord.xy[:n_pixs] # pixel_coordinates
        radec = coord.radec[:n_stars]
        print(f"Found {len(xy)} star positions")
        print(f"Pixel coordinates:\n{xy}")
        print(f"Sky coordinates:\n{radec}")
        
        print("\nStep 2: Starting triangle matching...")
        trimatch = TriMatch(xy,radec,match_radius=match_radius)
        matches = trimatch.matches_array    
        print(f"Found {len(matches)} matched triangles")
        
        print("\nStep 3: Starting WCS fitting...")
        fixedpara = OrderedDict({"CRPIX":(np.array(img.shape) + 1 )/2}) # Fixed parameters
        p0 = [ra_guess,dec_guess,0,0,0,0] # Initial parameters
        print("Initial fitting parameters:", p0)
        fit = FitParam(xy, radec, matches,method = "Normal",fixedpara = fixedpara,
                inipara=p0, target_err = target_err, min_data_points = min_data_points)
        print("Fitted WCS parameters:")
        for key, value in fit.wcs_fitted.items():
            print(f"{key}: {value}")
        
        print("\nStep 4: Testing fit quality...")
        # Get fitted coordinates
        xy2 = fit.coords[0]
        wcs_fit = WCS(wcspara = fit.wcs_fitted)
        radec_fit = wcs_fit.xy2sky(xy2)
        # Get true coordinates 
        radec_true = fit.coords[1]
        WCSSolver.residual(radec_fit, radec_true, plot = plot)
        print("WCS fitting completed!")
        return fit.wcs_fitted

    @staticmethod
    def guider_solver(guider_path: str,
                     ra_guess_guider: float,
                     dec_guess_guider: float,
                     n_pixs_guider: int = 8,
                     n_stars_guider: int = 15,
                     match_radius_guider: float = 0.05,
                     target_err_guider: float = 0.1,
                     min_data_points_guider: int = 3,
                     plot: bool = False):
        """
        Solve WCS solution for guider camera.

        Parameters
        ----------
        guider_path : str
            Path to guider camera FITS file
        ra_guess_guider : float
            Initial RA guess for guider field center [deg]
        dec_guess_guider : float
            Initial Dec guess for guider field center [deg]
        n_pixs_guider : int, optional
            Number of pixel coordinates to use for guider matching, default 8
        n_stars_guider : int, optional
            Number of catalog stars to use for guider matching, default 15
        match_radius_guider : float, optional
            Radius for triangle matching, default 0.05
        target_err_guider : float, optional
            Target error threshold for guider solution [arcsec], default 0.1
        min_data_points_guider : int, optional
            Minimum number of matched points required for guider, default 3
        plot : bool, optional
            Whether to plot results, default False

        Returns
        -------
        dict
            Dictionary containing fitted WCS parameters
        """
        print("\nStarting to solve guider WCS...")
        # 1-1: load the guider image
        print("1-1: Loading guider image")
        guider = LoadGuider(guider_path)
        guiderimg = guider.img
        
        # 1-2: get the pixel coordinates and ra dec of the stars
        print("1-2: Getting star pixel coordinates and sky coordinates")
        gui_coord = CoordData(guider, mode="Guider", radec_point=[ra_guess_guider, dec_guess_guider], plot_result=plot)
        gui_xy = gui_coord.xy[:n_pixs_guider] # pixel_coordinates
        gui_radec = gui_coord.radec[:n_stars_guider]
        print(f"Found {len(gui_coord.xy)} stars in the image")
        print(f"Found {len(gui_coord.radec)} catalog stars in the field")
        
        # 1-3: triangle matching
        print("1-3: Triangle matching")
        trimatch = TriMatch(gui_xy, gui_radec, match_radius=match_radius_guider)
        matches = trimatch.matches_array
        print(f"Number of matched triangles: {len(matches)}")
        
        # 1-4: WCS fitting
        print("1-4: WCS fitting")
        gfixedpara = OrderedDict({"GCRPIX":np.array(((guiderimg.shape[1]+1)/2, (guiderimg.shape[0]+1)/2))})
        gp0 = [ra_guess_guider, dec_guess_guider, 0, 0, 180]
        print(f"Initial fitting parameters: {gp0}")
        gfit = FitParam(gui_xy, gui_radec, matches, method="Guider", fixedpara=gfixedpara,
                inipara=gp0, target_err=target_err_guider, min_data_points=min_data_points_guider)
        print(f"Fitted WCS parameters:\n{gfit.gwcs_fitted}")
        
        # Check fit quality
        # Get fitted coordinates
        xy2 = gfit.coords[0]
        gwcs_fit = WCS(gwcspara=gfit.gwcs_fitted)
        radec_fit = gwcs_fit.guider_xy2sky(xy2)
        # Get true coordinates
        radec_true = gfit.coords[1]
        # Calculate and plot residuals
        WCSSolver.residual(radec_fit, radec_true, plot=plot)

        return gfit.gwcs_fitted
    
    @staticmethod
    def iwcs_solver(ifu_path: str = None,
                  ra_guess_ifu: float = None,  # Initial RA guess for IFU, in degrees
                  dec_guess_ifu: float = None,  # Initial Dec guess for IFU, in degrees 
                  n_pixs_ifu: int = 7,  # Number of pixel coordinates to use for IFU
                  n_stars_ifu: int = 15,  # Number of catalog stars to use for IFU
                  backparam: dict = None,  # Background fitting parameters for IFU centering
                  match_radius_ifu: float = 0.02,  # Triangle matching radius for IFU
                  target_err_ifu: float = 0.1,  # Target error threshold for IFU, in arcsec
                  min_data_points_ifu: int = 3,  # Minimum matched points required for IFU
                  gwcs_fitted: OrderedDict = None,  # Guider camera WCS parameters
                  mode: str = "guider",  # Mode for IFU WCS solving: "guider" or "nasmyth"
                  plot: bool = False):
        """
        Solve WCS parameters for IFU.

        Parameters
        ----------
        ifu_path : str
            Path to IFU image file
        ra_guess_ifu : float
            Initial RA guess for IFU field center, in degrees
        dec_guess_ifu : float
            Initial Dec guess for IFU field center, in degrees
        n_pixs_ifu : int, optional
            Number of pixel coordinates to use for IFU matching, default 7
        n_stars_ifu : int, optional
            Number of catalog stars to use for IFU matching, default 15
        backparam : dict, optional
            Background fitting parameters for IFU star centering
        match_radius_ifu : float, optional
            Triangle matching radius for IFU, default 0.02
        target_err_ifu : float, optional
            Target error threshold for IFU solution, in arcsec, default 0.1
        min_data_points_ifu : int, optional
            Minimum number of matched points required for IFU, default 3
        mode : str, optional
            Mode for IFU WCS solving, either "guider" or "nasmyth", default "guider"
        plot : bool, optional
            Whether to plot results, default False

        Returns
        -------
        dict
            Dictionary containing fitted WCS parameters
        """
        print("\nStarting to solve IFU WCS...")
        # 2-1: load the ifu image
        print("2-1: Loading IFU image")
        rss = LoadRSS(ifu_path)
        
        # 2-2: get the pixel coordinates and ra dec of the stars
        print("2-2: Getting star pixel coordinates and sky coordinates")
        ifu_coord = CoordData(rss, mode="IFU", radec_point=[ra_guess_ifu, dec_guess_ifu], backparam=backparam, plot_result=plot)
        ifu_xy = ifu_coord.xy[:n_pixs_ifu] # pixel_coordinates
        ifu_radec = ifu_coord.radec[:n_stars_ifu]
        print(f"Found {len(ifu_coord.xy)} stars in the image")
        print(f"Found {len(ifu_coord.radec)} catalog stars in the field")
        
        # 2-3: triangle matching：
        print("2-3: Triangle matching")
        trimatch=TriMatch(ifu_xy,ifu_radec,match_radius=match_radius_ifu)
        matches=trimatch.matches_array
        print(f"Number of matched triangles: {len(matches)}")
        
        # 2-4: WCS fitting:
        print("2-4: Intermediate WCS fitting")
        ifixedpara = OrderedDict({"ICRPIX":np.array([1,1]),
                                "GCRVAL":np.array([gwcs_fitted["GCRVAL1"],gwcs_fitted["GCRVAL2"]]),
                                "GLONPOLE":gwcs_fitted["GLONPOLE"]}) # fixed parameters
        
        # 根据模式选择初始参数
        if mode.lower() == "nasmyth":
            ip0 = [305.4, 89.787, 0, 0, 144.6]  # nasmyth模式的初始参数
        else:  # 默认为guider模式
            ip0 = [360, 89.82, 0, 0, 180]  # guider模式的初始参数
            
        print(f"Initial fitting parameters: {ip0}")
        ifit = FitParam(ifu_xy,ifu_radec,matches,method="IFU",fixedpara=ifixedpara,
                        inipara=ip0, target_err = target_err_ifu, min_data_points= min_data_points_ifu)
        print(f"Fitted WCS parameters:\n{ifit.iwcs_fitted}")
        
        # Check fit quality
        # Get fitted coordinates
        xy2 = ifit.coords[0]
        iwcs_fit = WCS(gwcspara=gwcs_fitted, iwcspara=ifit.iwcs_fitted)
        radec_fit = iwcs_fit.ifs_xy2sky(xy2)
        # Get true coordinates 
        radec_true = ifit.coords[1]
        # Calculate and plot residuals
        WCSSolver.residual(radec_fit, radec_true, plot=plot)
        
        return ifit.iwcs_fitted
    @staticmethod
    def relative_solver(guider_path: str = None,
                       ifu_path: str = None,
                       save_path: str = None,
                       # guider parameters
                       ra_guess_guider: float = None,  # Initial RA guess for guider, in degrees
                       dec_guess_guider: float = None,  # Initial Dec guess for guider, in degrees
                       n_pixs_guider: int = 8,  # Number of pixel coordinates to use for guider
                       n_stars_guider: int = 15,  # Number of catalog stars to use for guider
                       match_radius_guider: float = 0.05,  # Triangle matching radius for guider
                       target_err_guider: float = 0.1,  # Target error threshold for guider, in arcsec
                       min_data_points_guider: int = 3,  # Minimum matched points required for guider
                       # ifu parameters  
                       ra_guess_ifu: float = None,  # Initial RA guess for IFU, in degrees
                       dec_guess_ifu: float = None,  # Initial Dec guess for IFU, in degrees
                       n_pixs_ifu: int = 7,  # Number of pixel coordinates to use for IFU
                       n_stars_ifu: int = 15,  # Number of catalog stars to use for IFU
                       backparam: dict = None,  # Background fitting parameters for IFU centering
                       match_radius_ifu: float = 0.02,  # Triangle matching radius for IFU
                       target_err_ifu: float = 0.1,  # Target error threshold for IFU, in arcsec
                       min_data_points_ifu: int = 3,  # Minimum matched points required for IFU
                       mode: str = "guider",  # Mode for IFU WCS solving: "guider" or "nasmyth"
                       plot: bool = False):
        """
        Solve relative WCS solution between guider and IFU.

        Parameters
        ----------
        guider_path : str
            Path to guider image file
        ifu_path : str
            Path to IFU image file
        save_path : str
            Path to save output WCS file
        ra_guess_guider : float
            Initial RA guess for guider field center, in degrees
        dec_guess_guider : float
            Initial Dec guess for guider field center, in degrees
        n_pixs_guider : int, optional
            Number of pixel coordinates to use for guider matching, default 8
        n_stars_guider : int, optional
            Number of catalog stars to use for guider matching, default 15
        match_radius_guider : float, optional
            Triangle matching radius for guider, default 0.05
        target_err_guider : float, optional
            Target error threshold for guider solution, in arcsec, default 0.1
        min_data_points_guider : int, optional
            Minimum number of matched points required for guider, default 3
        ra_guess_ifu : float
            Initial RA guess for IFU field center, in degrees
        dec_guess_ifu : float
            Initial Dec guess for IFU field center, in degrees
        n_pixs_ifu : int, optional
            Number of pixel coordinates to use for IFU matching, default 7
        n_stars_ifu : int, optional
            Number of catalog stars to use for IFU matching, default 15
        backparam : dict, optional
            Background fitting parameters for IFU star centering
        match_radius_ifu : float, optional
            Triangle matching radius for IFU, default 0.02
        target_err_ifu : float, optional
            Target error threshold for IFU solution, in arcsec, default 0.1
        min_data_points_ifu : int, optional
            Minimum number of matched points required for IFU, default 3
        mode : str, optional
            Mode for IFU WCS solving, either "guider" or "nasmyth", default "guider"
        plot : bool, optional
            Whether to plot results, default False

        Returns
        -------
        OrderedDict
            Dictionary containing fitted IFU WCS parameters
        """
        # Step 1: solve the guider wcs
        print("\nStarting to solve guider WCS...")
        gwcs_fitted = WCSSolver.guider_solver(guider_path, ra_guess_guider, dec_guess_guider, 
                                     n_pixs_guider, n_stars_guider, match_radius_guider,
                                     target_err_guider, min_data_points_guider, plot)

        print("\nStarting to solve IFU WCS...")
        # Step 2: solve IFU WCS
        iwcs_fitted = WCSSolver.iwcs_solver(ifu_path, ra_guess_ifu, dec_guess_ifu,
                                  n_pixs_ifu, n_stars_ifu, backparam, match_radius_ifu,
                                  target_err_ifu, min_data_points_ifu,
                                  gwcs_fitted, mode, plot=plot)
        # Save data:
        if save_path is not None:
            if mode.lower() == "guider":
                # In guider mode, only save iwcspara parameters
                LoadIWCS.Create_Fits(save_path=save_path, iwcspara=iwcs_fitted, guider_nasmyth_iwcspara=None)
            elif mode.lower() == "nasmyth":
                # In nasmyth mode, only save guider_nasmyth_iwcspara parameters
                LoadIWCS.Create_Fits(save_path=save_path, iwcspara=None, guider_nasmyth_iwcspara=iwcs_fitted)
            
        return iwcs_fitted
        
    @staticmethod
    def all_relative_solver(guider_path: str = None,
                           ifu_path: str = None,
                           nasmyth_guider_path: str = None,  # Independent guider path for nasmyth mode
                           save_path: str = None,
                           # guider parameters
                           ra_guess_guider: float = None,  # Initial RA guess for guider, in degrees
                           dec_guess_guider: float = None,  # Initial Dec guess for guider, in degrees
                           n_pixs_guider: int = 8,  # Number of pixel coordinates to use for guider
                           n_stars_guider: int = 15,  # Number of catalog stars to use for guider
                           match_radius_guider: float = 0.05,  # Triangle matching radius for guider
                           target_err_guider: float = 0.1,  # Target error threshold for guider, in arcsec
                           min_data_points_guider: int = 3,  # Minimum matched points required for guider
                           # nasmyth guider parameters
                           ra_guess_nasmyth_guider: float = None,  # Initial RA guess for nasmyth guider
                           dec_guess_nasmyth_guider: float = None,  # Initial Dec guess for nasmyth guider
                           n_pixs_nasmyth_guider: int = 8,  # Number of pixel coordinates for nasmyth guider
                           n_stars_nasmyth_guider: int = 15,  # Number of catalog stars for nasmyth guider
                           match_radius_nasmyth_guider: float = 0.05,  # Triangle matching radius for nasmyth guider
                           target_err_nasmyth_guider: float = 0.1,  # Target error threshold for nasmyth guider
                           min_data_points_nasmyth_guider: int = 3,  # Minimum matched points for nasmyth guider
                           # ifu parameters  
                           ra_guess_ifu: float = None,  # Initial RA guess for IFU, in degrees
                           dec_guess_ifu: float = None,  # Initial Dec guess for IFU, in degrees
                           n_pixs_ifu: int = 7,  # Number of pixel coordinates to use for IFU
                           n_stars_ifu: int = 15,  # Number of catalog stars to use for IFU
                           backparam: dict = None,  # Background fitting parameters for IFU centering
                           match_radius_ifu: float = 0.02,  # Triangle matching radius for IFU
                           target_err_ifu: float = 0.1,  # Target error threshold for IFU, in arcsec
                           min_data_points_ifu: int = 3,  # Minimum matched points required for IFU
                           plot: bool = False):
        """
        Solve relative WCS solutions for both guider and nasmyth modes.

        Parameters
        ----------
        guider_path : str
            Path to guider image file
        ifu_path : str
            Path to IFU image file
        nasmyth_guider_path : str
            Path to nasmyth mode guider image file
        save_path : str
            Path to save output WCS file
        ra_guess_guider : float
            Initial RA guess for guider field center, in degrees
        dec_guess_guider : float
            Initial Dec guess for guider field center, in degrees
        n_pixs_guider : int, optional
            Number of pixel coordinates to use for guider matching, default 8
        n_stars_guider : int, optional
            Number of catalog stars to use for guider matching, default 15
        match_radius_guider : float, optional
            Triangle matching radius for guider, default 0.05
        target_err_guider : float, optional
            Target error threshold for guider solution, in arcsec, default 0.1
        min_data_points_guider : int, optional
            Minimum number of matched points required for guider, default 3
        ra_guess_nasmyth_guider : float
            Initial RA guess for nasmyth guider field center, in degrees
        dec_guess_nasmyth_guider : float
            Initial Dec guess for nasmyth guider field center, in degrees
        n_pixs_nasmyth_guider : int, optional
            Number of pixel coordinates to use for nasmyth guider matching, default 8
        n_stars_nasmyth_guider : int, optional
            Number of catalog stars to use for nasmyth guider matching, default 15
        match_radius_nasmyth_guider : float, optional
            Triangle matching radius for nasmyth guider, default 0.05
        target_err_nasmyth_guider : float, optional
            Target error threshold for nasmyth guider solution, in arcsec, default 0.1
        min_data_points_nasmyth_guider : int, optional
            Minimum number of matched points required for nasmyth guider, default 3
        ra_guess_ifu : float
            Initial RA guess for IFU field center, in degrees
        dec_guess_ifu : float
            Initial Dec guess for IFU field center, in degrees
        n_pixs_ifu : int, optional
            Number of pixel coordinates to use for IFU matching, default 7
        n_stars_ifu : int, optional
            Number of catalog stars to use for IFU matching, default 15
        backparam : dict, optional
            Background fitting parameters for IFU star centering
        match_radius_ifu : float, optional
            Triangle matching radius for IFU, default 0.02
        target_err_ifu : float, optional
            Target error threshold for IFU solution, in arcsec, default 0.1
        min_data_points_ifu : int, optional
            Minimum number of matched points required for IFU, default 3
        plot : bool, optional
            Whether to plot results, default False

        Returns
        -------
        tuple
            Tuple containing (guider_iwcs_fitted, nasmyth_iwcs_fitted)
        """
        print("\nSolving WCS for both guider and nasmyth modes...")
        
        # Solve for guider mode
        print("\n=== SOLVING GUIDER MODE ===")
        guider_iwcs_fitted = WCSSolver.relative_solver(
            guider_path=guider_path,
            ifu_path=ifu_path,
            save_path=None,  # Don't save yet
            ra_guess_guider=ra_guess_guider,
            dec_guess_guider=dec_guess_guider,
            n_pixs_guider=n_pixs_guider,
            n_stars_guider=n_stars_guider,
            match_radius_guider=match_radius_guider,
            target_err_guider=target_err_guider,
            min_data_points_guider=min_data_points_guider,
            ra_guess_ifu=ra_guess_ifu,
            dec_guess_ifu=dec_guess_ifu,
            n_pixs_ifu=n_pixs_ifu,
            n_stars_ifu=n_stars_ifu,
            backparam=backparam,
            match_radius_ifu=match_radius_ifu,
            target_err_ifu=target_err_ifu,
            min_data_points_ifu=min_data_points_ifu,
            mode="guider",
            plot=plot
        )
        
        # Solve for nasmyth mode using independent guider data
        print("\n=== SOLVING NASMYTH MODE ===")
        nasmyth_iwcs_fitted = WCSSolver.relative_solver(
            guider_path=nasmyth_guider_path,  # Use independent nasmyth guider data
            ifu_path=ifu_path,
            save_path=None,  # Don't save yet
            ra_guess_guider=ra_guess_nasmyth_guider,  # Use nasmyth guider parameters
            dec_guess_guider=dec_guess_nasmyth_guider,
            n_pixs_guider=n_pixs_nasmyth_guider,
            n_stars_guider=n_stars_nasmyth_guider,
            match_radius_guider=match_radius_nasmyth_guider,
            target_err_guider=target_err_nasmyth_guider,
            min_data_points_guider=min_data_points_nasmyth_guider,
            ra_guess_ifu=ra_guess_ifu,
            dec_guess_ifu=dec_guess_ifu,
            n_pixs_ifu=n_pixs_ifu,
            n_stars_ifu=n_stars_ifu,
            backparam=backparam,
            match_radius_ifu=match_radius_ifu,
            target_err_ifu=target_err_ifu,
            min_data_points_ifu=min_data_points_ifu,
            mode="nasmyth",
            plot=plot
        )
        
        # Save both results to a single file
        LoadIWCS.Create_Fits(
            save_path=save_path, 
            iwcspara=guider_iwcs_fitted, 
            guider_nasmyth_iwcspara=nasmyth_iwcs_fitted
        )
        
        print(f"\nBoth guider and nasmyth mode WCS parameters saved to {save_path}")
        return (guider_iwcs_fitted, nasmyth_iwcs_fitted)

    @staticmethod 
    def iwcs_transform(gwcs_params, iwcs_params):
        """Calculate WCS parameters for IFU based on guider and intermediate WCS parameters

        Parameters
        ----------
        gwcs_params : OrderedDict
            Guider WCS parameters containing:
            - GCRPIX1, GCRPIX2 : float, reference pixel coordinates [pixel]
            - GCRVAL1, GCRVAL2 : float, reference sky coordinates [deg]
            - GCD1_1, GCD2_2 : float, pixel scale [deg/pixel]
            - GLONPOLE : float, longitude of celestial pole in guider's native coordinates [deg]
        iwcs_params : OrderedDict
            Intermediate WCS parameters containing:
            - ICRPIX1, ICRPIX2 : float, reference pixel coordinates [pixel]
            - ICRVAL1, ICRVAL2 : float, reference sky coordinates [deg]
            - ICD1_1, ICD2_2 : float, pixel scale [deg/pixel]
            - ILONPOLE : float, longitude of guider's celestial pole in IFU's native coordinates [deg]

        Returns
        -------
        wcspara : OrderedDict
            Final IFU WCS parameters containing:
            - CRPIX1, CRPIX2 : float, reference pixel coordinates [pixel]
            - CRVAL1, CRVAL2 : float, reference sky coordinates [deg]
            - CD1_1, CD1_2, CD2_1, CD2_2 : float, transformation matrix [deg/pixel]
        """
        # input:
        lon_ifu_g = float(iwcs_params["ICRVAL1"])
        lat_ifu_g = float(iwcs_params["ICRVAL2"])
        # parameter:

        lon_gui_p = float(gwcs_params["GCRVAL1"])
        lat_gui_p = float(gwcs_params["GCRVAL2"])
        lon_pole_g = float(gwcs_params["GLONPOLE"])
        # output:
        ra0_i, dec0_i = WCS.sphere_rotate(
            lon_ifu_g, lat_ifu_g, lon_gui_p, lat_gui_p,
            lon_pole_g)  # CRVAL1/2 of IFS
        ra0_i = float(ra0_i)
        dec0_i = float(dec0_i)
        
        # input
        lon_pole_g = float(gwcs_params["GLONPOLE"])  # polar longitude in Guider
        lat_pole_g = float(gwcs_params["GCRVAL2"])  # polar latitude in Guider
        #  parameter:
        lon_gui_i = float(iwcs_params["ILONPOLE"])  # Guider longitude in IFS
        lat_gui_i = float(iwcs_params["ICRVAL2"])  # Guider latitude in IFS
        lon_ifu_g = float(iwcs_params["ICRVAL1"])  # IFS longitude in Guider
        # output:
        # Calculate polar longitude in IFS native coordinates
        lon_pole_i, lat_pole_i = WCS.sphere_rotate(
            lon_pole_g, lat_pole_g, lon_gui_i, lat_gui_i,
            lon_ifu_g)  # polar longitude in IFS
            
        theta = np.deg2rad(lon_pole_i - 180)  # rotation angle between IFS to sky

        
        coord = SkyCoord(ra0_i, dec0_i, unit=u.deg)
        ra_str = coord.ra.to_string(unit=u.hourangle, sep=':', precision=2)
        dec_str = coord.dec.to_string(unit=u.deg, sep=':', precision=2)
        
        print(f"IFU Position Angle: {lon_pole_i - 180}")
        print(f"IFU Center: RA={ra_str}, DEC={dec_str}")
        
        pixel_size_i1 = np.abs(iwcs_params["ICD1_1"])  # x spaxel length of IFS
        pixel_size_i2 = np.abs(iwcs_params["ICD2_2"])  # y spaxel length of IFS
        wcspara = OrderedDict({
            "CRPIX1": iwcs_params["ICRPIX1"],
            "CRPIX2": iwcs_params["ICRPIX2"], 
            "CRVAL1": ra0_i,
            "CRVAL2": dec0_i,
            "CD1_1": -np.cos(theta) * pixel_size_i1,
            "CD1_2": np.sin(theta) * pixel_size_i1,
            "CD2_1": np.sin(theta) * pixel_size_i2,
            "CD2_2": np.cos(theta) * pixel_size_i2
        })
        return wcspara

    @staticmethod
    def ifu_solver(guider_path: str = None,
                   ifu_path: str = None,
                   iwcs_path: str = None,
                   save_path: str = None,
                   ra_guess_guider: float = None,
                   dec_guess_guider: float = None,
                   n_pixs_guider: int = 8,
                   n_stars_guider: int = 15,
                   match_radius_guider: float = 0.05,
                   target_err_guider: float = 0.1,
                   min_data_points_guider: int = 3,
                   mode: str = "guider",
                   plot: bool = False):
        """
        Solve WCS parameters for IFU using guider camera data and IWCS reference file.

        Parameters
        ----------
        guider_path : str
            Path to guider camera FITS file
        ifu_path : str
            Path to IFU FITS file
        iwcs_path : str
            Path to IWCS reference file
        save_path : str
            Path to save output WCS file
        ra_guess_guider : float
            Initial RA guess for guider field center [deg]
        dec_guess_guider : float
            Initial Dec guess for guider field center [deg]
        n_pixs_guider : int, optional
            Number of pixel coordinates to use for guider matching, default 8
        n_stars_guider : int, optional
            Number of catalog stars to use for guider matching, default 15
        match_radius_guider : float, optional
            Triangle matching radius for guider, default 0.05
        target_err_guider : float, optional
            Target error threshold for guider solution [arcsec], default 0.1
        min_data_points_guider : int, optional
            Minimum number of matched points required for guider, default 3
        mode : str, optional
            Mode for IFU WCS solving, either "guider" or "nasmyth", default "guider"
        plot : bool, optional
            Whether to plot results, default False

        Returns
        -------
        wcspara : OrderedDict
            Dictionary containing fitted IFU WCS parameters:
            - CRPIX1, CRPIX2 : float, reference pixel coordinates [pixel]
            - CRVAL1, CRVAL2 : float, reference sky coordinates [deg]
            - CD1_1, CD1_2, CD2_1, CD2_2 : float, transformation matrix [deg/pixel]
        """        
        
        print("\nLoading IWCS reference file...")
        iwcs = LoadIWCS(iwcs_path)
        print("\nLoading IFU data...")
        rss = LoadRSS(ifu_path,plot_map=plot)
        print("Loading guider camera data...")
        gwcs_fitted = WCSSolver.guider_solver(guider_path=guider_path,
                                            ra_guess_guider=ra_guess_guider,
                                            dec_guess_guider=dec_guess_guider,
                                            n_pixs_guider=n_pixs_guider,
                                            n_stars_guider=n_stars_guider,
                                            match_radius_guider=match_radius_guider,
                                            target_err_guider=target_err_guider,
                                            min_data_points_guider=min_data_points_guider,
                                            plot=plot)
        
        print("\nCalculating IFU WCS parameters...")
        if mode.lower() == "guider":
            wcspara = WCSSolver.iwcs_transform(gwcs_fitted, iwcs.guider_iwcspara)
        elif mode.lower() == "nasmyth":
            wcspara = WCSSolver.iwcs_transform(gwcs_fitted, iwcs.guider_nasmyth_iwcspara)
        else:
            raise ValueError("Mode must be either 'guider' or 'nasmyth'")
            
        print("IFU WCS parameters:")
        for key, value in wcspara.items():
            print(f"{key}: {value}")
        print("WCS parameters calculation completed")

        print("\nSaving results...")
        rss.Create_Fits(save_path =save_path,wcspara = wcspara, gwcspara = gwcs_fitted, iwcspara = iwcs.guider_iwcspara if mode.lower() == "guider" else iwcs.guider_nasmyth_iwcspara)
        print("Processing completed!")
        return wcspara

    @staticmethod
    def residual(radec_fit, radec_true, plot = False):

        # 残差
        delta_radec = (radec_fit - radec_true)*3600
        dra = delta_radec[:,0] * np.cos(radec_true[:,1] / 180 * np.pi)
        ddec = delta_radec[:,1]
        RMSE =  np.sqrt((dra**2 + ddec**2).sum()/len(dra))
        print("Fitted celestial coordinates: \n", radec_fit)          
        print("True celestial coordinates: \n", radec_true)
        print("Fitting residuals (arcsec): \n", delta_radec)
        print("Root Mean Square Error of fitting residuals (arcsec): {}".format(RMSE))
        if plot:
            plt.figure(figsize=(5,5))
            ax=plt.gca()
            ax.scatter(dra ,ddec,s=3,c="C1")
            ax.scatter(0, 0, c='red', marker='+', s=200) # 在(0,0)处画大十字
            ax.set_xlabel("$\Delta RA(arcsec)$",fontsize=15)
            ax.set_ylabel("$\Delta DEC(arcsec)$",fontsize=15)
            ax.text(0.95, 0.95, f'RMSE: {RMSE:.3f}"', 
                   transform=ax.transAxes,
                   horizontalalignment='right',
                   verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.8))
        return dra, ddec, RMSE