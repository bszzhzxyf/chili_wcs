"""
Identifier:     chili_wcs/wcs.py
Name:           wcs.py
Description:    WCS transform functions，calculate sky coordinates from pixels.
Author:         Yifei Xiong
Created:        2023-11-30
Modified-History:
    2023-11-30: add module header
"""

import numpy as np
from astropy.wcs import WCS as aWCS


class WCS():
    """
    A class for World Coordinate System (WCS) transformations.

    This class handles the transformation between pixel coordinates and celestial coordinates
    using different WCS parameters for standard WCS, MCI ,and IFS.

    Parameters
    ----------
    header : astropy.fits.header, optional
        Fits Headers that include WCS parameters.
    wcspara : collection.dict, optional
        Standard WCS parameters.
    mwcspara : collection.dict, optional
        MCI_WCS parameters.
    iwcspara : collection.dict, optional
        IFS_WCS parameters.
    pywcs : astropy.WCS, optional
        Astropy WCS class.

    Attributes
    ----------
    CRPIX : np.ndarray
        Pixel coordinates of the reference point.
    CD : np.ndarray
        Coordinate transformation matrix.
    CRVAL : np.ndarray
        Celestial coordinates of the reference point.

    MCRPIX : np.ndarray
        Pixel coordinates of the MCI reference point.
    MCD : np.ndarray
        Coordinate transformation matrix for MCI.
    MCRVAL : np.ndarray
        Celestial coordinates of the MCI reference point.
    MLONPOLE : float
        Longitude of the celestial pole.

    ICRPIX : np.ndarray
        Pixel coordinates of the IFS reference point.
    ICD : np.ndarray
        Coordinate transformation matrix for IFS.
    ICRVAL : np.ndarray
        Celestial coordinates of the IFS reference point.
    ILONPOLE : float
        Longitude of the celestial pole for IFS.

    Methods
    -------
    xy2sky(xy)
        Transform pixel coordinates to celestial coordinates for standard WCS.
    mci_xy2sky(xy)
        Transform pixel coordinates to celestial coordinates for MCI.
    ifs_xy2sky(xy)
        Transform pixel coordinates to celestial coordinates for IFS.
    """

    def __init__(self,
                 header: dict = None,
                 wcspara: dict = None,
                 mwcspara: dict = None,
                 iwcspara: dict = None,
                 pywcs: type(aWCS()) = None):

        self.CRPIX = np.array([0, 0])
        self.CD = np.array([[0, 0], [0, 0]])
        self.CRVAL = np.array([0, 0])

        self.MCRPIX = np.array([0, 0])
        self.MCD = np.array([[0, 0], [0, 0]])
        self.MCRVAL = np.array([0, 0])
        self.MLONPOLE = 0

        self.ICRPIX = np.array([0, 0])
        self.ICD = np.array([[0, 0], [0, 0]])
        self.ICRVAL = np.array([0, 0])
        self.ILONPOLE = 0

        if header is not None:
            self.header = header
            self.CRPIX = np.array(
                [self.header['CRPIX1'], self.header['CRPIX2']])
            self.CD = np.array([[self.header['CD1_1'], self.header['CD1_2']],
                                [self.header['CD2_1'], self.header['CD2_2']]])
            self.CRVAL = np.array(
                [self.header['CRVAL1'], self.header['CRVAL2']])

        if wcspara is not None:
            self.wcspara = wcspara
            self.CRPIX = np.array(
                [self.wcspara['CRPIX1'], self.wcspara['CRPIX2']])
            self.CD = np.array(
                [[self.wcspara['CD1_1'], self.wcspara['CD1_2']],
                 [self.wcspara['CD2_1'], self.wcspara['CD2_2']]])
            self.CRVAL = np.array(
                [self.wcspara['CRVAL1'], self.wcspara['CRVAL2']])
            # self.LONPOLE = self.param['LONPOLE']
        if mwcspara is not None:
            self.mwcspara = mwcspara
            self.MCRPIX = np.array(
                [self.mwcspara['MCRPIX1'], self.mwcspara['MCRPIX2']])
            self.MCD = np.array([[self.mwcspara['MCD1_1'], 0],
                                 [0, self.mwcspara['MCD2_2']]])
            self.MCRVAL = np.array(
                [self.mwcspara['MCRVAL1'], self.mwcspara['MCRVAL2']])
            self.MLONPOLE = self.mwcspara['MLONPOLE']
        if iwcspara is not None:
            self.iwcspara = iwcspara
            self.ICRPIX = np.array(
                [self.iwcspara['ICRPIX1'], self.iwcspara['ICRPIX2']])
            self.ICD = np.array([[self.iwcspara['ICD1_1'], 0],
                                 [0, self.iwcspara['ICD2_2']]])
            self.ICRVAL = np.array(
                [self.iwcspara['ICRVAL1'], self.iwcspara['ICRVAL2']])
            self.ILONPOLE = self.iwcspara['ILONPOLE']

        if pywcs is not None:
            self.CRPIX = np.array([pywcs.wcs.crpix[0], pywcs.wcs.crpix[1]])
            self.CD = np.array([[pywcs.wcs.cd[0, 0], pywcs.wcs.cd[0, 1]],
                                [pywcs.wcs.cd[1, 0], pywcs.wcs.cd[1, 1]]])
            self.CRVAL = np.array([pywcs.wcs.crval[0], pywcs.wcs.crval[1]])

    @staticmethod
    def arg(x, y):
        """
        if not isinstance(x, type(np.array([3]))):
            x = np.array([x])
        if not isinstance(y, type(np.array([3]))):
            y = np.array([y])
        """
        arg = np.ones_like(x).astype(float)
        for i in range(len(x)):
            if (x[i] >= 0) and (y[i] >= 0):
                arg[i] = np.arctan2(y[i], x[i])
            elif (x[i] < 0) and (y[i] >= 0):
                arg[i] = np.arctan2(y[i], x[i])
            elif (x[i] < 0) and (y[i] < 0):
                arg[i] = np.arctan2(y[i], x[i]) + 2 * np.pi
            elif (x[i] >= 0) and (y[i] < 0):
                arg[i] = np.arctan2(y[i], x[i]) + 2 * np.pi
        return arg

    # Step1
    @staticmethod
    def pix2cpix(x, y, x0, y0):
        # Step 1
        # input x,y is column vector(array)(initial is 0)
        u = x - (x0 - 1)
        v = y - (y0 - 1)
        return u, v

    @staticmethod
    def cpix2proj(u, v, CD):
        # Step 2
        cpix = np.array([u, v])
        proj = np.array(CD @ cpix)
        return proj

    # Step3
    @staticmethod
    def proj2sphere(proj):
        # Step 3
        phi = WCS.arg(-proj[1], proj[0])
        R = np.sqrt(proj[0]**2 + proj[1]**2)
        with np.errstate(divide='ignore'):
            theta = np.arctan(180 / (np.pi * R))
        for i, r in enumerate(R):
            if r == 0:
                phi[i] = 0
                theta[i] = np.pi / 2
        return phi, theta

    # Step4
    @staticmethod
    def sphere_rotate(phi, theta, ra0, dec0, phi_p):
        # Step 4
        phi_p = np.deg2rad(phi_p)
        dec0 = np.deg2rad(dec0)
        t = np.arctan2(
            -np.cos(theta) * np.sin(phi - phi_p),
            np.sin(theta) * np.cos(dec0) -
            np.cos(theta) * np.sin(dec0) * np.cos(phi - phi_p))
        ra = ra0 + np.degrees(t)
        dec = np.degrees(
            np.arcsin(
                np.sin(theta) * np.sin(dec0) +
                np.cos(theta) * np.cos(dec0) * np.cos(phi - phi_p)))
        return ra, dec

    # Normal Method
    @staticmethod
    def wcs_transform(x, y, x0, y0, CD, ra0, dec0, phi_p):
        u, v = WCS.pix2cpix(x, y, x0, y0)
        proj = WCS.cpix2proj(u, v, CD)
        phi, theta = WCS.proj2sphere(proj)
        ra, dec = WCS.sphere_rotate(phi, theta, ra0, dec0, phi_p)
        radec = np.column_stack([ra, dec])
        return radec

    def xy2sky(self, xy):
        # input pixel xy is column vector
        x, y = xy.T
        # parameters
        phi_p = 180
        x0, y0 = self.CRPIX
        ra0, dec0 = self.CRVAL
        CD = self.CD
        # tranform
        radec = WCS.wcs_transform(x, y, x0, y0, CD, ra0, dec0, phi_p)
        return radec

    # MCI transform
    @staticmethod
    def mci_transform(mx, my, mx0, my0, mCD, mra0, mdec0, mphi_p):
        u, v = WCS.pix2cpix(mx, my, mx0, my0)
        proj = WCS.cpix2proj(u, v, mCD)
        phi, theta = WCS.proj2sphere(proj)
        ra, dec = WCS.sphere_rotate(phi, theta, mra0, mdec0, mphi_p)
        radec = np.column_stack([ra, dec])  # two column array
        return radec

    def mci_xy2sky(self, xy):
        # input pixel xy
        x, y = xy.T
        # parameter
        mx0, my0 = self.MCRPIX
        mra0, mdec0 = self.MCRVAL
        mCD = self.MCD
        mphi_p = self.MLONPOLE
        # transform
        radec = WCS.mci_transform(x, y, mx0, my0, mCD, mra0, mdec0, mphi_p)
        return radec

    # IFS transform
    @staticmethod
    def ifs_transform(x, y, mra0, mdec0, mphi_p, ix0, iy0, iCD, icv1, icv2,
                      iphi_p):
        iu, iv = WCS.pix2cpix(x, y, ix0, iy0)
        iproj = WCS.cpix2proj(iu, iv, iCD)
        iphi, itheta = WCS.proj2sphere(iproj)
        phi, theta = WCS.sphere_rotate(iphi, itheta, icv1, icv2, iphi_p)
        phi, theta = np.deg2rad(phi), np.deg2rad(theta)
        ra, dec = WCS.sphere_rotate(phi, theta, mra0, mdec0, mphi_p)
        radec = np.column_stack([ra, dec])  # two column array
        return radec

    def ifs_xy2sky(self, xy):
        # input pixel xy
        x, y = xy.T
        # parameter
        # MCI
        mra0, mdec0 = self.MCRVAL
        mphi_p = self.MLONPOLE
        # IFS
        ix0, iy0 = self.ICRPIX
        icv1, icv2 = self.ICRVAL
        iCD = self.ICD
        iphi_p = self.ILONPOLE
        ###################
        # transform
        radec = WCS.ifs_transform(x, y, mra0, mdec0, mphi_p, ix0, iy0, iCD,
                                  icv1, icv2, iphi_p)
        return radec
