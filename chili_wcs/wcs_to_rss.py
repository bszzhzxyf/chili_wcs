"""
Identifier:     csst_ifs_wcs/wcs_to_rss.py
Name:           wcs_to_rss.py
Description:    Calculate WCS parameter and add it to rss header.
Author:         Yifei Xiong
Created:        2023-11-30
Modified-History:
    2023-11-30: add module header.
    2024-6-2: debug
"""

import numpy as np
from collections import OrderedDict
from .load_data import LoadRSS, LoadFGS, LoadIWCS
from .wcs import WCS


def wcs_to_rss_FGS(path_ifs_rss: str, path_guider_img: str, path_iwcs_ref: str, output_path: str):
    """
    Generate inferred wcs.

    This function provide indirect WCS generate from FGS.With FGS data.

    Parameters
    ----------
    path_ifs_rss : str
        Input RSS file path.
    path_guider_img : str
        Input Guider image path.
    path_iwcs_ref : str
        Input reference file path , which contain relative position relation parameter.
    output_path : str
        Output path,save RSS fits file.

    Returns
    -------
    exit_code: int
        "exit_code" for status
        "290":good status
        "291":wrong status.
    out_name: str
        Output file path + file name.
    wcs_para: dict
        WCS parameter for RSS.
    """
    try:
        guider = LoadFGS(path_guider_img)
        iwcs = LoadIWCS(path_iwcs_ref)
        rss = LoadRSS(path_ifs_rss)
    except:
        exit_code = 291
        out_name = None
        wcs_para = None
        return exit_code, out_name, wcs_para
    mwcspara = guider.mwcspara
    iwcspara = iwcs.iwcspara
    mi_wcs = WCS(mwcspara=mwcspara, iwcspara=iwcspara)
    x0_y0_i = np.array([[iwcspara["ICRPIX1"] - 1, iwcspara["ICRPIX2"] - 1]])
    ra0_i, dec0_i = mi_wcs.ifs_xy2sky(x0_y0_i)[0]  # CRVAL1/2 of IFS
    # calculate polar longitude in IFS native coordinates
    # input
    lon_pole_m = np.deg2rad(mwcspara["MLONPOLE"])  # polar longitude in MCI
    lat_pole_m = np.deg2rad(mwcspara["MCRVAL2"])  # polar latitude in MCI
    # Parameter
    lon_mci_i = iwcspara["ILONPOLE"]  # MCI longitude in IFS
    lat_mci_i = iwcspara["ICRVAL2"]  # MCI latitude in IFS
    lon_ifs_m = iwcspara["ICRVAL1"]  # IFS longitude in MCI
    # output
    lon_pole_i, lat_pole_i = WCS.sphere_rotate(
        lon_pole_m, lat_pole_m, lon_mci_i, lat_mci_i,
        lon_ifs_m)  # polar longitude in IFS
    theta = np.deg2rad(lon_pole_i - 180)  # rotation angle between IFS to sky
    pixel_size_i1 = np.abs(iwcspara["ICD1_1"])  # x spaxel length of IFS
    pixel_size_i2 = np.abs(iwcspara["ICD2_2"])  # y spaxel length of IFS
    wcs_para = OrderedDict({
        "CRPIX1": iwcspara["ICRPIX1"],
        "CRPIX2": iwcspara["ICRPIX2"],
        "CRVAL1": ra0_i,
        "CRVAL2": dec0_i,
        "CD1_1": -np.cos(theta) * pixel_size_i1,
        "CD1_2": np.sin(theta) * pixel_size_i1,
        "CD2_1": np.sin(theta) * pixel_size_i2,
        "CD2_2": np.cos(theta) * pixel_size_i2
    })
    rss.wcs = WCS(wcspara=wcs_para)
    rss.Create_Fits(output_path)
    exit_code = 290
    # change name
    infile0 = list(path_ifs_rss)
    infile0[-6] = "g"
    out_name = "".join(infile0)
    return exit_code, out_name, wcs_para


def fake_wcs(input_file: str, output_path: str):
    """
    Generate inferred wcs , add it to fits header,and outputfile.

    This function provide direct WCS generate from PA,RA,DEC header.

    Parameters
    ----------
    input_file : str
        Input file path such as
        '/data/ifspip/ASTM/0.6.1.2/CSST_IFS_SCIE_20230110131140_20230110131140_300000081_A_L1_R6100_bewnrfcg.fits'.
    output_path : str
        Output savepath such as
        "/data/ifspip/ASTM/0.6.1.2/".

    Returns
    -------
    exit_code: int
        "exit_code" for status
        "290":good status
        "291":wrong status.
    out_name: str
        Output file path + file name.
    wcs_para: dict
        WCS parameter for RSS.
    """
    try:
        rss = LoadRSS(input_file, x0=31 / 2, y0=31 / 2)
    except:
        exit_code = 291
        out_name = None
        wcs_para = None
        return exit_code, out_name, wcs_para
    theta_r = rss.hdu[0].header["POS_ANG1"] / 180 * np.pi
    ray = rss.hdu[0].header["RA_PNT0"]
    decy = rss.hdu[0].header["DEC_PNT0"]
    ra_obj = rss.hdu[0].header["RA_OBJ"]
    dec_obj = rss.hdu[0].header["DEC_OBJ"]
    d_ra = -(ray - ra_obj) * np.cos(dec_obj / 180 * np.pi) / np.cos(
        decy / 180 * np.pi)
    d_dec = decy - dec_obj
    dec0 = dec_obj + d_dec
    ra0 = ra_obj + d_ra
    wcs_para = OrderedDict({
        "CRPIX1": 16.5,
        "CRPIX2": 16.5,
        "CRVAL1": ra0,
        "CRVAL2": dec0,
        "CD1_1": -np.cos(-theta_r) * 0.2 / 3600,
        "CD1_2": np.sin(-theta_r) * 0.2 / 3600,
        "CD2_1": np.sin(-theta_r) * 0.2 / 3600,
        "CD2_2": np.cos(-theta_r) * 0.2 / 3600
    })
    rss.wcs = WCS(wcspara=wcs_para)
    rss.Create_Fits(output_path)
    exit_code = 290
    # change name
    infile = input_file
    infile0 = list(infile)
    infile0[-6] = "g"
    out_name = "".join(infile0)
    return exit_code, out_name, wcs_para
