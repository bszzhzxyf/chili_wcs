"""
Identifier:     chili_wcs/__init__.py
Name:           __init__.py
Description:    chili_wcs __init__
Author:         Yifei Xiong
Created:        2023-11-16
Modified-History:
    2023-11-16:Create by Yifei Xiong
"""

from .csst_ifs_l1_wcs import PipelineL1IFSWCS
from .coord_data import CoordData
from .fit_wcs import TriMatch, FitParam
from .load_data import LoadRSS, LoadFGS, LoadIWCS
from .wcs_to_rss import wcs_to_rss_FGS, fake_wcs
from .wcs import WCS


__version__ = "0.0.1 dev"
__all__ = ["PipelineL1IFSWCS", "CoordData", "TriMatch", "FitParam", "LoadRSS", "LoadFGS", "LoadIWCS", "WCS", "fake_wcs", "wcs_to_rss_FGS"]
