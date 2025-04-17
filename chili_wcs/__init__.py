"""
Identifier:     chili_wcs/__init__.py
Name:           __init__.py
Description:    chili_wcs __init__
Author:         Yifei Xiong
Created:        2023-11-16
Modified-History:
    2023-11-16:Create by Yifei Xiong
    2025-04-18:Modified by Yifei Xiong
"""

from .coord_data import CoordData
from .fit_wcs import TriMatch, FitParam
from .load_data import LoadRSS, LoadGuider, LoadIWCS
from .wcs_to_rss import  fake_wcs, wcs_to_rss_Guider
from .wcs import WCS


__version__ = "0.0.3 dev"
__all__ = ["PipelineL1IFSWCS", "CoordData", "TriMatch", "FitParam", "LoadRSS", "LoadGuider", "LoadIWCS", "WCS", "fake_wcs", "wcs_to_rss_Guider"]
