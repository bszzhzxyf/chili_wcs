"""
Identifier:     chili_wcs/guider.py
Name:           guider.py
Description:    predict guider pointing
Author:         Yifei Xiong
Created:        2024/06/04
Modified-History:
    2024/06/04:add Mockguider class
"""
from .wcs import WCS
import numpy as np

class MockGuider():
    def __init__(self, ra_IFU,dec_IFU,PA_IFU,config):
        self.ra_IFU = ra_IFU
        self.dec_IFU = dec_IFU
        self.PA_IFU = PA_IFU
        self.config = config
        self.pointing = self.guider_pointing(self.ra_IFU,self.dec_IFU,self.PA_IFU)

    def guider_pointing(self,ra_IFU,dec_IFU,PA_IFU):
        # relative paramter
        ICRVAL1 = 270                   # <IFS本地天球坐标>的“北天极”在<Guider本地天球坐标>中的经度
        ICRVAL2 = 90 - 10.85/60                   # <IFS本地天球坐标>的“北天极”在<Guider本地天球坐标>中的纬度
        ILONPOLE = 90                      # <Guider本地天球坐标>的“北天极”在<IFS本地天球坐标>中的经度

        lon_gui_i = GCRVAL1 = np.deg2rad(ILONPOLE) # guider center longitude in IFS native sky
        lat_gui_i = GCRVAL2 = np.deg2rad(ICRVAL2)  # guider center latitude in IFS native sky
        lon_pole_i = LONPOLE = np.deg2rad(180 + PA_IFU) # Polar longitude in IFS native sky
        lat_pole_i = np.deg2rad(dec_IFU)   # Polar latitude in IFS native sky
        # Calculate guider center pointing
        ra0_guider, dec0_guider = WCS.sphere_rotate(phi = lon_gui_i,
                                                    theta = lat_gui_i, 
                                                    ra0 = ra_IFU, 
                                                    dec0 = dec_IFU, 
                                                    phi_p = lon_pole_i)
        # Calculate guider PA
        
        lon_pole_g, lat_pole_g = WCS.sphere_rotate(phi = lon_pole_i,
                                                theta = lat_pole_i, 
                                                ra0 = ICRVAL1, 
                                                dec0 = ICRVAL2, 
                                                phi_p = ILONPOLE)
        PA_guider = lon_pole_g - 180
        return ra0_guider, dec0_guider, PA_guider