## chili_wcs

This is a package for chili astrometry calibration

## dependency
numpy==1.26.1\
scipy==1.11.3\
matplotlib==3.7.2\
pandas==2.0.3\
setuptools==68.0.0\
astropy==5.3.4\
astroquery==0.4.6\
sep==1.2.1

## installation
chili_wcs can be installed with the following shell command

```shell
git clone https://github.com/bszzhzxyf/chili_wcs.git
cd chili_wcs
pip install .
```
## tutorial
1.本代码可用于生成CHILI积分视场光谱仪的WCS数据,并添加到RSS的头文件中

2.本代码可以给定IFU中心的RA、DEC和PA角预测导星相机的RA、DEC和PA角：
```
from chili_wcs.guider import guider_pointing

ra_IFU =  111.11    # RA in degree
dec_IFU = 22.22     # DEC in degree
PA_IFU = 10         # PA in degree 

ra0_guider, dec0_guider, PA_guider = guider_pointing(ra_IFU,dec_IFU,PA_IFU)  # unit: degree
```

3.本代码可用于IFS和其他图像的WCS参数定标，提供数据载入、星像定心、Gaia星表导入、三角匹配、像素到天球的WCS转换算法、参数解算等功能:\
This code can also be used for WCS parameter calibration of IFS and other images, providing star image centering, Gaia catalog, Triange matching,WCS transform, Parameter fitting and other functions:
```
from chili_wcs.load_data import LoadRSS  # 载入RSS模块
from chili_wcs.coord_data import CoordData  # 星点像素坐标与天球坐标
from chili_wcs.fit_wcs import TriMatch, FitParam  # 三角匹配, 参数拟合
from chili_wcs.wcs import WCS  # WCS底片模型

path = "./rss.fits"
rss = LoadRSS(path, path_filter=None)

# Coordinate Datas  用于拟合的输入输出坐标数据准备
rss_coord = CoordData(rss, rad=6., radec_point=[ra0, dec0])
rss_xy = rss_coord.xy  # pixel_coordinates
rss_radec = rss_coord.radec # ra dec data

# Triange matching  三角匹配像素坐标和天球坐标
trimatch = TriMatch(rss_xy, rss_radec)
matches = trimatch.matches_array  

# Fit parameter 解算WCS参数
inipara = OrderedDict({"CRPIX": np.array([16.5, 16.5])})    
p0 = [0, 0, 0, 0, 0, 0]
fit = FitParam(rss_xy,
                rss_radec,
                matches,
                method="Normal",
                inipara=inipara,
                p0=p0)}
print(fit.bestparam)
print(fit.wcs_fitted)
```