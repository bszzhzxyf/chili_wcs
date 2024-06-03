## csst_ifs_wcs

This is a package for csst-ifs astrometry calibration

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
csst_ifs_wcs can be installed with the following shell command

```shell
git clone https://csst-tb.bao.ac.cn/code/csst-l1/ifs/csst_ifs_wcs.git
cd csst_ifs_wcs
pip install .
```

or this single line command
```shell
pip install --force-reinstall git+https://csst-tb.bao.ac.cn/code/csst-l1/ifs/csst_ifs_wcs.git
```
## tutorial
本代码可用于生成CSST-IFS的WCS数据,并添加到RSS的头文件中,使用方法如下:\
This code is used to generate WCS parameters and add it into RSS header:

```
from csst_ifs_wcs.csst_ifs_l1_wcs import PipelineL1IFSWCS

input_rss_file = "../data/CSST_IFS_SCIE_20230110083940_20230110083940_300000025_A_L1_R6100_bewnrfc_.fits"
output_dir = "../outpath/"

ifs_wcs = PipelineL1IFSWCS()
ifs_wcs.para_set(path_ifs_rss=input_rss_file,
                    path_guider_img=None,
                    path_iwcs_ref=None,
                    output_path=output_dir)
result = ifs_wcs.run()
```

本代码也可用于IFS和其他图像的WCS参数定标，提供数据载入、星像定心、Gaia星表导入、三角匹配、像素到天球的WCS转换算法、参数解算等功能:\
This code can also be used for WCS parameter calibration of IFS and other images, providing star image centering, Gaia catalog, Triange matching,WCS transform, Parameter fitting and other functions:
```
from csst_ifs_wcs.load_data import LoadRSS  # 载入RSS模块
from csst_ifs_wcs.coord_data import CoordData  # 星点像素坐标与天球坐标
from csst_ifs_wcs.fit_wcs import TriMatch, FitParam  # 三角匹配, 参数拟合
from csst_ifs_wcs.wcs import WCS  # WCS底片模型

path = "../data/CSST_IFS_SCIE_20230110083940_20230110083940_300000025_A_L1_R6100_bewnrfc_.fits"
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