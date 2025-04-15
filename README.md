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
Chili 是位于云南丽江的2.4m望远镜上的积分视场光谱仪，本代码是为了给该仪器进行天体测量WCS定标而生的。
代码主要能解决以下几个问题:
1.相对位置定标：通过观测密集星场，可以标定IFU和Chili导星以及耐焦导星相机之间的相对位置关系。
2.观测计划: 给定IFU中心指向RA和DEC以及PA角，预测chili导星和耐焦导星相机的指向和PA 以及生成参考天图。
3.WCS定标: 给定IFU科学观测时导星相机拍摄的图像，反推出IFU的WCS参数以及每根光纤对应的RA、DEC

为实现上述功能，代码主要具有星像质心定位、星像和星表配准、WCS参数解算等功能(以后再补充文档)。

快速入门可以参考，在/chili_wcs/example 文件中提供了两个使用的例子

其中WCS_calibration_20250415.ipynb里介绍了如何使用本程序标定IFU和导星相机之间的相对位置关系，以及观测后如何利用导星相机的图像预测IFU的WCS参数和每个Fiber在天球坐标系下的位置, /chili_wcs/example/calibration_data里保存了用于定标的数据，为kopff27标准星，/chili_wcs/example/example_data/results/IWCS_20250415.fits 文件里保存了导星和IFU之间的相对位置关系参数，可用来对不同仪器之间的WCS参数进行转换。

而/chili_wcs/example/Chili_Plan_Tool.ipynb 给出了如何利用本代码，给定IFU的中心指向和PA角，结合相对位置关系参数，预测导星相机的指向，并给出预测的天区图，预测结果在/chili_wcs/example/plantoolresults中查看，可以作为观测时的参考图像。

下图展示了使用 Chili_Plan_Tool 生成的预测天区图示例，显示了 IFU、导星相机和耐焦导星相机的视场：
![Chili天区预测图](chili_wcs/example/plantoolresults/M1/ChiliSky.jpg)