import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = [
        req.strip()
        for req in f.readlines()
        if not req.startswith("#") and req.__contains__("==")
    ]

setuptools.setup(
    name='csst_ifs_wcs',
    version="0.0.1",
    author='Yifei Xiong',
    author_email='xiongyf@shao.ac.cn',
    description='WCS package for CSST IFS pipeline',  # short description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://csst-tb.bao.ac.cn/code/csst-l1/ifs/csst_ifs_wcs',
    project_urls={
        'Source': 'https://csst-tb.bao.ac.cn/code/csst-l1/ifs/csst_ifs_wcs',
    },
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=["Development Status :: 5 - Production/Stable",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                 "Programming Language :: Python :: 3",
                 "Topic :: Scientific/Engineering :: Physics",
                 "Topic :: Scientific/Engineering :: Astronomy"],
    package_dir={'csst_ifs_wcs': 'csst_ifs_wcs'},
    # include_package_data=True,
    package_data={"": ["LICENSE", "README.md"],
                  },
    # install_requires=['numpy',
    #                   'scipy', 'matplotlib',
    #                    'sep', 'photutils'],
    python_requires='>=3.11',
    install_requires=requirements,
)

