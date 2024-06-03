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
    name='chili_wcs',
    version="0.0.1",
    author='Yifei Xiong',
    author_email='xiongyf@shao.ac.cn',
    description='WCS package for Chili pipeline',  # short description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/bszzhzxyf/chili_wcs',
    project_urls={
        'Source': 'https://github.com/bszzhzxyf/chili_wcs',
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
    package_dir={'chili_wcs': 'chili_wcs'},
    # include_package_data=True,
    package_data={"": ["LICENSE", "README.md"],
                  },
    # install_requires=['numpy',
    #                   'scipy', 'matplotlib',
    #                    'sep', 'photutils'],
    python_requires='>=3.7',
    install_requires=requirements,
)

