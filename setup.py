import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='nidmd',
     version='0.1.3',
     author="Arnaud Dhaene (EPFL)",
     author_email="arnaud.dhaene@epfl.ch",
     description="Dynamic Mode Decomposition of time-series fMRI",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/arnauddhaene/nidmd",
     packages=setuptools.find_packages(),
     include_package_data=True,
     install_requires=['pandas', 'numpy', 'plotly', 'nilearn', 'matplotlib',
                       'scikit-learn', 'scipy', 'sklearn'],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
