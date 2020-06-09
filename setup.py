import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='dmd',  
     version='0.1',
     scripts=['dmd'] ,
     author="Arnaud Dhaene",
     author_email="arnaud.dhaene@epfl.ch",
     description="Dynamic Mode Decomposition of time-series fMRI",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/arnauddhaene/dmd",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )

