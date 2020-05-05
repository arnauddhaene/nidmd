# DynamicModeToolbox

## Installation

First, make sure you have a working installation of Python 3.7.0 (other versions should work as well but have not been tested).
Run ```python --version``` in your Terminal, which should return 

```
Python 3.7.0
```

The second step is installing dependencies. To do so in a safe manner, create a virtual environment at the base of the project and install all needed libraries in it

```
pip install virtualenv
virtualenv venv/
source venv/bin/activate
pip install -r requirements/base.txt
```

Assuming this installation runs without errors, you should be able to launch the Dynamic Mode Decomposition Dashboard using the following command

```
fbs run
```