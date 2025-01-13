Getting Started
================

AFO access
----------

Firstly, the AFO code must be obtained. The code base is managed through Github which gives everyone the ability 
to use and develop the model. The following process can be used to get the source code.

#. Create a GitHub account.

#. Install git.

    - https://git-scm.com/downloads

#. Clone AFO repository (this essentially downloads all the AFO source code onto your local device).

    - In the git terminal: ``git clone https://github.com/youngmr290/Australian-Farm-Optimising-Model.git``

#. If you make code changes, please create a pull request (via GitHub online) so that we can merge your changes into the master branch.


Requirments
-----------

AFO is a Python based optimisation program. To run it requires access to Python and the necessary Python
packages, as well as a linear programming solver. There are a number of options available that satisfies these
requirements however only the most commonly used method is documented below.

#. Install Python:

    - Download the installer https://www.python.org/downloads/.

#. Create a virtual environment (venv) inside the AFO directory (the following steps can be executed in your terminal):

    - go to the file location where AFO is: ``cd /path/to/AFO``
    - create the environment: ``python -m venv venv`` 
    - activate the environment: ``venv\Scripts\activate.bat`` (note sometimes the ".bat" extension is not required. Once successfull the terminal should show (venv) at the start.)
    - install the relevant packages: ``pip install -r requirements.txt``

#. Now you should be able to run AFO. Remember if you are running via VSCode or Pycharm make sure you set the Python interpreter inside this new environment.

