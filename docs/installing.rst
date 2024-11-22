Installing
==========

Requirments
-----------

AFO is a Python based optimisation program. To run it requires access to Python and the necessary Python
packages, as well as a linear programming solver. There are a number of options available that satisfies these
requirements however only the most commonly used method is documented below.

#. Install Anaconda version 2023.09-0:

    - Download the installer (https://repo.anaconda.com/archive/).
    - Run the installer and follow the instructions.

#. Install Pyomo version 6.6.2: Pyomo is a Python package that is not installed with Anaconda by default.

    - Open Anaconda prompt which is located in start menu under the Anaconda folder.
    - Enter: ``pip install --force-reinstall -v "pyomo==6.6.2"``

    .. tip:: To install pyomo to a specific env:  conda install --name AFO -c conda-forge pyomo (pyomo = package, AFO = env)

#. Install numpy_financial:

    - Open Anaconda prompt which is located in start menu under the Anaconda folder.
    - Enter: ``pip install numpy_financial``

#. Install solver: AFO requires a linear programming solver to optimise the problem. For this HiGHS is used because it
   is the best open source solver available.

    - Open Anaconda prompt which is located in start menu under the Anaconda folder.
    - ``pip install highspy``


AFO access
----------

AFO is managed through Github which gives everyone the ability to use and develop the model. The following
process can be used to get the source code.

#. Create a GitHub account.

#. Install git.

    - https://git-scm.com/downloads

#. Clone AFO repository (this essentially downloads all the AFO source code onto your local device).

    - In the git terminal: ``git clone https://github.com/youngmr290/Australian-Farm-Optimising-Model.git``

#. If code changes are made, please create a pull request (via GitHub online) so that we can merge your changes into the master branch.