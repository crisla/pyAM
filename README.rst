pyAM
====

|Build Status| |Coverage Status| |Codacy Badge| |GitHub License| |Latest Version| |Downloads| |DOI|

.. |Build Status| image:: https://travis-ci.org/davidrpugh/pyAM.svg?branch=master
   :target: https://travis-ci.org/davidrpugh/pyAM
.. |Coverage Status| image:: https://coveralls.io/repos/davidrpugh/pyAM/badge.svg?branch=master
   :target: https://coveralls.io/github/davidrpugh/pyAM?branch=master
.. |Codacy Badge| image:: https://www.codacy.com/project/badge/f051d7b5ccce47cfa3d6907c9a1bd6bf
   :target: https://www.codacy.com/app/drobert-pugh/pyAM
.. |GitHub license| image:: https://img.shields.io/github/license/davidrpugh/pyAM.svg
   :target: https://img.shields.io/github/license/davidrpugh/pyAM.svg
.. |Latest Version| image:: https://img.shields.io/pypi/v/pyAM.svg
   :target: https://pypi.python.org/pypi/pyAM/
.. |Downloads| image:: https://img.shields.io/pypi/dm/pyAM.svg
   :target: https://pypi.python.org/pypi/pyAM/
.. |DOI| image:: https://zenodo.org/badge/doi/10.5281/zenodo.20223.svg   
   :target: http://dx.doi.org/10.5281/zenodo.20223

Python package for solving assortative matching models with two-sided heterogeneity.

Installation
------------

Assuming you have `pip`_ on your computer (as will be the case if you've `installed Anaconda`_) you can install the latest stable release of ``pyam`` by typing
    
.. code:: bash

    $ pip install pyam

at a terminal prompt.

.. _pip: https://pypi.python.org/pypi/pip
.. _`installed Anaconda`: http://quant-econ.net/getting_started.html#installing-anaconda

Contributing
------------
If you wish to contribute to the project you will likely want to install from source. First your will need to fork and then clone the source repository.

.. code:: bash

    $ git clone https://github.com/YOUR-USERNAME/pyAM.git 

Next create a new `conda` development environment 

.. code:: bash
    
    $ conda create -n pyam-dev python anaconda

activate the newly created development environment

.. code:: bash

    $ source activate pyam-dev

and install additional dependencies not available within Anaconda.

.. code:: bash

    $ pip install pycollocation
    $ pip install seaborn

Finally, change into your local clone of the `pyam` source directory and install the package in development mode.

.. code:: bash

    $ pip install -e .


Example notebooks
-----------------
At the moment there are two example notebooks, one for `positive assortative matching`_ and one for `negative assortative matching`_ in the `examples` directory.  The positive assortative matching works fine; the negative assortative matching, however, does not yet work (I suspect because of a poor algorithm for the initial guess).

.. _`positive assortative matching`: https://github.com/davidrpugh/pyAM/blob/master/examples/positive-assortative-matching.ipynb
.. _`negative assortative matching`: https://github.com/davidrpugh/pyAM/blob/master/examples/negative-assortative-matching.ipynb
