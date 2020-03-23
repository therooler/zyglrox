Installation
============

.. note::

	In order to make use of the GPU, make sure that you have the correct prerequisites installed.
	``zyglrox`` relies on Tensorflow 1.15.0, which has the same requirements as 1.14.0:

	- **Python 3.3-3.7**

	- **CUDNN 7.4**

	- **CUDA 10.0**
	See the tensorflow `documentation <https://www.tensorflow.org/install/source#tested_build_configurations>`_ for more information.

Setup a Conda environment using the ``environment.yml`` file

   1. `Install Conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_

   2. Clone the git

   .. code-block:: bash

      git clone https://gitlab.com/rooler/zyglrox.git

   3. Create a virtual environment

   .. code-block:: bash

      conda env create -f environment.yml

   It is good to know that you can easily update the env

   .. code-block:: bash

      conda env update -f environment.yml

   To remove the environment use

   .. code-block:: bash

      conda remove --name qve --all

   4. Activate the environment as follows:

   .. code-block:: bash

      conda activate zyglrox

Finally, navigate to the ``zyglrox`` directory and run

.. code-block:: bash

   python setup.py install

If you care able to run the tutorials, you're good to go!