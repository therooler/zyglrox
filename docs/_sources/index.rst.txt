.. zyglrox documentation master file, created by
   sphinx-quickstart on Tue Nov 19 16:12:22 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Why zyglrox?
##################
At the moment, other quantum computation python frameworks lack the performance needed for quantum machine learning applications.
By building a simulator on top of Google's TensorFlow, we can easily utilize the GPU to speed up computations where necessary.
Additionally, ``zyglrox`` can easily be plugged into a wide range of deep learning models, enabling fast experimentation
with hybrid quantum machine learning models.

DISCLAIMER: This package is not a Google product and is in no way associated with `Tensorflow Quantum <https://www.tensorflow.org/quantum>`_,
Google's official Tensorflow Quantum Simulator. Additionally, ``zyglrox`` is based on TensorFlow 1.x, whereas Tensorflow Quantum is based on TensorFlow 2.x.

No I mean why did you name it zyglrox?
######################################
Well, because of reasons_ and because it contains x, y and z so it sounds techy.

.. _reasons: https://ask.fm/MishaPeriphery/answers/113713888747

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   source/installation

.. toctree::
   :maxdepth: 2
   :caption: zyglrox

   source/core
   source/models

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   source/tutorials

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

