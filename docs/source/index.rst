DstParser Documentation
========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Introduction
------------

DstParser is a tool designed to parse and process DST (Data Summary Tape) files, commonly used in cosmic ray experiments.
This documentation provides an overview of the installation, usage, and API of DstParser.

.. note::
   This project is under active development. Contributions are welcome!

Installation
------------

To install DstParser, use the following command:

.. code-block:: bash

   pip install dstparser

Usage
-----

Here is an example of how to use DstParser:

.. code-block:: python

   from dstparser import DstParser

   parser = DstParser("path/to/dstfile.dst")
   data = parser.parse()
   print(data)

API Reference
-------------

The API reference can be found below:

.. automodule:: dstparser
   :members:
   :undoc-members:
   :show-inheritance:

License
-------

DstParser is released under the MIT License.


