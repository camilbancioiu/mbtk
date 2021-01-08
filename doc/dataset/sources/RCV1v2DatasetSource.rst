===============================================
``RCV1v2DatasetSource``
===============================================

Before reading about this module further, it is recommended to understand
the :class:`mbtk.dataset.DatasetMatrix.DatasetMatrix` class.

About RCV1v2
------------

The RCV1v2 is a processed dataset based on the Reuters Corpus Volume 1,
thus it is a dataset of text documents. The RCV1v2 differs from the
original Reuters Corpus Volume 1 by representing each document as a list of
sorted stemmed tokens, and not as the natural language text of the Reuters
Corpus Volume 1 articles. This makes RCV1v2 a good starting point for creating
experimental datasets, but loses some of the information found in the original
Reuters articles. Refer to :cite:`lyrl2004` and :cite:`lyrl2004readme_online`
for more details about the RCV1v2 and its format.

A prerequisite of using this class is to download the RCV1v2 files. Use the
``download_rcv1v2.sh`` script bundle with MBFF, which will do it for you.
It will create a new folder named ``dataset_rcv1v2``, where it will
download and unpack the required files.


Reading RCV1v2 documents
------------------------

The :class:`RCV1v2DatasetSource <mbtk.dataset.sources.RCV1v2DatasetSource>`
class reads the files of the RCV1v2 package and constructs a collection of
documents by the "bag of words" principle. Namely, the document is represented
by the words that it contains and the number of occurences of each of its
words.

Instantiating a new :class:`RCV1v2DatasetSource
<mbtk.dataset.sources.RCV1v2DatasetSource>` requires a ``configuration``
dictionary passed to the constructor. This dictionary will affect all
subsequent calls to :meth:`create_dataset_matrix()
<mbtk.dataset.sources.RCV1v2DatasetSource.RCV1v2DatasetSource.create_dataset_matrix>`.
It may contain the following:

* ``configuration['sourcefolder']`` must contain the path to the folder
  containing the downloaded RCV1v2 files, as described above. This item
  must not be absent.
* ``configuration['filters']`` may contain a dictionary that specifies
  criteria by which documents from RCV1v2 should be imported or ignored.
  Currently, only the following possibilities are available:

  * Specifying no filter at all, namely ``configuration['filters'] = {}``
    or not setting ``configuration['filters']`` at all. This results in
    *all documents* being loaded.
  * Specifying a single industry filter , e.g.
    ``configuration['filters']['industry'] = 'I3302'``. See
    :cite:`lyrl2004` for a list of industries and what they mean.

* ``configuration['feature_type']`` must contain either one of the strings
  ``'wordcount'`` or ``'binary'``. If missing, it is automatically set to
  ``'wordcount'``. Its value determines what type of values will be written
  in the :class:`DatasetMatrix <mbtk.dataset.DatasetMatrix.DatasetMatrix>`
  object returned by :meth:`create_dataset_matrix
  <mbtk.dataset.sources.RCV1v2DatasetSource.RCV1v2DatasetSource.create_dataset_matrix>`.

After instantiating, call the :meth:`create_dataset_matrix()
<mbtk.dataset.sources.RCV1v2DatasetSource.RCV1v2DatasetSource.create_dataset_matrix>`
method to get a :class:`DatasetMatrix
<mbtk.dataset.DatasetMatrix.DatasetMatrix>` object, which contains the
documents in RCV1v2 represented as a `document-term matrix`_, as follows:

  .. _document-term matrix: https://en.wikipedia.org/wiki/Document-term_matrix

* The ``X`` matrix represents each document as a row and each token (word)
  as a column. Each cell of the matrix contains the count in the word
  corresponding to its column, in the document corresponding to its row,
  assuming ``configuration['feature_type'] == 'wordcount'``. On the other
  hand, if ``configuration['feature_type'] == 'binary'``, then the cells of
  the matrix will contain either ``0`` or ``1``, representing the absence or
  presence of a word in a document.
* The ``Y`` matrix represents each document as a row as well, but the
  columns represent the classes to which a document may belong. The cells of
  this matrix thus contain binary values (``1`` if the corresponding document
  belongs to the class corresponding to the column, ``0`` otherwise).
* The ``row_labels`` list will contain the numeric IDs of the loaded documents,
  in order, where each document ID in ``row_labels`` corresponds to a row in
  ``X`` and a row in ``Y``, at the same position.
* The ``column_labels_X`` list will contain the tokens corresponding to the
  columns in ``X`` (the features), in order.
* The ``column_labels_Y`` list will contain the topics corresponding to the
  columns in ``Y`` (the objective variables, or classes), in order.

Examples
--------

.. sourcecode:: python

  from pathlib import Path
  from mbtk.dataset.sources.RCV1v2DatasetSource import RCV1v2DatasetSource

  configuration = {
          'sourcefolder': Path('/var/work/Personal/PhD/Datasets/Reuters/RCV1-v2'),
          'filters': { },
          'feature_type': 'binary'
          }
  source = RCV1v2DatasetSource(configuration)
  # TODO

Module :mod:`mbtk.dataset.sources.RCV1v2DatasetSource`
------------------------------------------------------

.. automodule:: mbtk.dataset.sources.RCV1v2DatasetSource
  :members:
  :undoc-members:

References
----------
.. bibliography:: references.bib

