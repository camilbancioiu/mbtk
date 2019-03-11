import numpy
import scipy

from mbff.dataset.sources.DatasetSource import DatasetSource
from mbff.dataset.DatasetMatrix import DatasetMatrix

import mbff.utilities.functions as util

class RCV1v2DatasetSource(DatasetSource):
    """
    A class which reads the RCV1v2 dataset files, as published by LYRL2004.

    :var dict configuration: The configuration dictionary received by the constructor, stored for later reference.
    :var str sourcefolder: The folder where the downloaded RCV1v2 files are located, read from ``configuration``.
    :var str sourcefile_documentIDs: The name of the file containing the list of all document IDs.
    :var str sourcefile_document_tokens: The name of the file containing all the documents of the dataset, in token form.
    :var str sourcefile_topic_assignments: The name of the file containing assignments of document IDs to topics.
    :var str sourcefile_industry_assignments: The name of the file containing assignments of document IDs to industries.
    """

    def __init__(self, configuration):
        self.configuration = configuration
        self.sourcefolder = self.configuration['sourcefolder']

        if not 'feature_type' in self.configuration.keys():
            self.configuration['feature_type'] = 'wordcount'

        if not 'filters' in self.configuration.keys():
            self.configuration['filters'] = {}

        self.sourcefile_documentIDs          = '{}/oa7.rcv1v2-ids.txt'.format(self.sourcefolder)
        self.sourcefile_document_tokens      = '{}/token/lyrl2004_tokens_all.dat'.format(self.sourcefolder)
        self.sourcefile_topic_assignments    = '{}/oa8.rcv1-v2.topics.qrels.txt'.format(self.sourcefolder)
        self.sourcefile_industry_assignments = '{}/oa9.rcv1-v2.industries.qrels.txt'.format(self.sourcefolder)


    def create_dataset_matrix(self, label='rcv1v2', feature_type=''):
        """
        Create a :class:`DatasetMatrix
        <mbff.dataset.DatasetMatrix.DatasetMatrix>` object containing a
        document-term matrix based on the documents in the RCV1v2 dataset
        (previously downloaded).

        If ``configuration['filters']`` has been defined, then only the
        documents that match the specified filters will be represented as rows
        of the returned ``DatasetMatrix`` object. Otherwise, all documents in
        RCV1v2 will be loaded.

        If ``configuration['feature_type'] == 'wordcount'``, then the ``X``
        matrix of the returned ``DatasetMatrix`` object will contain the counts
        of each word in every document.

        If ``configuration['feature_type'] == 'binary'``, then the ``X`` matrix
        of the returned ``DatasetMatrix`` object will contain only values of
        ``0`` and ``1``, indicating the absence and presence, respectively, of
        a word in a document. See the `Wikipedia article on document-term
        matrices`_ for more details.

        .. _Wikipedia article on document-term matrices:\
        https://en.wikipedia.org/wiki/Document-term_matrix

        :param str label: The label to be set on the returned ``DatasetMatrix`` instance.
        :return: A ``DatasetMatrix`` containing a document-term matrix in ``X`` and a class-assignment matrix in ``Y``.
        :rtype: mbff.dataset.DatasetMatrix.DatasetMatrix
        """
        documentIDs = []
        if 'industry' in self.configuration['filters'].keys():
            documentIDs = self.read_documentIDs_in_industry(self.configuration['filters']['industry'])
        elif len(self.configuration['filters']) == 0:
            documentIDs = self.read_all_documentIDs()
        else:
            raise ValueError("Unsupported RCV1v2 document filter specified. Either specify \
                    the 'industry' filter or no filter at all.")

        if feature_type == '':
            feature_type = self.configuration.get('feature_type', 'wordcount')

        documents = self.read_documents(documentIDs)
        words = self.gather_complete_word_list(documents)
        topics = self.gather_complete_topic_list(documents)

        dok_matrix_words, dok_matrix_topics = self.create_dok_matrices(documents, documentIDs, words, topics, feature_type)

        datasetmatrix = DatasetMatrix(label)
        datasetmatrix.X = dok_matrix_words.tocsr()
        datasetmatrix.Y = dok_matrix_topics.tocsr()
        datasetmatrix.row_labels = list(map(str, documentIDs))
        datasetmatrix.column_labels_X = words
        datasetmatrix.column_labels_Y = topics

        return datasetmatrix


    def read_all_documentIDs(self):
        """
        Return the sorted list of all document IDs defined in the downloaded RCV1v2
        files.

        :return: Sorted list of document IDs available in RCV1v2.
        :rtype: list
        """
        documentIDs = numpy.loadtxt(self.sourcefile_documentIDs, dtype='uint32').tolist()
        return sorted(documentIDs)


    def read_documentIDs_in_industry(self, industry):
        """
        Return the list of document IDs defined in the downloaded RCV1v2 which
        were categorized into the industry code ``industry``.

        :param str industry: An industry code, as defined by RCV1v2.
        """
        documentIDs = []
        with open(self.sourcefile_industry_assignments, 'r') as sourcefile:
            for line in sourcefile:
                industry_assignment = line.split()
                if industry_assignment[0] == industry:
                    documentIDs.append(int(industry_assignment[1]))
        return sorted(documentIDs)


    def gather_complete_word_list(self, documents):
        """
        Retrieve the entire vocabulary used by the ``documents`` received as
        argument.

        :param dict documents: A dictionary with document IDs as keys and :class:`RCV1v2Document` instances as values.
        :return: A sorted list of unique words used by all the provided ``documents``.
        :rtype: list
        """
        return sorted(list(set([word for document in documents.values() for word in document.words])))


    def gather_complete_topic_list(self, documents):
        """
        Retrieve all the topics to which the provided ``documents`` belong.

        :param dict documents: A dictionary with document IDs as keys and :class:`RCV1v2Document` instances as values.
        :return: A sorted list of all the unique topics to which the provided ``documents`` belong.
        :rtype: list
        """
        return sorted(list(set([topic for document in documents.values() for topic in document.topics])))


    def create_dok_matrices(self, documents, documentIDs, words, topics, feature_type):
        """
        Convert a collection of :class:`RCV1v2Document` objects into a pair
        of matrices: a document-term matrix and a class-assignment matrix
        (class = RCV1v2 topic).

        :param dict documents: A dictionary with document IDs as keys and :class:`RCV1v2Document` instances as values.
        :param list(int) documentIDs: The list of document IDs which should be added to the returned matrices.
        :param list(str) words: The list of unique words used by the provided ``documents``, as returned by :meth:`gather_complete_word_list`.
        :param list(str) topics: The list of unique topics to which the provided ``documents`` belong, as returned by :meth:`gather_complete_topic_list`.
        :param str feature_type: The type of values to generate for the document-term matrix (``'wordcount'`` or ``'binary'``).
        :return: A tuple containing the document-term matrix and class-assignment matrix.
        :rtype: tuple(scipy.sparse.dok_matrix, scipy.sparse.dok_matrix)
        """
        words_index = util.create_index(words)
        topics_index = util.create_index(topics)

        dok_matrix_words = scipy.sparse.dok_matrix((len(documents), len(words)), dtype=numpy.int32)
        dok_matrix_topics = scipy.sparse.dok_matrix((len(documents), len(topics)), dtype=numpy.int32)

        for row in range(0, len(documentIDs)):
            documentID = documentIDs[row]
            document = documents[documentID]

            if feature_type == 'wordcount':
                for word in document.words:
                    column = words_index[word]
                    dok_matrix_words[row, column] = document.word_frequencies[word]

            if feature_type == 'binary':
                for word in document.words:
                    column = words_index[word]
                    dok_matrix_words[row, column] = 1

            for topic in document.topics:
                column = topics_index[topic]
                dok_matrix_topics[row, column] = document.topic_values[topic]

        return (dok_matrix_words, dok_matrix_topics)


    def read_documents(self, requested_documentIDs):
        """
        Read the files of the RCV1v2 dataset and instantiate
        :class:`RCV1v2Document` objects corresponding to the documents
        specified by the list ``requested_documentIDs``.

        :param list(int) requested_documentIDs: The list of document IDs which should be read from the RCV1v2 files.
        :return: A dictionary with documentIDs as keys and instances of :class:`RCV1v2Document` objects as values.
        :rtype: dict
        """
        if len(requested_documentIDs) == 0:
            return {}

        documents = {}
        sourcefile_tokens = open(self.sourcefile_document_tokens, 'r')

        document = None
        for line in sourcefile_tokens.readlines():
            line = line.strip()

            # A line of the form '.I 000' marks the beginning of a new document
            # in the file, with the ID '000'.  If a previous document was being
            # read, it means we have reached its end, so we must save it to the
            # ``documents`` dictionary and start reading a new document.
            if line[0:3] == '.I ':
                # If a previous document was being read, save it to ``documents``.
                if document != None:
                    documents[document.did] = document

                # Read the document ID from '.I 000' (extract the 000) and see
                # if we are interested in it (i.e. is it in the argument
                # ``requested_documentIDs``). If we are interested in this new document ID,
                # then instantiate a new RCV1v2Document and set it to receive
                # the upcoming lines.
                documentID = int(line[3:].strip())
                if documentID in requested_documentIDs:
                    document = RCV1v2Document()
                    document.did = documentID
                    document.source = self
                else:
                    document = None
            # A line of the form '.W' does not interest us and is skipped.
            # Empty lines are also skipped.
            elif line[0:2] == '.W':
                continue
            elif line == '':
                continue
            # If the current line in the token file is neither of the form '.I 000',
            # nor '.W', nor an empty line, then it contains the actual
            # tokenized words of the current document. Feed this line to the
            # current document.
            else:
                if document != None:
                    document.read_line_with_tokens(line)

        # At the end of the file, add the last document to ``documents``.
        if document != None:
            documents[document.did] = document

        sourcefile_tokens.close()

        self.assign_topics_to_documents(documents)

        return documents


    def assign_topics_to_documents(self, documents):
        """
        Read the files of the RCV1v2 dataset to discover to what topics do the
        provided ``documents`` belong and update the document objects directly,
        with the topics each belongs to. See :class:`RCV1v2Document` for
        more details.

        :param dict documents: A dictionary with document IDs as keys and :class:`RCV1v2Document` instances as values.
        :return: Nothing
        """
        with open(self.sourcefile_topic_assignments, 'r') as sourcefile:
            for line in sourcefile:
                assignment = line.split()
                documentID = int(assignment[1])
                topic = assignment[0]
                try:
                    documents[documentID].add_topic(topic)
                except KeyError:
                    pass



class RCV1v2Document():
    """
    A simple class representing a single document retrieved from the RCV1v2 files.

    :var int did: The document ID.
    :var list(str) words: The list of unique words that appear in this document.
    :var dict word_frequencies: A dictionary with words as keys and integers as\
        values, representing the number of times each word appears in this\
        document.
    :var list(str) topics: The list of topics to which this document belongs.
    :var dict topic_values: A dictionary with topics as keys and integers as\
        values, currently only binary. If this document belongs to a topic, then\
        its corresponding value in this dictionary is ``1``, or ``0`` otherwise.
    :var RCV1v2DatasetSource source: The :class:`RCV1v2DatasetSource`\
        instance which retrieved and manages this document.
    """

    def __init__(self):
        self.did = 0
        self.words = []
        self.word_frequencies = {}
        self.topics = []
        self.topic_values = {}
        self.source = None


    def read_line_with_tokens(self, line):
        """
        Add a new line of words to this document, updating the word frequencies
        and the current list of unique words in the document.

        :param str line: A line containing tokens for this document, taken from RCV1v2 files.
        :return: Nothing
        """
        line_words = line.split()
        for word in line_words:
            try:
                self.word_frequencies[word] += 1
            except KeyError:
                self.word_frequencies[word] = 1
        self.words = list(self.word_frequencies.keys())


    def add_topic(self, topic):
        """
        Add a topic to the current document, updating the ``topics`` list directly.

        :param str topic: The identifier of a topic to which this document belongs.
        :return: Nothing
        """
        self.topics.append(topic)
        self.topic_values[topic] = 1

