import numpy
import scipy

from mbff.datasets.sources.DatasetSource import DatasetSource
from mbff.datasets.DatasetMatrix import DatasetMatrix
import mbff.utilities as util

class RCV1v2DatasetSource(DatasetSource):
    """
    A class which reads the RCV1v2 dataset files, as published by LYRL2004.

    :var sourcefolder: The folder where the downloaded RCV1v2 files are located.
    :var sourcefile_documentIDs: The name of the file containing the list of all document IDs.
    :var sourcefile_document_tokens: The name of the file containing all the documents of the dataset, in token form.
    :var sourcefile_topic_assignments: The name of the file containing assignments of document IDs to topics.
    :var sourcefile_industry_assignments: The name of the file containing assignments of document IDs to industries.
    """

    def __init__(self, configuration):
        self.configuration = configuration
        self.sourcefolder = self.configuration['sourcefolder']

        self.sourcefile_documentIDs         = '{}/oa7.rcv1v2-ids.txt'.format(self.sourcefolder)
        self.sourcefile_document_tokens      = '{}/token/lyrl2004_tokens_all.dat'.format(self.sourcefolder)
        self.sourcefile_topic_assignments    = '{}/oa8.rcv1-v2.topics.qrels.txt'.format(self.sourcefolder)
        self.sourcefile_industry_assignments = '{}/oa9.rcv1-v2.industries.qrels.txt'.format(self.sourcefolder)


    def create_dataset_matrix(self, label='rcv1v2'):
        documentIDs = []
        if 'industry' in self.configuration['filters'].keys():
            documentIDs = self.read_documentIDs_in_industry(self.configuration['filters']['industry'])
        elif len(self.configuration['filters']) == 0:
            documentIDs = self.read_all_documentIDs()
        else:
            raise ValueError("Unsupported RCV1v2 document filter specified. Either specify \
                    the 'industry' filter or no filter at all.")

        documents = self.read_documents(documentIDs)
        words = self.gather_complete_word_list(documents)
        topics = self.gather_complete_topic_list(documents)

        dok_matrix_words, dok_matrix_topics = self.create_dok_matrices(documents, documentIDs, words, topics)

        datasetmatrix = DatasetMatrix(label)
        datasetmatrix.X = dok_matrix_words.tocsr()
        datasetmatrix.Y = dok_matrix_topics.tocsr()
        datasetmatrix.row_labels = list(map(str, documentIDs))
        datasetmatrix.column_labels_X = words
        datasetmatrix.column_labels_Y = topics

        return datasetmatrix


    def read_all_documentIDs(self):
        documentIDs = numpy.loadtxt(self.sourcefile_documentIDs, dtype='uint32').tolist()
        return sorted(documentIDs)


    def read_documentIDs_in_industry(self, industry):
        documentIDs = []
        with open(self.sourcefile_industry_assignments, 'r') as sourcefile:
            for line in sourcefile:
                industry_assignment = line.split()
                if industry_assignment[0] == industry:
                    documentIDs.append(int(industry_assignment[1]))
        return sorted(documentIDs)


    def gather_complete_word_list(self, documents):
        return sorted(list(set([word for document in documents.values() for word in document.words])))


    def gather_complete_topic_list(self, documents):
        return sorted(list(set([topic for document in documents.values() for topic in document.topics])))


    def create_dok_matrices(self, documents, documentIDs, words, topics):
        words_index = util.create_index(words)
        topics_index = util.create_index(topics)

        dok_matrix_words = scipy.sparse.dok_matrix((len(documents), len(words)), dtype=numpy.int32)
        dok_matrix_topics = scipy.sparse.dok_matrix((len(documents), len(topics)), dtype=numpy.int32)

        for row in range(0, len(documentIDs)):
            documentID = documentIDs[row]
            document = documents[documentID]

            if self.configuration['feature_type'] == 'wordcount':
                for word in document.words:
                    column = words_index[word]
                    dok_matrix_words[row, column] = document.word_frequencies[word]

            if self.configuration['feature_type'] == 'binary':
                for word in document.words:
                    column = words_index[word]
                    dok_matrix_words[row, column] = 1

            for topic in document.topics:
                column = topics_index[topic]
                dok_matrix_topics[row, column] = document.topic_values[topic]

        return (dok_matrix_words, dok_matrix_topics)


    def read_documents(self, requested_documentIDs):
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
        """
        line_words = line.split()
        for word in line_words:
            try:
                self.word_frequencies[word] += 1
            except KeyError:
                self.word_frequencies[word] = 1
        self.words = list(self.word_frequencies.keys())


    def add_topic(self, topic):
        self.topics.append(topic)
        self.topic_values[topic] = 1

