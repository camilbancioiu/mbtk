import math
import numpy
import scipy
import scipy.sparse
from pprint import pprint
from mpprint import mpprint
import operator

import utilities as util

dataset_path = './dataset_rcv1v2'
industry_name_file = dataset_path + '/' + 'oa5.rcv1.industries.hier.txt' 
toy_pair_did_list_file = dataset_path + '/' + 'toy_pair_did.txt'
test_did_list_file = dataset_path + '/' + 'testdid.txt'
full_did_list_file = dataset_path + '/' + 'oa7.rcv1v2-ids.txt'

did_list_file = test_did_list_file

def fetch_rcv1_token(industry = ''):
    rcv1_token = RCV1v2Dataset()
    
    if industry == '':
        rcv1_token.set_dids(read_did_list())
    else:
        rcv1_token.set_dids_by_industry(industry)

    rcv1_token.load_topic_assignments_to_documents()

    test_rcv1_token(rcv1_token)
    return rcv1_token

def trim_rcv1_token(r, freqs):
    r.remove_topics_with_freq_less_than(freqs[0])
    r.remove_topics_with_freq_greater_than(freqs[1])

def get_default_industry():
    return "I3302020"

def get_default_small_industry():
    return "I75100" # Space transport

def read_did_list():
    return numpy.loadtxt(did_list_file, dtype='uint32').tolist()

# Create dictionary of assignments between industries and documents.
# Form: {'industry': ['document1', 'document2', ... ]}
def create_industry_index():
    industries = {}
    with open(RCV1v2Dataset.industry_assignments_file, 'r') as f:
        for line in f:
            assignment = line.split()
            industry = assignment[0]
            document = assignment[1]
            try:
                industries[industry].append(int(document))
            except KeyError:
                industries[industry] = [int(document)]

    return industries

# Create dictionary of names of industries.
# Form: {'industryCode': 'industryName'}
def create_industry_name_index(index):
    names = {}
    with open(industry_name_file, 'r') as f:
        for line in f:
            for industry in index:
                if line.find(industry) != -1:
                    names[industry] = line[-line.find('cd: ') + 4:]

    return names
        
def get_industry_counts_between(mind, maxd):
    index = create_industry_index()
    names = create_industry_name_index(index)
    
    restricted_index = list(filter(lambda x: mind < len(index[x]) < maxd, index))
    return [(i, len(index[i]), names[i]) for i in restricted_index]

def create_industry_list_between(mind, maxd):
    industries = get_industry_counts_between(mind, maxd)
    industries = map(operator.itemgetter(0), industries)
    industries = list(map(str.strip, industries))
    return industries


## Component classes

class RCV1v2Document():
    def __init__(self):
        self.did = 0
        self.words = []
        self.word_frequencies = {}
        self.topics = []
        self.dataset = None

    def add_words_line(self, line):
        line_words = line.split()
        for word in line_words:
            try:
                self.word_frequencies[word] += 1
            except KeyError:
                self.word_frequencies[word] = 1
        self.words = list(self.word_frequencies.keys())

    def remove_topic(self, topic):
        self.topics = list(filter(lambda t: t != topic, self.topics))

    def words_to_csr(self):
        indices = [self.dataset.words_index[word] for word in self.words]
        data = [self.word_frequencies[word] for word in self.words]
        return (indices, data)

    def topics_to_csr(self):
        indices = [self.dataset.topics_index[topic] for topic in self.topics]
        data = [1 for topic in self.topics]
        return (indices, data)

class RCV1v2Dataset():
    dataset_path = './dataset_rcv1v2'
    document_tokens_file = dataset_path + '/token/' + 'lyrl2004_tokens_all.dat'
    topic_assignments_file = dataset_path + '/' + 'oa8.rcv1-v2.topics.qrels.txt'
    industry_assignments_file = dataset_path + '/' + 'oa9.rcv1-v2.industries.qrels.txt'

    def __init__(self):
        self.dids = []
        self.docs = {}
        self.docs_list = []
        self.docs_index = {}
        self.docs_lookup = {}
        self.words = []
        self.words_index = {}
        self.words_lookup = {}
        self.words_csr_freq = None
        self.words_csr = None
        self.topics = []
        self.topics_index = {}
        self.topics_lookup = {}
        self.topics_csr = None
        self.topic_frequencies = []

    def set_dids(self, dids):
        self.dids = sorted(dids)
        self.load_documents(self.dids)

    def set_dids_by_industry(self, industry, load=True):
        self.dids = []
        with open(self.industry_assignments_file, 'r') as f:
            for line in f:
                assignment = line.split()
                if assignment[0] == industry:
                    self.dids.append(int(assignment[1]))

        if load:
            self.load_documents(self.dids)

    def get_words_row(self, did):
        return self.words_csr.getrow(self.docs_index[did])

    def get_words_freq_row(self, did):
        return self.words_csr_freq.getrow(self.docs_index[did])

    def get_words_from_row(self, did):
        row = self.get_words_row(did)
        return sorted([self.words_lookup[col_index] for col_index in row.indices])

    def load_documents(self, dids):
        self.docs = {}
        f = open(self.document_tokens_file, 'r')
        document = None
        for line in f.readlines():
            line = line.strip()
            if line[0:3] == '.I ':
                if document != None:
                    self.docs[document.did] = document
                did = int(line[3:].strip())
                if did in dids:
                    document = RCV1v2Document()
                    document.did = did
                    document.dataset = self
                else:
                    document = None
            elif line[0:2] == '.W':
                continue
            elif line == '':
                continue
            else:
                if document != None:
                    document.add_words_line(line)
        if document != None:
            self.docs[document.did] = document

        self.update_docs()
        mpprint("Loaded {0} documents.".format(len(self.docs_list)))
        return self.docs

    def update_docs(self):
        self.docs_list = [self.docs[did] for did in self.dids]
        self.docs_index = util.create_index(self.dids)
        self.docs_lookup = util.create_lookup(self.dids)
        self.words = sorted(list(set([word for doc in self.docs_list for word in doc.words])))
        self.words_index = util.create_index(self.words)
        self.words_lookup = util.create_lookup(self.words)
        self.words_csr_freq = util.convert_to_csr([doc.words_to_csr() for doc in self.docs_list])
        self.words_csr = util.binarize_csr(self.words_csr_freq)
        mpprint("Updated document data.")

    def load_topic_assignments_to_documents(self):
        with open(self.topic_assignments_file, 'r') as f:
            for line in f:
                assignment = line.split()
                try:
                    self.docs[int(assignment[1])].topics.append(assignment[0])
                except KeyError:
                    pass
        self.update_topics()
        mpprint("Loaded {0} topics.".format(len(self.topics)))

    def update_topics(self):
        self.topics = sorted(list(set([topic for doc in self.docs_list for topic in doc.topics])))
        self.topics_index = util.create_index(self.topics)
        self.topics_lookup = util.create_lookup(self.topics)
        self.topics_csr = util.convert_to_csr([doc.topics_to_csr() for doc in self.docs_list])

        topics_csc = self.topics_csr.tocsc()
        cols = topics_csc.get_shape()[1]
        self.topic_frequencies = [numpy.sum(topics_csc.getcol(i).toarray()) for i in range(0, cols)]
        self.topic_frequencies_index = dict(zip(self.topics, self.topic_frequencies))

    def remove_topic(self, topic, update=True):
        for d in self.docs_list:
            d.remove_topic(topic)
            if len(d.topics) == 0:
                self.dids = list(filter(lambda did: did != d.did, self.dids))
        if update:
            self.update_docs()
            self.update_topics()

    def remove_topics_with_freq_less_than(self, f):
        f = math.ceil(len(self.docs_list) * f)
        topics_to_drop = list(filter(lambda t: self.topic_frequencies_index[t] < f, self.topics))
        for t in topics_to_drop:
            self.remove_topic(t, False)
        self.update_docs()
        self.update_topics()

    def remove_topics_with_freq_greater_than(self, f):
        f = math.ceil(len(self.docs_list) * f)
        topics_to_drop = list(filter(lambda t: self.topic_frequencies_index[t] > f, self.topics))
        for t in topics_to_drop:
            self.remove_topic(t, False)
        self.update_docs()
        self.update_topics()

    def lookup_selected_words(self, word_indices):
        return sorted(self.words_lookup[wi] for wi in word_indices)

    def get_stats(self):
        return str(len(self.docs_list)) + " docs"               \
               + ", on " + str(len(self.words)) + " words"      \
               + ", in " + str(len(self.topics)) + " topics"



## Tests

def test_rcv1_token(r):
    test_rcv1_token_loaded_documents(r)
    test_rcv1_token_document_token_count_in_csr(r)

def test_rcv1_token_loaded_documents(r):
    dids = r.dids
    documents = r.docs
    for did in dids:
        documents[did]

def test_rcv1_token_document_token_count_in_csr(r):
    dids = r.dids
    documents = r.docs
    for did in dids:
        doc = documents[did]

        # Test representation of frequencies in words_csr_freq
        row = r.get_words_freq_row(did)
        doc_tokens = sorted(doc.words)
        row_tokens = r.get_words_from_row(did)
        doc_sum = sum(doc.word_frequencies.values()) 
        row_sum = numpy.sum(row.toarray())
        if doc_sum != row_sum:
            raise Exception('Word freq in doc ' + str(did) + ' is improperly represented in CSR (row sum error).')
            print('doc: ' + str(doc_tokens[0:5]))
            print('row: ' + str(row_tokens[0:5]))
            print('row index: ' + str(r['docs_index'][did]))
        if doc_tokens != row_tokens:
            raise Exception('Word freq in doc ' + str(did) + ' is improperly represented in CSR (words do not match).')
            print('doc: ' + str(doc_tokens[0:5]))
            print('row: ' + str(row_tokens[0:5]))
            print('row index: ' + str(r['docs_index'][did]))

        # Test representation of presence/absence in words_csr
        row = r.get_words_row(did)
        doc_tokens = sorted(doc.words)
        row_tokens = r.get_words_from_row(did)
        doc_sum = len(doc.words)
        row_sum = numpy.sum(row.toarray())
        if doc_sum != row_sum:
            raise Exception('Word presence in doc ' + str(did) + ' is improperly represented in CSR (row sum error).')
            print('doc: ' + str(doc_tokens[0:5]))
            print('row: ' + str(row_tokens[0:5]))
            print('row index: ' + str(r['docs_index'][did]))
        if doc_tokens != row_tokens:
            raise Exception('Word presence in doc ' + str(did) + ' is improperly represented in CSR (words do not match).')
            print('doc: ' + str(doc_tokens[0:5]))
            print('row: ' + str(row_tokens[0:5]))
            print('row index: ' + str(r['docs_index'][did]))
