import numpy
import scipy

import tests.utilities as testutil

from mbtk.dataset.sources.RCV1v2DatasetSource import RCV1v2DatasetSource
from mbtk.dataset.DatasetMatrix import DatasetMatrix



def test_read_all_documentIDs():
    source = RCV1v2DatasetSource(default_configuration())
    all_documentIDs = source.read_all_documentIDs()
    assert default_all_documentIDs__as_int() == all_documentIDs



def test_read_documentIDs_in_industry():
    source = RCV1v2DatasetSource(default_configuration())
    for industry in ['FNG', 'ART', 'ASTRO', 'ENGN', 'MYTH']:
        expected = default_documentIDs_industry__as_int(industry)
        calculated = source.read_documentIDs_in_industry(industry)
        assert expected == calculated



def test_read_documents_words_topics():
    source = RCV1v2DatasetSource(default_configuration())
    documentIDs = source.read_all_documentIDs()

    documents = source.read_documents(documentIDs)
    assert 16 == len(documents)

    expected_document_IDs = sorted([document.did for document in documents.values()])
    assert default_all_documentIDs__as_int() == expected_document_IDs

    words = source.gather_complete_word_list(documents)
    assert default_words() == words

    topics = source.gather_complete_topic_list(documents)
    assert default_topics() == topics



def test_word_frequencies():
    source = RCV1v2DatasetSource(default_configuration())
    document = source.read_documents([302])[302]
    assert ['ascend', 'dopamine', 'galaxy', 'night', 'sonata'] == sorted(document.words)

    assert 6 == document.word_frequencies['ascend']
    assert 9 == document.word_frequencies['dopamine']
    assert 2 == document.word_frequencies['galaxy']
    assert 8 == document.word_frequencies['night']
    assert 5 == document.word_frequencies['sonata']



def test_generating_the_datasetmatrix__wordcount():
    configuration = default_configuration()

    source = RCV1v2DatasetSource(configuration)
    datasetmatrix = source.create_dataset_matrix('rcv1v2_test')

    expected_X = default_document_term_matrix()
    calculated_X = datasetmatrix.X
    assert DatasetMatrix.sparse_equal(expected_X, calculated_X) is True

    expected_Y = default_document_topic_matrix()
    calculated_Y = datasetmatrix.Y
    assert DatasetMatrix.sparse_equal(expected_Y, calculated_Y) is True

    assert default_all_documentIDs__as_row_labels() == datasetmatrix.row_labels
    assert default_words() == datasetmatrix.column_labels_X
    assert default_topics() == datasetmatrix.column_labels_Y



def test_generating_the_datasetmatrix__binary():
    configuration = default_configuration()
    configuration['feature_type'] = 'binary'

    source = RCV1v2DatasetSource(configuration)
    datasetmatrix = source.create_dataset_matrix('rcv1v2_test')

    expected_X = default_binary_document_term_matrix()
    calculated_X = datasetmatrix.X
    assert DatasetMatrix.sparse_equal(expected_X, calculated_X) is True

    expected_Y = default_document_topic_matrix()
    calculated_Y = datasetmatrix.Y
    assert DatasetMatrix.sparse_equal(expected_Y, calculated_Y) is True

    assert default_all_documentIDs__as_row_labels() == datasetmatrix.row_labels
    assert default_words() == datasetmatrix.column_labels_X
    assert default_topics() == datasetmatrix.column_labels_Y



def default_configuration():
    configuration = {
        'sourcepath': testutil.test_folder / 'rcv1v2_test_dataset',
        'filters': {},
        'feature_type': 'wordcount'
    }
    return configuration



def default_all_documentIDs__as_int():
    documentIDs = [
        101, 102, 103, 104,
        105, 106, 107, 108,
        201, 202,
        301, 302,
        401, 402, 403, 404]
    return documentIDs



def default_all_documentIDs__as_row_labels():
    documentIDs = [
        101, 102, 103, 104,
        105, 106, 107, 108,
        201, 202,
        301, 302,
        401, 402, 403, 404]
    return list(map(str, documentIDs))



def default_documentIDs_industry__as_int(industry):
    documentIDs = []
    if industry == 'FNG':
        documentIDs = [101, 102, 104, 106, 301, 302, 403]
    elif industry == 'ASTRO':
        documentIDs = [102, 104, 107, 202, 401]
    elif industry == 'ART':
        documentIDs = [101, 103, 104, 107, 108, 201, 202, 401, 404]
    elif industry == 'ENGN':
        documentIDs = [105, 201, 301, 401, 402, 403, 404]

    return documentIDs



def default_documentIDs_industry__as_row_labels(industry):
    documentIDs = []
    if industry == 'FNG':
        documentIDs = [101, 102, 104, 106, 301, 302, 403]
    elif industry == 'ASTRO':
        documentIDs = [102, 104, 107, 202, 401]
    elif industry == 'ART':
        documentIDs = [101, 103, 104, 107, 108, 201, 202, 401, 404]
    elif industry == 'ENGN':
        documentIDs = [105, 201, 301, 401, 402, 403, 404]

    return list(map(str, documentIDs))



def default_words():
    return sorted(['galaxy', 'almond', 'python', 'rocket', 'carbohydrate',
                   'oxygen', 'polyrhythm', 'firefly', 'dopamine', 'sonata',
                   'night', 'ascend', 'difference', 'session', 'rhapsody',
                   'cloud'])



def default_topics():
    return ['AR', 'ENC', 'SD', 'UNKN']



def default_document_term_matrix():
    # The columns are:
    # 0      1      2            3      4         5        6       7      8     9      10         11     12       13     14      15
    # almond ascend carbohydrate cloud difference dopamine firefly galaxy night oxygen polyrhythm python rhapsody rocket session sonata
    return scipy.sparse.csr_matrix(numpy.array([
        # 0  1  2  3  4  5  6  7  8  9 10 11 12  13 14 15
        [8, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 5, 0, 12, 2, 0],  # 101
        [10, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 4, 0, 3, 0, 0],  # 102
        [2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 0, 8, 0, 0],   # 103
        [10, 0, 0, 0, 0, 2, 0, 4, 0, 0, 0, 9, 0, 7, 0, 0],  # 104
        [10, 0, 0, 0, 0, 0, 0, 8, 0, 0, 2, 6, 0, 3, 0, 0],  # 105
        [10, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 4, 0, 6, 0, 0],  # 106
        [7, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 2, 5, 0, 0],   # 107
        [10, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 4, 0, 6, 0, 0],  # 108
        [0, 0, 10, 0, 2, 0, 8, 0, 0, 9, 5, 0, 0, 0, 0, 0],  # 201
        [0, 0, 4, 0, 0, 2, 7, 0, 0, 9, 5, 0, 0, 0, 0, 0],   # 202
        [0, 7, 0, 0, 0, 6, 0, 0, 5, 0, 0, 0, 0, 2, 0, 11],  # 301
        [0, 6, 0, 0, 0, 9, 0, 2, 8, 0, 0, 0, 0, 0, 0, 5],   # 302
        [0, 0, 0, 10, 6, 0, 0, 0, 0, 2, 0, 0, 9, 0, 5, 0],  # 401
        [0, 0, 0, 6, 4, 2, 0, 0, 0, 0, 0, 0, 6, 0, 8, 0],   # 402
        [0, 0, 0, 8, 3, 2, 0, 0, 0, 0, 0, 0, 12, 0, 6, 0],  # 403
        [2, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 6, 0, 6, 0]    # 404
    ]))



def default_binary_document_term_matrix():
    # The columns are:
    # 0      1      2            3      4         5        6       7      8     9      10         11     12       13     14      15
    # almond ascend carbohydrate cloud difference dopamine firefly galaxy night oxygen polyrhythm python rhapsody rocket session sonata
    return scipy.sparse.csr_matrix(numpy.array([
        # 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],   # 101
        [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],   # 102
        [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],   # 103
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],   # 104
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],   # 105
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],   # 106
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],   # 107
        [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],   # 108
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],   # 201
        [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],   # 202
        [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],   # 301
        [0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],   # 302
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],   # 401
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],   # 402
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],   # 403
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]    # 404
    ]))



def default_document_topic_matrix():
    # The columns are:
    # 0        1       2        3
    # arboreal encoded sidereal unknown
    return scipy.sparse.csr_matrix(numpy.array([
        # 0  1  2  3
        [1, 0, 0, 1],     # 101
        [1, 0, 0, 1],     # 102
        [1, 0, 0, 1],     # 103
        [1, 0, 0, 1],     # 104
        [1, 0, 1, 0],     # 105
        [1, 0, 1, 0],     # 106
        [1, 0, 1, 0],     # 107
        [1, 0, 1, 0],     # 108
        [0, 1, 1, 0],     # 201
        [0, 1, 1, 0],     # 202
        [0, 0, 1, 1],     # 301
        [0, 0, 1, 1],     # 302
        [1, 1, 0, 1],     # 401
        [1, 1, 0, 1],     # 402
        [1, 1, 0, 1],     # 403
        [1, 1, 0, 1]      # 404
    ]))
