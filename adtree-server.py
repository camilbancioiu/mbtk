import zmq
import time
import sys
import pickle

from pathlib import Path


def load_AD_tree(path):
    if path is None:
        return None

    print('Preloading AD-tree from', path)
    start = time.time()
    AD_tree = None
    try:
        with path.open('rb') as f:
            AD_tree = pickle.load(f)
    except FileNotFoundError:
        raise

    print('Duration: {:<.2f}s'.format(time.time() - start))
    return AD_tree



if __name__ == '__main__':
    socket = zmq.Context().socket(zmq.REP)
    socket.bind('tcp://127.0.0.1:8888')

    adtree_path = Path(sys.argv[1])
    print(adtree_path)

    adtree = load_AD_tree(adtree_path)

    while True:
        print()
        (request, arguments) = socket.recv_pyobj()
        print('Received request {} for {}'.format(request, arguments))
        if request == 'make_pmf':
            variables = arguments
            print('Preparing joint PMF...')

            pmf = adtree.make_pmf(variables)
            print('Done. It has {} keys. Sending...'.format(len(pmf)))

            socket.send_pyobj(pmf)
            print('Sent.')

        if request == 'query_count':
            print('Preparing query count...')
            query = arguments

            n = adtree.query_count(query)
            print('Done. Result is {}. Sending...'.format(n))

            socket.send_pyobj(n)
            print('Sent.')
