import zmq


class ADTreeClient():

    def __init__(self, server_address):
        self.socket = zmq.Context().socket(zmq.REQ)
        self.socket.connect(server_address)


    def query_count(self, query):
        self.socket.send_pyobj(('query_count', query))
        result = self.socket.recv_pyobj()
        return result


    def make_pmf(self, variables):
        self.socket.send_pyobj(('make_pmf', variables))
        result = self.socket.recv_pyobj()
        return result
