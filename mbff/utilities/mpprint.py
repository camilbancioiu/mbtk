import multiprocessing
import os
import sys

mpprint_lock = multiprocessing.Lock()
mpprint_format = 'P{pid:<6} :|\t{line}'

def mpprint(text=''):
    mpprint_lock.acquire()
    print(prepend_pid(text))
    sys.stdout.flush()
    mpprint_lock.release()


def prepend_pid(text):
    pid = os.getpid()
    return '\n'.join([mpprint_format.format(pid=pid, line=line) for line in text.splitlines()])
