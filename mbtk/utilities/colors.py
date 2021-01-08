HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


def green(text):
    return OKGREEN + str(text) + ENDC


def blue(text):
    return OKBLUE + str(text) + ENDC


def red(text):
    return RED + str(text) + ENDC


def yellow(text):
    return YELLOW + str(text) + ENDC


def bold(text):
    return BOLD + str(text) + ENDC


def underline(text):
    return UNDERLINE + str(text) + ENDC
