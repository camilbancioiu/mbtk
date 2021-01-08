class LockablePathException(Exception):

    def __init__(self, lockablepath, folder, message):
        self.lockablepath = lockablepath
        self.folder = folder
        self.message = message
        super().__init__(self.message)



class CLICommandNotHandled(Exception):

    def __init__(self, command):
        self.command = command
        self.message = "CLI command {} not handled.".format(self.command)
        super().__init__(self.message)
