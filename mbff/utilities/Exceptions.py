class LockablePathException(Exception):

    def __init__(self, lockablepath, folder, message):
        self.lockablepath = lockablepath
        self.folder = folder
        self.message = message
        super().__init__(self.message)




