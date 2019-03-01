class ExperimentException(Exception):

    def __init__(self, definition, message):
        self.definition = definition
        self.message = message
        super().__init__(self.message)


class ExperimentFolderException(Exception):

    def __init__(self, definition, folder, message):
        self.experiment_definition = definition
        self.folder = folder
        self.message = message
        super().__init__(self.message)
