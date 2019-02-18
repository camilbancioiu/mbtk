class DatasetMatrixNotFinalizedError(Exception):
    def __init__(self, datasetmatrix, attempt):
        self.datasetmatrix = datasetmatrix
        self.message = "DatasetMatrix not finalized. " + attempt
        super().__init__(self.message)



class DatasetMatrixFinalizedError(Exception):
    def __init__(self, datasetmatrix, attempt):
        self.datasetmatrix = datasetmatrix
        self.message = "DatasetMatrix already finalized. " + attempt
        super().__init__(self.message)



class ExperimentalDatasetFolderException(Exception):
    def __init__(self, definition, folder, message):
        self.exds_definition = definition
        self.folder = folder
        self.message = message
        super().__init__(self.message)
