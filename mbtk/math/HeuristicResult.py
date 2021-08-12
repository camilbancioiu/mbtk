import time

from mbtk.math.Variable import Variable, JointVariables, Omega


class HeuristicResult:

    def __init__(self):
        self.index = -1
        self.X = None
        self.Y = None
        self.Z = []
        self.heuristic = None
        self.heuristic_value = None
        self.extra_info = None
        self.start_time = 0
        self.end_time = 0
        self.duration = 0.0


    def set_variables(self, X, Y, Z):
        self.X = self.get_variable_representation(X)
        self.Y = self.get_variable_representation(Y)
        self.Z = self.get_variable_representation(Z)


    def start_timing(self):
        self.start_time = time.time()


    def end_timing(self):
        self.end_time = time.time()
        self.duration = (self.end_time - self.start_time)


    def set_heuristic(self, name, value):
        self.heuristic = name
        self.heuristic_value = value


    def get_variable_representation(self, variable):
        if variable is None:
            return '∅'

        if isinstance(variable, JointVariables):
            return self.get_variable_representation(variable.IDs())

        if isinstance(variable, Omega):
            return 'Ω'

        if isinstance(variable, Variable):
            if variable.ID == -1024:
                return 'Ω'

            return str(variable.ID)

        if isinstance(variable, int):
            if variable == -1:
                return 'unnamed'

            if variable == -1024:
                return 'Ω'

            return str(variable)

        if isinstance(variable, set) or isinstance(variable, list):
            if len(variable) == 0:
                return 'Ω'

            if len(variable) == 1:
                return str(list(variable)[0])

            return self.get_multiple_variables_representation(variable)

        return str(variable)


    def get_multiple_variables_representation(self, variables):
        return '{' + ', '.join([str(var) for var in variables]) + '}'
