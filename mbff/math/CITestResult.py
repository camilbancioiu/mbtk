import time
import mbff.math.Variable

class CITestResult:

    def __init__(self):
        self.independent = None
        self.dependent = None
        self.X = None
        self.Y = None
        self.Z = []
        self.statistic = None
        self.statistic_value = None
        self.statistic_parameters = dict()
        self.test_distribution = None
        self.test_distribution_parameters = dict()
        self.p_value = None
        self.significance = None
        self.computed_d_separation = None

        self.start_time = 0
        self.end_time = 0
        self.duration = 0.0


    def __eq__(self, other):
        return  (
                self.independent == other.independent
            and self.dependent == other.dependent
            and self.X == other.X
            and self.Y == other.Y
            and self.Z == other.Z
            and self.statistic == self.statistic
            and self.statistic_value == self.statistic_value
            and self.statistic_parameters == self.statistic_parameters
            and self.test_distribution == self.test_distribution
            and self.test_distribution_parameters == self.test_distribution_parameters
            and self.p_value == self.p_value
            and self.significance == self.significance
            and self.computed_d_separation == self.computed_d_separation
            )


    def set_independent(self, independent, significance):
        self.independent = independent
        self.dependent = not independent
        self.significance = significance


    def set_dependent(self, dependent, significance):
        self.dependent = dependent
        self.independent = not dependent
        self.significance = significance


    def set_variables(self, X, Y, Z):
        self.X = self.get_variable_representation(X)
        self.Y = self.get_variable_representation(Y)
        self.Z = self.get_variable_representation(Z)


    def start_timing(self):
        self.start_time = time.time()


    def end_timing(self):
        self.end_time = time.time()
        self.duration = (self.end_time - self.start_time)


    def get_variable_representation(self, variable):
        if variable is None:
            return '∅'

        if type(variable) == int:
            if variable == -1:
                return 'unnamed'

            if variable == -1024:
                return 'Ω'

            return variable

        if isinstance(variable, set) or isinstance(variable, list):
            if len(variable) == 0:
                return '∅'

        try:
            return variable.simple_representation()
        except AttributeError:
            return str(variable)


    def set_statistic(self, name, value, params):
        self.statistic = name
        self.statistic_value = value
        self.statistic_parameters = params


    def set_distribution(self, name, p_value, params):
        self.test_distribution = name
        self.p_value = p_value
        self.test_distribution_parameters = params


    def __str__(self):
        self.i_or_d = ''
        if self.independent:
            self.i_or_d = 'I'
        else:
            self.i_or_d = 'D'

        d_sep_verification = ''
        if not self.computed_d_separation is None:
            if self.independent == self.computed_d_separation:
                d_sep_verification = '✔'
            else:
                d_sep_verification = '✘'
        self.i_or_d += d_sep_verification
        self.duration_in_seconds = self.duration / 1000

        format_string = (
            "CI test {X:>12} ⊥ {Y:<12} | {Z:<16}: {i_or_d}"
            " @ {significance:6.4f}"
            " with {statistic}={statistic_value:<8.2f}"
            " at p={p_value:<9.6f} on {test_distribution}"
            ", Δt={duration_in_seconds:>10.4f}s"
            )

        return format_string.format(**self.__dict__)


