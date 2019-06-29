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
        try:
            self.X = X.ID
            if self.X is None or self.X == -1:
                self.X = X.name
        except:
            self.X = X

        try:
            self.Y = Y.ID
            if self.Y is None or self.Y == -1:
                self.Y = Y.name
        except:
            self.Y = Y

        if Z is None:
            self.Z = '∅'
        else:
            try:
                self.Z = Z.ID
                if self.Z is None or self.Z == -1:
                    self.Z = Z.name
            except:
                self.Z = Z


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

        format_string = (
            "CI test {X:>3} ⊥ {Y:<3} | {Z:<10}: {i_or_d}"
            " @ {significance:6.4f}"
            " with {statistic}={statistic_value:<8.2f}"
            " at p={p_value:<9.6f} on the {test_distribution} distribution"
            )
        
        return format_string.format(**self.__dict__)


