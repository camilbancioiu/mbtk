import time

import mbtk.utilities.colors as col


class CITestResult:

    def __init__(self):
        self.index = -1
        self.independent = None
        self.dependent = None
        self.insufficient_samples = None
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
        self.extra_info = None

        self.tolerance__statistic_value = 1e-11
        self.tolerance__p_value = 1e-11

        self.start_time = 0
        self.end_time = 0
        self.duration = 0.0


    def __eq__(self, other):
        return (
            self.independent == other.independent and
            self.dependent == other.dependent and
            self.insufficient_samples == other.insufficient_samples and
            self.X == other.X and
            self.Y == other.Y and
            self.Z == other.Z and
            self.statistic == other.statistic and
            abs(self.statistic_value - other.statistic_value) <= self.tolerance__statistic_value and
            self.statistic_parameters == other.statistic_parameters and
            self.test_distribution == other.test_distribution and
            self.test_distribution_parameters == other.test_distribution_parameters and
            abs(self.p_value - other.p_value) <= self.tolerance__p_value and
            self.significance == other.significance and
            self.computed_d_separation == other.computed_d_separation
        )


    def diff(self, other):
        if self.insufficient_samples != other.insufficient_samples:
            return 'Differring \'insufficient_samples\': {} vs {}'.format(self.insufficient_samples, other.insufficient_samples)
        if self.independent != other.independent:
            return 'Differring \'independent\': {} vs {}'.format(self.independent, other.independent)
        if self.dependent != other.dependent:
            return 'Differing \'dependent\': {} vs {}'.format(self.dependent, other.dependent)
        if self.X != other.X:
            return 'Differing \'X\': {} vs {}'.format(self.X, other.X)
        if self.Y != other.Y:
            return 'Differing \'Y\': {} vs {}'.format(self.Y, other.Y)
        if self.Z != other.Z:
            return 'Differing \'Z\': {} vs {}'.format(self.Z, other.Z)
        if self.statistic != other.statistic:
            return 'Differing \'statistic\': {} vs {}'.format(self.statistic, other.statistic)
        if abs(self.statistic_value - other.statistic_value) > self.tolerance__statistic_value:
            return 'Differing \'statistic_value\': {} vs {}'.format(self.statistic_value, other.statistic_value)
        if self.statistic_parameters != other.statistic_parameters:
            return 'Differing \'statistic_parameters\': {} vs {}'.format(self.statistic_parameters, other.statistic_parameters)
        if self.test_distribution != other.test_distribution:
            return 'Differing \'test_distribution\': {} vs {}'.format(self.test_distribution, other.test_distribution)
        if self.test_distribution_parameters != other.test_distribution_parameters:
            return 'Differing \'test_distribution_parameters\': {} vs {}'.format(self.test_distribution_parameters, other.test_distribution_parameters)
        if abs(self.p_value - other.p_value) > self.tolerance__p_value:
            return 'Differing \'p_value\': {} vs {}'.format(self.p_value, other.p_value)
        if self.significance != other.significance:
            return 'Differing \'significance\': {} vs {}'.format(self.significance, other.significance)
        if self.computed_d_separation != other.computed_d_separation:
            return 'Differing \'computed_d_separation\': {} vs {}'.format(self.computed_d_separation, other.computed_d_separation)


    def set_independent(self, independent, significance):
        self.independent = independent
        self.dependent = not independent
        self.significance = significance
        self.insufficient_samples = None


    def set_dependent(self, dependent, significance):
        self.dependent = dependent
        self.independent = not dependent
        self.significance = significance
        self.insufficient_samples = None


    def set_insufficient_samples(self):
        self.insufficient_samples = True
        self.independent = None
        self.dependent = None
        self.significance = None
        self.statistic_value = -1
        self.p_value = -1


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
                return 'Ω'

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


    def accurate(self):
        if self.computed_d_separation is not None:
            return self.independent == self.computed_d_separation
        else:
            return None


    def __str__(self):
        if not self.insufficient_samples:
            (view, format_string) = self.render__sufficient_samples()
        else:
            (view, format_string) = self.render__insufficient_samples()

        output = format_string.format(**view)
        if hasattr(self, 'extra_info'):
            if self.extra_info is not None:
                output += self.extra_info

        return output


    def render__sufficient_samples(self):
        view = dict()
        view.update(self.__dict__)
        view['startcode'] = ''
        view['endcode'] = ''

        view['i_or_d'] = ''
        if self.independent:
            view['i_or_d'] = 'I'
        else:
            view['i_or_d'] = 'D'

        d_sep_verification = ''
        accurate = self.accurate()
        if accurate is not None:
            if accurate:
                d_sep_verification = '✔'
            else:
                d_sep_verification = '✘'
                view['startcode'] = col.RED
                view['endcode'] = col.ENDC
        view['i_or_d'] += d_sep_verification
        view['duration_in_seconds'] = self.duration
        view['index'] = self.index

        format_string = (
            "{startcode}"
            "CI test {index}: {X:>4} ⊥ {Y:<4} | {Z:<20}: {i_or_d}"
            " @ {significance:6.4f}"
            " with {statistic}={statistic_value:>8.6f}"
            " at p={p_value:<8.6f} on {test_distribution}"
            ", Δt={duration_in_seconds:>10.4f}s"
            "{endcode}"
        )

        return (view, format_string)


    def render__insufficient_samples(self):
        view = dict()
        view.update(self.__dict__)
        view['startcode'] = ''
        view['endcode'] = ''
        view['i_or_d'] = 'INSUFFICIENT SAMPLES'
        view['startcode'] = col.YELLOW
        view['endcode'] = col.ENDC
        view['duration_in_seconds'] = self.duration
        view['index'] = self.index

        format_string = (
            "{startcode}"
            "CI test {index}: {X:>4} ⊥ {Y:<4} | {Z:<20}: {i_or_d}"
            ", Δt={duration_in_seconds:>10.4f}s"
            "{endcode}"
        )

        return (view, format_string)
