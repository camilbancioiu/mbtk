import numpy
import operator
import itertools
from collections import Counter

from mbff.math.Exceptions import VariableInstancesOfUnequalCount


class Variable:

    def __init__(self, instances, name='unnamed'):
        self.ID = -1
        self.name = name
        self.instances = instances
        self.lazy_instances_loader = None
        self.values = None
        self.update_values()


    def load_instances(self):
        if not self.lazy_instances_loader is None:
            self.instances = self.lazy_instances_loader()
        self.update_values()


    def update_values(self):
        if not self.instances is None:
            self.values = sorted(list(Counter(self.instances).keys()))


    def simple_representation(self):
        if self.ID == -1:
            return self.name
        else:
            return self.ID



class Omega(Variable):

    def __init__(self, instance_count):
        super().__init__(None)
        self.ID = -1024
        self.name = 'Ω'
        self.instances = UniformInstances(1, instance_count)
        self.values = [1]


    def simple_representation(self):
        return self.name



class UniformInstances:

    def __init__(self, value, instance_count):
        self.value = value
        self.instance_count = instance_count


    def __iter__(self):
        return itertools.repeat(self.value, times=self.instance_count)


    def __len__(self):
        return self.instance_count



class JointVariables(Variable):

    def __init__(self, *variables):
        super().__init__(None)

        self.variables = self.flatten_variables_list(variables)
        self.ID = None
        self.name = '(' + ', '.join([var.name for var in self.variables]) + ')'
        self.variableIDs = [var.ID for var in self.variables]
        self.lazy_instances_loader = None

        self.lazy_instances_loader = self.lazy_joint_instances_loader

        if self.all_variables_have_instances():
            self.validate_variables()
            self.instances = self.lazy_joint_instances_loader()
            self.values = None
            self.update_values()
        else:
            self.instances = None
            self.values = None


    def simple_representation(self):
        return '(' + ', '.join([str(var.simple_representation()) for var in self.variables]) + ')'


    def lazy_joint_instances_loader(self):
        for variable in self.variables:
            variable.load_instances()
        self.validate_variables()
        if len(self.variables) == 1:
            return self.variables[0].instances
        return list(zip(*[var.instances for var in self.variables]))


    def all_variables_have_instances(self):
        for var in self.variables:
            if var.instances is None:
                return False
        return True


    def flatten_variables_list(self, variables):
        flattened_variables_list = []
        for variable in variables:
            if type(variable) == Variable:
                flattened_variables_list.append(variable)
            if type(variable) == JointVariables:
                flattened_variables_list.extend(variable.variables)
        return flattened_variables_list


    def validate_variables(self):
        validate_variable_instances_lengths(self.variables)



class SortedJointVariables(JointVariables):

    def __init__(self, *variables):
        variables = sorted(variables, operator.attrgetter("ID"))
        super().__init__(variables)



class IndexVariable(Variable):

    def __init__(self, variable):
        super().__init__(None)

        self.source_variable = variable

        # An IndexVariable is just a representation of a simple Variable or
        # JointVariables, and therefore can be used in place of the source
        # variable
        self.ID = self.source_variable.ID
        self.values = None
        self.values_index = None

        self.update_values()


    def update_values(self):
        if not self.source_variable.values is None:
            self.values_index = dict(enumerate(self.source_variable.values))
            self.values = sorted(list(self.values_index.keys()))



def validate_variable_instances_lengths(variables):
    lengths = [len(var.instances) for var in variables]
    for i in range(len(lengths) - 1):
        if lengths[i] != lengths[i+1]:
            raise VariableInstancesOfUnequalCount([variables[i], variables[i+1]])

