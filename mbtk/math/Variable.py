import operator
import itertools
from collections import Counter

from mbtk.math.Exceptions import VariableInstancesOfUnequalCount


class Variable:
    ID: int
    name: str

    def __init__(self, instances, name='unnamed'):
        self.ID = -1
        self.name = name
        self.instances_list = instances
        self.lazy_instances_loader = None
        self.values = None


    def __len__(self):
        return len(self.instances())


    def IDs(self) -> list[int]:
        return [self.ID]


    def instances(self):
        if self.instances_list is None:
            self.load_instances()
        return self.instances_list


    def load_instances(self):
        if self.lazy_instances_loader is not None:
            self.instances_list = self.lazy_instances_loader()


    def update_values(self):
        if self.instances is not None:
            self.values = sorted(list(Counter(self.instances()).keys()))



class Omega(Variable):

    def __init__(self, instance_count):
        super().__init__(None)
        self.ID = -1024
        self.name = 'Ω'
        self.values = [1]
        self.instance_count = instance_count


    def instances(self):
        return itertools.repeat(1, times=self.instance_count)


    def __len__(self):
        return self.instance_count



class JointVariables(Variable):

    def __init__(self, *variables):
        super().__init__(None)

        self.variables = self.flatten_variables_list(variables)
        self.ID = None
        self.name = '{' + ', '.join([var.name for var in self.variables]) + '}'
        # TODO get rid of self.variableIDs, and replace with self.ID everywhere?
        self.variableIDs = [var.ID for var in self.variables]

        if self.all_variables_have_instances():
            self.validate_variables()

        self.values = None


    def IDs(self) -> list[int]:
        varIDs = [var.IDs() for var in self.variables]
        return list(itertools.chain(*varIDs))


    def instances(self):
        if len(self.variables) == 1:
            return self.variables[0].instances()
        return zip(*[var.instances() for var in self.variables])


    def __len__(self):
        return len(self.variables[0])


    def all_variables_have_instances(self):
        for var in self.variables:
            if var.instances_list is None:
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
        validate_variable_lengths(self.variables)



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
        if self.source_variable.values is not None:
            self.values_index = dict(enumerate(self.source_variable.values))
            self.values = sorted(list(self.values_index.keys()))



def validate_variable_lengths(variables):
    lengths = [len(var) for var in variables]
    for i in range(len(lengths) - 1):
        if lengths[i] != lengths[i + 1]:
            raise VariableInstancesOfUnequalCount([variables[i], variables[i + 1]])
