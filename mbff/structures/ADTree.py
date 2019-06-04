import collections


class ADTree:
    
    def __init__(self, matrix, column_values):
        self.matrix = matrix
        self.column_values = column_values
        self.root = ADNode(self, -1, '*', row_selection=None)


    def query(self, values, given={}):
        pass

    
    def query_count(self, values, given={}, query_node=None):
        if query_node is None:
            query_node = self.root

        if len(values) == 0:
            return query_node.count

        column_indices = sorted(list(values.keys()))
        column_index = column_indices[0]
        value = values[column_index]

        vary = query_node.Vary_children[column_index]
        node = vary.get_AD_child_with_value(value)
        



class ADNode:

    def __init__(self, tree, column_index, value, row_selection=None):
        self.tree = tree
        self.row_selection = row_selection
        if self.row_selection is None:
            self.count = self.tree.matrix.get_shape()[0]
        else:
            self.count = len(self.row_selection)
        self.column_index = column_index
        self.value = value
        self.Vary_children = []

        self.create_Vary_children()


    def create_Vary_children(self):
        column_count = self.tree.matrix.get_shape()[1]
        for column_index in range(self.column_index + 1, column_count):
            varyNode = VaryNode(self.tree, column_index, self.row_selection)
            self.Vary_children.append(varyNode)



class VaryNode:

    def __init__(self, tree, column_index, row_selection=None):
        self.tree = tree
        self.row_subselections = None
        self.row_selection = row_selection
        self.column_index = column_index
        self.AD_children = []

        self.create_row_subselections_by_value()
        self.values = self.tree.column_values[column_index]
        self.most_common_value = self.discoverMCV()
        self.create_AD_children()


    def create_AD_children(self):
        for value in self.values:
            row_selection = self.row_subselections[value]
            adNode = None
            if len(row_selection) > 0 and value != self.most_common_value:
                adNode = ADNode(self.tree, self.column_index, value, row_selection)
            self.AD_children.append(adNode)


    def create_row_subselections_by_value(self):
        """
        Group the rows of the self.tree.matrix by their value in the first column,
        but regardless of the rest of the columns. Only iterates over the rows
        found in row_selection.
        """

        self.row_subselections = collections.defaultdict(list)

        if self.row_selection is None:
            # If row_selection isn't set yet, initialize it to cover all the
            # rows.
            row_count = self.tree.matrix.get_shape()[0]
            self.row_selection = range(0, row_count)

        # Iterate over the rows in self.row_selection, putting each row_index
        # into a list corresponding to its value, in the self.row_subselections
        # dictionary.
        for row_index in self.row_selection:
            value = self.tree.matrix[row_index, self.column_index]
            self.row_subselections[value].append(row_index)


    def get_AD_child_with_value(self, value):
        for child in self.AD_children:
            if child is None:
                continue
            if child.value == value:
                return child
        return None


    def discoverMCV(self):
        return max(self.values, key=lambda v: len(self.row_subselections[v])) 

