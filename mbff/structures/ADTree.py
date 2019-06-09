import collections

INDENT = "|---"


class ADTree:
    """
    An implementation of the All-Dimensions tree (AD-tree), a data structure
    proposed by Andrew Moore and Mary Soon Lee in 1998, while affiliated with
    the Carnegie Mellon University, Pittsburgh.

    An AD-tree is a data structure that stores the sample counts within a
    dataset in an efficient manner. Building the AD-tree of a dataset removes
    the need to iterate over its samples everytime we want to calculate the
    joint or conditional probability distributions of the attributes / features
    / columns.

    After having the AD-tree, one can delete the dataset from memory and just
    query the tree to obtain any desired probability distribution.

    To save memory, an AD-tree can be built sparsely. As described by Moore and
    Lee themselves, large parts of an AD-tree can be reconstructed at
    query-time from other parts of the tree, thus greatly reducing its size.
    The sparsity of the AD-tree revolves around not allocating memory for nodes
    that would represent zero counts and for nodes that represent the most
    common value found in any attribute. The latter principle is the most
    effective, because normally, the nodes representing the most common values
    of any attribute would generate the largest subtrees (the subtrees least
    likely to reach zero-count leaves in few levels).

    The sparsity of an AD-tree comes with a compromise: querying becomes slower
    and more difficult to implement, due to the requirement to calculate
    missing values, which is done by performing multiple extra queries.
    """
    
    def __init__(self, matrix, column_values):
        self.matrix = matrix
        self.column_values = column_values
        self.root = ADNode(self, -1, '*', row_selection=None, level=0)


    def p(self, values, given={}):
        """An alias of :py:meth:query."""
        return self.query(values, given)


    def query(self, values, given={}):
        """
        Query the tree, requesting the joint or conditional probability of a
        specific combination of attributes-to-values, given a separate
        combination of attributes-to-values.
        """
        joint_query = {}
        joint_query.update(given)
        joint_query.update(values)
        conditioned_count = self.query_count(joint_query)

        conditioning_count = self.query_count(given)

        return 1.0 * conditioned_count / conditioning_count

    
    def query_count(self, values, query_node=None):
        """
        Query the tree, requesting the number of samples that the dataset has
        for a specific combination of attributes-to-values.
        """
        if query_node is None:
            query_node = self.root

        if len(values) == 0:
            return query_node.count

        # Retrieve the column index we are currently on, within the tree, and
        # what value has been requested in the query for that column index.
        column_indices = sorted(list(values.keys()))
        column_index = column_indices[0]
        value = values[column_index]

        # Retrieve the Vary node among our immediate children that represents
        # the current column index.
        vary = query_node.get_Vary_child_for_column(column_index)

        # Retrieve the ADNode among the children of the aforementioned Vary
        # node which represents the value that was requested in the query for
        # the current column index.
        node = vary.get_AD_child_for_value(value)

        # Prepare the query that will be passed down to the descendants, which
        # has the current column index removed from it (but otherwise identical).
        next_values = values.copy()
        next_values.pop(column_index)

        # We previously retrieved the ADNode that represents the current piece
        # of the query we're processing (i.e. current column index and its
        # value from the query). Now we must see whether this ADNode is None or
        # not, because 'None' has special meaning in an AD-tree.
        if node is not None:
            # The ADNode for the current value exists, query it deeper.
            return self.query_count(next_values, node)
        else:
            if vary.most_common_value != value:
                # The ADNode is None, because of zero count.
                return 0
            else:
                if len(next_values) == 0:
                    # The ADNode is None because it represents the most common
                    # value MCV, and there are no more values left to iterate
                    # down on. Being at the bottom of the tree, we can
                    # calculate the count it would have contained
                    return query_node.count - vary.sum_non_MCV_children_count()
                else:
                    # The ADNode is None because it represents the most common
                    # value MCV, but there are values left to iterate down on.
                    # Because we cannot recurse into None, we have to find out
                    # how many samples are there in the dataset when EXCLUDING
                    # the current column and value (which brought us to the
                    # None node): once among the siblings of the None node, and
                    # then among the nodes that are not involved with the
                    # current column and value. The difference of these two
                    # counts is the count that would have been stored in the
                    # None node.
                    nonMCVsiblings = vary.get_non_MCV_children()
                    query_count_in_siblings = 0
                    for sibling in nonMCVsiblings:
                        query_count_in_siblings += self.query_count(next_values, sibling)
                    query_count_in_parent = self.query_count(next_values, query_node)
                    return query_count_in_parent - query_count_in_siblings


    def __str__(self):
        return str(self.root)


class ADNode:

    def __init__(self, tree, column_index, value, row_selection=None, level=0):
        self.tree = tree
        self.level = level
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
            varyNode = VaryNode(self.tree, column_index, self.row_selection, level=self.level+1)
            self.Vary_children.append(varyNode)


    def get_Vary_child_for_column(self, column_index):
        for child in self.Vary_children:
            if child.column_index == column_index:
                return child
        return None

    
    def __str__(self):
        children = self.children_to_string()
        if len(children) > 0:
            children = "\n" + children
        return "{}AD col{}={} ({}){}".format(INDENT*(self.level), self.column_index, self.value, self.count, children)


    def children_to_string(self):
        rendered_children = []
        for child in self.Vary_children:
            rendered_children.append(str(child))
        return "\n".join(rendered_children)



class VaryNode:

    def __init__(self, tree, column_index, row_selection=None, level=0):
        self.tree = tree
        self.level = level
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
                adNode = ADNode(self.tree, self.column_index, value, row_selection, level=self.level+1)
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


    def get_AD_child_for_value(self, value):
        for child in self.AD_children:
            if child is None:
                continue
            if child.value == value:
                return child
        return None


    def get_non_MCV_children(self):
        return [child for child in self.AD_children if child is not None]


    def sum_non_MCV_children_count(self):
        return sum([child.count for child in self.AD_children if child is not None])


    def discoverMCV(self):
        return max(self.values, key=lambda v: len(self.row_subselections[v])) 


    def __str__(self):
        children = self.children_to_string()
        if len(children) > 0:
            children = "\n" + children
        return "{}Vary col{} (MCV={}){}".format(INDENT*(self.level), self.column_index, self.most_common_value, children)


    def children_to_string(self):
        rendered_children = []
        for value in self.values:
            if value != self.most_common_value:
                child = self.get_AD_child_for_value(value)
                if child is None:
                    rendered_children.append("{}AD col{}={} (0) NULL".format(INDENT*(self.level+1), self.column_index, value))
                else:
                    rendered_children.append(str(child))
            else:
                rendered_children.append("{}AD col{}={} (MCV) NULL".format(INDENT*(self.level+1), self.column_index, value))
        return "\n".join(rendered_children)

