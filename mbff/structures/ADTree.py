import time
import collections

from mbff.math.PMF import PMF
from mbff.structures.ContingencyTree import ContingencyTreeNode


def connect_AD_tree_classes():
    """
    Ensure the classes used by this AD-tree implementation reference each
    other properly. The mbff.structures package contains multiple AD-tree
    implementations that inherit the base ADTree class. Each of these
    implementations will have its own implementations for the ADNode and
    VaryNode classes, and they must reference each other correctly as well.

    This function is called after all the three required classes have been
    defined.
    """
    ADTree.ADNodeClass = ADNode
    ADNode.VaryNodeClass = VaryNode
    VaryNode.ADNodeClass = ADNode



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

    ADNodeClass = None

    def __init__(self, matrix, column_values, leaf_list_threshold=0):
        self.matrix = matrix
        self.column_cache = dict()
        self.column_values = column_values
        self.ad_node_count = 0
        self.vary_node_count = 0
        self.leaf_list_threshold = leaf_list_threshold
        self.start_time = 0
        self.end_time = 0
        self.duration = 0.0
        self.size = 0

        self.create()


    def create(self):
        self.start_time = time.time()
        self.root = self.ADNodeClass(self, -1, -1, row_selection=None, level=0)
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time


    def make_pmf(self, variables):
        variables = sorted(variables)
        joint_ct = self.root.make_contingency_table(self, variables)

        pmf = PMF(None)
        total_count = 1.0 * self.root.count
        for key, count in joint_ct.items():
            pmf.probabilities[key] = count / total_count

        pmf.variable = JointVariablesIDs(variables)

        return pmf


    def p(self, values, given=None):
        """An alias of :py:meth:query."""
        if given is None:
            given = dict()

        return self.query(values, given)


    def query(self, values, given=None):
        """
        Query the tree, requesting the joint or conditional probability of a
        specific combination of attributes-to-values, given a separate
        combination of attributes-to-values.
        """
        if given is None:
            given = dict()

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

        if query_node.leaf_list_node:
            return self.query_count_in_leaf_list_node(values, query_node)

        # Retrieve the column index we are currently on, within the tree, and
        # what value has been requested in the query for that column index.
        column_indices = sorted(list(values.keys()))
        column_index = column_indices[0]
        value = values[column_index]

        # Retrieve the Vary node among our immediate Vary children that
        # represents the current column index.
        vary = query_node.get_Vary_child_for_column(column_index, self)

        # Retrieve the ADNode among the children of the aforementioned Vary
        # node which represents the value that was requested in the query for
        # the current column index.
        child = vary.get_AD_child_for_value(value, self)

        # Prepare the query that will be passed down to the descendants, which
        # has the current column index removed from it (but otherwise identical).
        next_values = values.copy()
        next_values.pop(column_index)

        # We previously retrieved the ADNode that represents the current piece
        # of the query we're processing (i.e. current column index and its
        # value from the query). Now we must see whether this ADNode is None or
        # not, because 'None' has special meaning in an AD-tree.
        if child is not None:
            # The ADNode for the current value exists, query it deeper.
            return self.query_count(next_values, child)
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


    def query_count_in_leaf_list_node(self, values, query_node):
        count = 0
        for row_index in query_node.row_selection:
            match = True
            for column_index, value in values.items():
                if self.matrix[row_index, column_index] != value:
                    match = False
            if match:
                count += 1
        return count


    def __str__(self):
        return str(self.root)



class JointVariablesIDs:
    """
    JointVariablesIDs mimics the mbff.math.Variable.JointVariables class. There
    is only one property, variableIDs, which the method ADtree.make_pmf() will
    set to the list of variables for which it constructed the PMF out of the
    data in the AD-tree. This is required by the classes in DoFCalculators,
    which require the variable IDs to be present in PMF.variable.variableIDs.
    """
    def __init__(self, variableIDs):
        self.variableIDs = variableIDs



class ADNode:

    __slots__ = ('level', 'row_selection', 'count', 'column_index', 'value',
                 'leaf_list_node', 'Vary_children')

    VaryNodeClass = None

    def __init__(self, tree, column_index, value, row_selection=None, level=0):
        self.level = level
        self.row_selection = row_selection
        if self.row_selection is None:
            self.count = tree.matrix.get_shape()[0]
        else:
            self.count = len(self.row_selection)
        self.column_index = column_index
        self.value = value
        self.leaf_list_node = False
        self.Vary_children = []

        tree.ad_node_count += 1

        if self.count < tree.leaf_list_threshold:
            self.leaf_list_node = True

        self.create_Vary_children(tree)


    def create_Vary_children(self, tree):
        if self.leaf_list_node:
            return

        VaryNodeClass = self.VaryNodeClass
        column_count = tree.matrix.get_shape()[1]
        for column_index in range(self.column_index + 1, column_count):
            node = VaryNodeClass(tree, column_index, self.row_selection, level=self.level + 1)
            # TODO insert directly at a proper index, after having created
            # Vary_children with None elements?
            self.Vary_children.append(node)


    def get_Vary_child_for_column(self, column_index, tree):
        # Apparently, this `for` loop over the self.Vary_children list is faster
        # than having self.Vary_children as a dictionary and looking up values
        # with column_index as key. TODO do more detailed profiling
        for child in self.Vary_children:
            if child.column_index == column_index:
                return child
        return None


    def __getstate__(self):
        row_selection = None
        if self.leaf_list_node:
            row_selection = self.row_selection
        return (self.count, self.column_index, self.value, self.leaf_list_node, self.Vary_children, row_selection)


    def __setstate__(self, state):
        (self.count, self.column_index, self.value, self.leaf_list_node, self.Vary_children, self.row_selection) = state


    def make_contingency_table(self, tree, columns=None):
        if columns is None:
            columns = list()

        contingency_tree = self.make_contingency_tree(tree, columns)
        contingency_table = contingency_tree.convert_to_dictionary()
        return contingency_table


    def make_contingency_tree(self, tree, columns=None):
        if columns is None:
            columns = list()

        if len(columns) == 0:
            return ContingencyTreeNode(self.column_index, self.value, self.count)

        if self.leaf_list_node:
            return self.make_contingency_tree_from_leaf_list(tree, columns)

        non_mcv_ct = ContingencyTreeNode(self.column_index, self.value, None)
        next_column = columns[0]
        vary = self.get_Vary_child_for_column(next_column, tree)

        for value in vary.values:
            if value != vary.most_common_value:
                child = vary.get_AD_child_for_value(value, tree)
                if child is not None:
                    child_ct = child.make_contingency_tree(tree, columns[1:])
                    non_mcv_ct.append_child(child_ct)

        mcv_child_ct = self.make_contingency_tree(tree, columns[1:])
        mcv_child_ct.column = next_column
        mcv_child_ct.value = vary.most_common_value

        if non_mcv_ct.children is not None:
            for value, child in non_mcv_ct.children.items():
                mcv_child_ct.subtract_in_place(child)

        contingency_tree = non_mcv_ct
        contingency_tree.append_child(mcv_child_ct)

        return contingency_tree


    def make_contingency_tree_from_leaf_list(self, tree, columns):
        matrix = tree.matrix
        ct = ContingencyTreeNode(self.column_index, self.value, None)

        if len(columns) == 1:
            column = matrix.getcol(columns[0]).toarray()
            for row_index in self.row_selection:
                key = [column[row_index, 0]]
                ct.add_count_to_leaf(columns, key, 1)
        else:
            # TODO try to optimize this loop
            for row_index in self.row_selection:
                key = [matrix[row_index, column_index] for column_index in columns]
                ct.add_count_to_leaf(columns, key, 1)
        return ct



class VaryNode:

    __slots__ = ('level', 'row_selection', 'column_index',
                 'AD_children', 'values', 'most_common_value')

    ADNodeClass = None

    def __init__(self, tree, column_index, row_selection=None, level=0):
        self.level = level
        self.row_selection = row_selection
        self.column_index = column_index

        # TODO replace with a dict()? but dicts are huge
        self.AD_children = []

        row_subselections = self.create_row_subselections_by_value(tree)
        self.values = tree.column_values[column_index]
        self.most_common_value = self.discoverMCV(row_subselections)

        tree.vary_node_count += 1

        self.create_AD_children(tree, row_subselections)


    def create_AD_children(self, tree, row_subselections):
        ADNodeClass = self.ADNodeClass
        for value in self.values:
            row_selection = row_subselections[value]
            node = None
            if len(row_selection) > 0 and value != self.most_common_value:
                node = ADNodeClass(tree, self.column_index, value, row_selection, level=self.level + 1)
            # TODO insert directly at a proper index, after having created
            # AD_children with None elements?
            self.AD_children.append(node)


    def get_ADNode_class(self):
        return ADNode


    def __getstate__(self):
        return (self.column_index, self.values, self.AD_children, self.most_common_value)


    def __setstate__(self, state):
        (self.column_index, self.values, self.AD_children, self.most_common_value) = state


    def create_row_subselections_by_value(self, tree):
        """
        Group the rows of the matrix by their value in the first column,
        but regardless of the rest of the columns. Only iterates over the rows
        found in row_selection.
        """
        row_subselections = collections.defaultdict(list)

        if self.row_selection is None:
            # If row_selection isn't set yet, initialize it to cover all the
            # rows.
            row_count = tree.matrix.get_shape()[0]
            self.row_selection = range(0, row_count)

        # Iterate over the rows in self.row_selection, putting each row_index
        # into a list corresponding to its value, in the row_subselections
        # dictionary.
        try:
            column = tree.column_cache[self.column_index]
        except KeyError:
            column = tree.matrix.getcol(self.column_index).transpose().toarray().ravel()
            tree.column_cache[self.column_index] = column

        for row_index in self.row_selection:
            row_subselections[column[row_index]].append(row_index)

        return row_subselections


    def get_AD_child_for_value(self, value, tree):
        # Apparently, this `for` loop over the self.AD_children list is faster
        # than having self.AD_children as a dictionary and looking up values
        # with `value` as key. TODO do more detailed profiling
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


    def discoverMCV(self, row_subselections):
        return max(self.values, key=lambda v: len(row_subselections[v]))



connect_AD_tree_classes()
