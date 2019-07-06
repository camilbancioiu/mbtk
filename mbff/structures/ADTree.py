import time
import collections
import itertools
import os

from mbff.math.PMF import PMF, CPMF
from mbff.structures.Exceptions import ADTreeCannotDescend_MCVNode
from mbff.structures.Exceptions import ADTreeCannotDescend_LeafListNode
from mbff.structures.Exceptions import ADTreeCannotDescend_ZeroCountNode

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

    def __init__(self, matrix, column_values, leaf_list_threshold=0, debug_options=(False, False)):
        self.matrix = matrix
        self.column_values = column_values
        self.ad_node_count = 0
        self.vary_node_count = 0
        self.leaf_list_threshold = leaf_list_threshold
        self.start_time = 0
        self.end_time = 0
        self.duration = 0.0
        self.size = 0

        (self.debug, self.debug_to_stdout) = debug_options

        if self.debug:
            self.debug_prepare__building()
            self.debug_prepare__querying()
            self.start_time = time.time()
            self.root = ADNode(self, -1, '*', row_selection=None, level=0)
            if self.debug_to_stdout:
                os.system('clear')
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time
        else:
            self.root = ADNode(self, -1, '*', row_selection=None, level=0)


    def debug_prepare__building(self):
        self.count_stats = dict()
        self.count_bins = 50
        for i in range(self.count_bins):
            self.count_stats[i] = 0
        self.sample_count = self.matrix.get_shape()[0]
        self.bin_size = int(self.sample_count / self.count_bins)
        self.leaf_list_nodes = 0


    def debug_prepare__querying(self):
        self.n_queries = 0
        self.n_queries_ll = 0
        self.n_cpmf = 0


    def debug_reset_query_counts(self):
        self.n_queries = 0
        self.n_queries_ll = 0


    def debug_node(self, node):
        count_bin = -1
        if isinstance(node, ADNode):
            if node.count == self.sample_count:
                return
            count_bin = int(self.count_bins * node.count / self.sample_count)
            self.count_stats[count_bin] += 1

            if node.leaf_list_node:
                self.leaf_list_nodes += 1

        if self.debug_to_stdout:
            os.system('clear')
            for i in range(self.count_bins):
                c = self.count_stats[i]
                print('{:>8} ({:2}): {}'.format(self.bin_size * (i + 1), i, c))
            print('AD-Node count', self.ad_node_count)
            print('Vary Node count', self.vary_node_count)
            print('Leaf list node count', self.leaf_list_nodes)
            if isinstance(node, ADNode):
                print("ADNode added to bin", count_bin)


    def make_cpmf_2(self, variables, given=list(), node=None):
        result = None
        if self.debug: print('ADTree.make_cpmf_quick: variables {}, given {} in progress...'.format(variables, given))
        if self.debug: self.debug_reset_query_counts()
        if self.debug: self.n_cpmf += 1
        if self.debug: start_time = time.time()

        if node is None:
            node = self.root

        conditioning_values = list()
        for conditioning_variable in given:
            conditioning_values.append(self.column_values[conditioning_variable])

        variable_values = list()
        for variable in variables:
            variable_values.append(self.column_values[variable])

        if len(given) > 0:
            cpmf = CPMF(None, None)

            for cvalues in itertools.product(*conditioning_values):
                query_given = dict(zip(given, cvalues))
                try:
                    conditioning_node = self.find_node_for_values(query_given, node)
                    conditioning_count = conditioning_node.count

                    pmf = PMF(None)
                    for values in itertools.product(*variable_values):
                        count = self.query_count(values, conditioning_node)
                        p = count / conditioning_count
                        if len(values) == 1:
                            values = values[0]
                        pmf.probabilities[values] = p
                except ADTreeCannotDescend_LeafListNode as exLeafListNode:
                    # TODO create PMF here
                    pass
                except ADTreeCannotDescend_MCVNode as exMCVNode:
                    # TODO create PMF here
                    pass
                except ADTreeCannotDescend_ZeroCountNode as exZeroNode:
                    # TODO create PMF here
                    pass

                cpmf.conditional_probabilities[cvalues] = pmf

        if self.debug: duration = time.time() - start_time
        if self.debug: print("...took {:.2f}s, done.".format(duration))
        return result


    def find_node_for_values(self, values, current_node=None):
        if current_node is None:
            current_node = self.root

        if len(values) == 0:
            return current_node

        if current_node.leaf_list_node is True:
            raise ADTreeCannotDescend_LeafListNode(self, values, current_node)

        # Retrieve the column index we are currently on, within the tree, and
        # what value has been requested in the query for that column index.
        column_indices = sorted(list(values.keys()))
        column_index = column_indices[0]
        value = values[column_index]

        # Retrieve the Vary node among our immediate children that represents
        # the current column index.
        vary = current_node.get_Vary_child_for_column(column_index)

        # Retrieve the ADNode among the children of the aforementioned Vary
        # node which represents the value that was requested in the query for
        # the current column index.
        child = vary.get_AD_child_for_value(value)

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
            return self.find_node_for_values(next_values, child)
        else:
            if vary.most_common_value != value:
                # The ADNode is None, because of zero count.
                raise ADTreeCannotDescend_ZeroCountNode(self, values)
            else:
                raise ADTreeCannotDescend_MCVNode(self, values)


    def make_cpmf(self, variables, given=list()):
        if self.debug: print('ADTree.make_cpmf: variables {}, given {}'.format(variables, given), end=" ")
        if self.debug: self.debug_reset_query_counts()
        if self.debug: self.n_cpmf += 1
        if self.debug: start_time = time.time()

        conditioning_values = list()
        for conditioning_variable in given:
            conditioning_values.append(self.column_values[conditioning_variable])

        variable_values = list()
        for variable in variables:
            variable_values.append(self.column_values[variable])

        if len(given) > 0:
            cpmf = CPMF(None, None)

            for cvalues in itertools.product(*conditioning_values):
                query_given = dict(zip(given, cvalues))
                pmf = PMF(None)
                for values in itertools.product(*variable_values):
                    query_values = dict(zip(variables, values))
                    p = self.p(query_values, query_given)
                    if len(values) == 1:
                        values = values[0]
                    pmf.probabilities[values] = p

                if len(cvalues) == 1:
                    cvalues = cvalues[0]
                cpmf.conditional_probabilities[cvalues] = pmf

            result = cpmf
        else:
            pmf = PMF(None)
            for values in itertools.product(*variable_values):
                query_values = dict(zip(variables, values))
                p = self.p(query_values)
                if len(values) == 1:
                    values = values[0]
                pmf.probabilities[values] = p

            result = pmf

        if self.debug: duration = time.time() - start_time
        if self.debug: print("took {:.2f}s, done.".format(duration))
        return result


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
        if self.debug: self.n_queries += 1

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

        # Retrieve the Vary node among our immediate children that represents
        # the current column index.
        vary = query_node.get_Vary_child_for_column(column_index)

        # Retrieve the ADNode among the children of the aforementioned Vary
        # node which represents the value that was requested in the query for
        # the current column index.
        child = vary.get_AD_child_for_value(value)

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
        if self.debug: self.n_queries_ll += 1
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
        self.leaf_list_node = False
        self.Vary_children = []

        self.tree.ad_node_count += 1

        if self.count < self.tree.leaf_list_threshold:
            self.leaf_list_node = True

        if self.tree.debug:
            self.tree.debug_node(self)

        if not self.leaf_list_node:
            self.create_Vary_children()


    def create_Vary_children(self):
        column_count = self.tree.matrix.get_shape()[1]
        for column_index in range(self.column_index + 1, column_count):
            varyNode = VaryNode(self.tree, column_index, self.row_selection, level=self.level + 1)
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
        return "{}AD col{}={} ({}){}".format(INDENT * (self.level), self.column_index, self.value, self.count, children)


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

        self.tree.vary_node_count += 1

        if self.tree.debug:
            self.tree.debug_node(self)

        self.create_AD_children()


    def create_AD_children(self):
        for value in self.values:
            row_selection = self.row_subselections[value]
            adNode = None
            if len(row_selection) > 0 and value != self.most_common_value:
                adNode = ADNode(self.tree, self.column_index, value, row_selection, level=self.level + 1)
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
        return "{}Vary col{} (MCV={}){}".format(INDENT * (self.level), self.column_index, self.most_common_value, children)


    def children_to_string(self):
        rendered_children = []
        for value in self.values:
            if value != self.most_common_value:
                child = self.get_AD_child_for_value(value)
                if child is None:
                    rendered_children.append("{}AD col{}={} (0) NULL".format(INDENT * (self.level + 1), self.column_index, value))
                else:
                    rendered_children.append(str(child))
            else:
                rendered_children.append("{}AD col{}={} (MCV) NULL".format(INDENT * (self.level + 1), self.column_index, value))
        return "\n".join(rendered_children)
