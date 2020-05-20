import os
import time
from mbff.structures.ADTree import ADTree, ADNode, VaryNode


class ADTreeDebug(ADTree):

    def __init__(self, matrix, column_values, leaf_list_threshold=0, debug=0):
        self.debug = debug
        super().__init__(matrix, column_values, leaf_list_threshold=0)

    def get_ADNode_class(self):
        return ADNodeDebug


    def create(self):
        if self.debug >= 1:
            self.debug_prepare__building()
            self.debug_prepare__querying()

        super().create()

        if self.debug >= 2: print('ADTree created, duration: {}s'.format(self.duration))
        if self.debug >= 3:
            os.system('clear')


    def make_pmf(self, variables):
        if self.debug >= 2: print('ADTree.make_pmf: variables {}, in progress...'.format(variables))
        if self.debug >= 1: self.debug_reset_query_counts()
        if self.debug >= 1: self.n_pmf += 1
        if self.debug >= 2: start_time = time.time()

        pmf = super().make_pmf(variables)

        if self.debug >= 2: duration = time.time() - start_time
        if self.debug >= 2: print("...took {:.2f}s, done.".format(duration))

        return pmf


    def query_count_in_leaf_list_node(self, values, query_node):
        if self.debug >= 1: self.n_queries_ll += 1

        count = super().query_count_in_leaf_list_node(self, values, query_node)
        return count


    def debug_prepare__building(self):
        self.count_stats = dict()
        self.count_bins = 50
        for i in range(self.count_bins):
            self.count_stats[i] = 0
        self.sample_count = self.matrix.get_shape()[0]
        self.bin_size = int(self.sample_count / self.count_bins)
        self.leaf_list_nodes = 0
        self.topmost_list_length = 5
        self.topmost_nodes = [None] * (self.topmost_list_length + 1)


    def debug_prepare__querying(self):
        self.n_queries = 0
        self.n_queries_ll = 0
        self.n_pmf = 0
        self.n_pmf_ll = 0


    def debug_reset_query_counts(self):
        self.n_queries = 0
        self.n_queries_ll = 0
        self.n_pmf_ll = 0


    def debug_node(self, node):
        count_bin = -1
        if isinstance(node, ADNode):
            if node.count == self.sample_count:
                return
            count_bin = int(self.count_bins * node.count / self.sample_count)
            self.count_stats[count_bin] += 1

            if node.leaf_list_node:
                self.leaf_list_nodes += 1

        if self.debug == 2:
            if node.level <= self.topmost_list_length:
                self.topmost_nodes[node.level] = node
                for i in range(node.level + 1, len(self.topmost_nodes)):
                    self.topmost_nodes[i] = None
                print(self.debug_print_topmost_nodes())

        if self.debug >= 3:
            os.system('clear')
            print('Building AD-tree with LLT={}'.format(self.leaf_list_threshold))
            for i in range(self.count_bins):
                c = self.count_stats[i]
                print('{:>8} ({:2}): {}'.format(self.bin_size * (i + 1), i, c))
            print('AD-Node count', self.ad_node_count)
            print('Vary Node count', self.vary_node_count)
            print('Leaf list node count', self.leaf_list_nodes)
            if isinstance(node, ADNode):
                print("ADNode added to bin", count_bin)


    def debug_print_topmost_nodes(self):
        output = []
        for node in self.topmost_nodes:
            if node is not None:
                try:
                    node_string = 'AD[{}:{}]'.format(node.column_index, node.value)
                except AttributeError:
                    node_string = 'Vary[{}]'.format(node.column_index)
                output.append(node_string)

        return ' > '.join(output)


    def query_count(self, values, query_node=None):
        if self.debug >= 1:
            self.n_queries += 1

        return super().query_count(values, query_node)



class ADNodeDebug(ADNode):

    def get_VaryNode_class(self):
        return VaryNodeDebug


    def create_Vary_children(self, tree):
        if tree.debug >= 1:
            tree.debug_node(self)

        super().create_Vary_children(tree)


    def make_contingency_tree_from_leaf_list(self, tree, columns):
        ct = super().make_contingency_tree_from_leaf_list(tree, columns)

        if tree.debug >= 1:
            tree.n_pmf_ll += 1

        return ct



class VaryNodeDebug(VaryNode):

    def get_ADNode_class(self):
        return ADNodeDebug


    def create_AD_children(self, tree, row_subselections):
        if tree.debug >= 1:
            tree.debug_node(self)

        super().create_AD_children(tree, row_subselections)
