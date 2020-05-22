from mbff.structures.ADTree import ADTree, ADNode, VaryNode


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
    DynamicADTree.ADNodeClass = DynamicADNode
    DynamicADNode.VaryNodeClass = DynamicVaryNode
    DynamicVaryNode.ADNodeClass = DynamicADNode



class DynamicADTree(ADTree):

    ADNodeClass = None

    def create(self):
        self.root = self.ADNodeClass(self, -1, -1, row_selection=None, level=0)



class DynamicADNode(ADNode):

    __slots__ = 'level'

    VaryNodeClass = None

    def create_Vary_children(self, tree):
        pass


    def get_Vary_child_for_column(self, column_index, tree):
        vary = super().get_Vary_child_for_column(column_index, tree)
        if vary is None:
            try:
                vary = self.VaryNodeClass(tree, column_index, self.row_selection, level=self.level + 1)
            except AttributeError:
                raise
            self.Vary_children.append(vary)
        return vary


    def __getstate__(self):
        return (self.count, self.column_index, self.value, self.leaf_list_node, self.Vary_children, self.row_selection, self.level)


    def __setstate__(self, state):
        (self.count, self.column_index, self.value, self.leaf_list_node, self.Vary_children, self.row_selection, self.level) = state



class DynamicVaryNode(VaryNode):

    __slots__ = 'level'

    ADNodeClass = None



connect_AD_tree_classes()
