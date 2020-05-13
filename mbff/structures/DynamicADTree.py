from mbff.structures.ADTree import ADTree, ADNode, VaryNode


class DynamicADTree(ADTree):

    def create(self):
        ADNodeClass = self.get_ADNode_class()
        self.root = ADNodeClass(self, -1, -1, row_selection=None, level=0)


    def get_ADNode_class(self):
        return DynamicADNode



class DynamicADNode(ADNode):

    def get_VaryNode_class(self):
        return DynamicVaryNode


    def create_Vary_children(self, tree):
        pass


    def get_Vary_child_for_column(self, column_index, tree):
        vary = super().get_Vary_child_for_column(column_index, tree)
        if vary is None:
            VaryNodeClass = self.get_VaryNode_class()
            vary = VaryNodeClass(tree, column_index, self.row_selection, level=self.level + 1)
            self.Vary_children.append(vary)
        return vary


    def __getstate__(self):
        return (self.count, self.column_index, self.value, self.leaf_list_node, self.Vary_children, self.row_selection)


    def __setstate__(self, state):
        (self.count, self.column_index, self.value, self.leaf_list_node, self.Vary_children, self.row_selection) = state




class DynamicVaryNode(VaryNode):

    def get_ADNode_class(self):
        return DynamicADNode
