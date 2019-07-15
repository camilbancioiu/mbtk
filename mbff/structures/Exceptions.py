class ADTreeCannotDescend_LeafListNode(Exception):

    def __init__(self, adtree, values, leaf_list_node):
        self.adtree = adtree
        self.values = values
        self.leaf_list_node = leaf_list_node
        self.message = "Cannot descend into a leaf list node."
        super().__init__(self.message)



class ADTreeCannotDescend_ZeroCountNode(Exception):

    def __init__(self, adtree, values):
        self.adtree = adtree
        self.values = values
        self.message = "Cannot descend into a zero count node."
        super().__init__(self.message)



class ADTreeCannotDescend_MCVNode(Exception):

    def __init__(self, adtree, values, parent_node, next_values):
        self.adtree = adtree
        self.values = values
        self.parent_node = parent_node
        self.next_values = next_values
        self.message = "Cannot descend into an MCV node."
        super().__init__(self.message)
