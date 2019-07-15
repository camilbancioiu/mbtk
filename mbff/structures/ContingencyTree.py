import collections


class ContingencyTreeNode:

    def __init__(self, column, value, count):
        self.column = column
        self.value = value
        self.count = count
        self.children = None


    def __str__(self):
        as_tuple = (self.column, self.value, self.count)
        return str(as_tuple)


    def append_child(self, child):
        try:
            self.children[child.value] = child
        except TypeError:
            if self.children is None:
                self.children = {child.value: child}
            else:
                raise


    def add_count_to_leaf(self, columns, values, count):
        if len(values) == 0:
            try:
                self.count += count
            except TypeError:
                self.count = count
        else:
            try:
                next_child = self.get(values[0])
            except (TypeError, KeyError):
                next_child = ContingencyTreeNode(columns[0], values[0], None)
                self.append_child(next_child)
            next_child.add_count_to_leaf(columns[1:], values[1:], count)


    def sum_with_existing_child(self, child):
        self.children[child.value].sum(child)


    def sum_with_child(self, child):
        try:
            self.children[child.value].sum(child)
        except KeyError:
            self.children[child.value] = child
        except TypeError:
            if self.children is None:
                self.children = {child.value: child}
            else:
                raise


    def sum_children(self):
        sum_tree = ContingencyTreeNode(self.column, self.value, None)
        for value, child in self.children.items():
            for subvalue, subchild in child.children.items():
                sum_tree.sum_with_child(subchild)

        return sum_tree


    def get(self, value):
        return self.children[value]


    def convert_to_dictionary(self, key=None, dictionary=None):
        if key is None:
            key = collections.deque()

        if dictionary is None:
            dictionary = dict()

        if self.value != -1:
            key.append(self.value)

        if self.children is None:
            if len(key) > 1:
                dict_key = tuple(key)
            else:
                dict_key = key[0]
            dictionary[dict_key] = self.count
        else:
            for value, child in self.children.items():
                child.convert_to_dictionary(key, dictionary)
        try:
            key.pop()
        except IndexError:
            pass
        return dictionary


    def disallow_mismatching_node(self, other):
        if self.column != other.column or self.value != other.value:
            raise ValueError("Cannot operate on mismatching nodes: {} + {}".format(self, other))


    def disallow_mismatching_node_children(self, other):
        if (
            (not (self.children is None and other.children is None)) or
            (set(self.children.keys()) != set(other.children.keys()))
        ):
            raise ValueError((
                "Cannot operate on nodes with mismatching children:\n"
                "\tLeft: {} with children: {}\n"
                "\tRight: {} with children: {}\n"
            ).format(self, other, self.children, other.children))


    def sum(self, other):
        if self.children is None:
            return ContingencyTreeNode(self.column, self.value, self.count + other.count)

        sum_node = ContingencyTreeNode(self.column, self.value, None)
        for child_value in self.children:
            sum_child = self.children[child_value]
            try:
                other_child = other.children[child_value]
                sum_child = sum_child.sum(other_child)
            except KeyError:
                pass
            sum_node.append_child(sum_child)

        return sum_node


    def subtract(self, other):
        if self.children is None:
            return ContingencyTreeNode(self.column, self.value, self.count - other.count)

        sum_node = ContingencyTreeNode(self.column, self.value, None)
        for child_value in self.children:
            sum_child = self.children[child_value]
            try:
                other_child = other.children[child_value]
                sum_child = sum_child.subtract(other_child)
            except KeyError:
                pass
            sum_node.append_child(sum_child)

        return sum_node


    def __add__(self, other):
        self.disallow_mismatching_node(other)
        self.disallow_mismatching_node_children(other)

        return self.sum(other)



    def __sub__(self, other):
        self.disallow_mismatching_node(other)
        self.disallow_mismatching_node_children(other)

        return self.subtract(other)