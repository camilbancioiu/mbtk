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


    def sum_in_place(self, other):
        if self.children is None:
            self.count += other.count
            return

        for child_value in self.children:
            child = self.children[child_value]
            try:
                other_child = other.children[child_value]
                child.sum_in_place(other_child)
            except KeyError:
                pass


    def subtract_in_place(self, other):
        if self.children is None:
            self.count -= other.count
            return

        for child_value in self.children:
            child = self.children[child_value]
            try:
                other_child = other.children[child_value]
                child.subtract_in_place(other_child)
            except KeyError:
                pass
