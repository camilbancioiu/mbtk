import collections


class ContingencyTable:

    def __init__(self):
        self.ct = dict()
        self.columns = collections.deque()
        self.update_columns()


    def __str__(self):
        output = ''
        output += 'ContingencyTable'
        output += '\n'
        output += '-' * (1 + 5 * len(self.columns) + 9) + '\n'
        tablehead = '|' + ('{:^4}|' * len(self.columns)) + '| COUNT||' + '\n'
        output += tablehead.format(*self.columns)
        output += '-' * (1 + 5 * len(self.columns) + 9) + '\n'
        rowformat = '|' + ('{:^4}|' * len(self.columns))
        keys = sorted(self.ct.keys())
        for key in keys:
            if not isinstance(key, tuple):
                output += rowformat.format(key) + '|{:>6}||'.format(self.ct[key]) + '\n'
            else:
                output += rowformat.format(*key) + '|{:>6}||'.format(self.ct[key]) + '\n'
        output += '-' * (1 + 5 * len(self.columns) + 9)
        return output


    def duplicate(self):
        new_ct = ContingencyTable()
        new_ct.ct = self.ct.copy()
        new_ct.columns = collections.deque(self.columns)
        new_ct.update_columns()
        return new_ct


    def items(self):
        return self.ct.items()


    def get(self, key):
        return self.ct.get(key, 0)


    def finalize(self):
        self.remove_leftmost_column()
        return self


    def remove_leftmost_column(self):
        self.columns.popleft()
        new_ct = dict()
        for key, count in self.ct.items():
            new_key = key[1:]
            if len(new_key) == 1:
                new_key = new_key[0]
            new_ct[new_key] = count
        self.update_columns()
        self.ct = new_ct


    def set_column_order(self, new_order):
        print()
        print(self.columns, len(self.columns))
        if len(self.columns) < 2:
            return
        new_ct = dict()
        for key, count in self.ct.items():
            print()
            print('key', key)
            new_key = tuple([key[self.column_index[o]] for o in new_order])
            new_ct[new_key] = count
        self.columns = collections.deque(new_order)
        self.update_columns(sortcolumns=False)
        self.ct = new_ct


    def update_columns(self, sortcolumns=True):
        columns = self.columns
        if sortcolumns:
            columns = sorted(self.columns)
        column_index = dict()
        for index, column in enumerate(columns):
            column_index[column] = index
        self.column_index = column_index


    def __getitem__(self, key):
        try:
            return self.ct[key]
        except KeyError:
            return 0


    def prepend_column(self, column, value):
        self.columns.appendleft(column)
        new_ct = dict()
        for key, count in self.ct.items():
            new_key = (value,) + key
            new_ct[new_key] = count
        self.update_columns()
        self.ct = new_ct


    def append_column(self, column, value):
        self.columns.append(column)
        new_ct = dict()
        for key, count in self.ct.items():
            new_key = key + (value,)
            new_ct[new_key] = count
        self.update_columns()
        self.ct = new_ct


    def marginalize_column(self, column):
        index_in_ct = self.column_index[column]
        new_ct = dict()
        del self.columns[index_in_ct]
        for key, count in self.ct.items():
            new_key = key[:index_in_ct] + key[index_in_ct + 1:]
            try:
                new_ct[new_key] += count
            except KeyError:
                new_ct[new_key] = count
        self.update_columns()
        self.ct = new_ct


    def insert_column(self, column, value, position):
        new_ct = dict()
        if position < 0:
            position = len(self.columns) + position
        for key, count in self.ct.items():
            new_key = key[:position] + (value,) + key[position:]
            new_ct[new_key] = count
        self.columns.insert(position, column)
        self.update_columns()
        self.ct = new_ct


    def marginalize_last_column(self):
        self.columns.pop()
        new_ct = dict()
        for key, count in self.ct.items():
            new_key = key[:-1]
            try:
                new_ct[new_key] += count
            except KeyError:
                new_ct[new_key] = count
        self.update_columns()
        self.ct = new_ct


    def set_columns(self, columns):
        self.columns = collections.deque(columns)
        self.update_columns()


    def append(self, column, value, ct):
        if column is not None and value is not None:
            ct.prepend_column(column, value)
            columns_changed = False
            for c in ct.columns:
                if c not in self.columns:
                    self.columns.append(c)
                    columns_changed = True
            if columns_changed:
                self.update_columns()
        self.ct.update(ct.ct)
        return self


    def deduct(self, other_ct):
        for key, count in other_ct.items():
            self.ct[key] -= other_ct[key]
        return self


    def group_keys_by_columns(self, columns):
        column_indices = [self.column_index[c] for c in columns]
        complement_column_indices = [self.column_index[c] for c in self.columns if c not in columns]

        grouped_keys = dict()

        for key, count in self.items():
            subkey = tuple([key[i] for i in column_indices])
            complement_subkey = tuple([key[i] for i in complement_column_indices])
            try:
                grouped_keys[subkey][complement_subkey] = key
            except KeyError:
                grouped_keys[subkey] = {complement_subkey: key}

        return grouped_keys
