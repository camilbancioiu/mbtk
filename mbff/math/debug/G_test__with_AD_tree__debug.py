from mbff.math.G_test__with_AD_tree import G_test


class G_test_debug(G_test):

    def use_preloaded_AD_tree(self, preloaded_AD_tree):
        super().use_preloaded_AD_tree(preloaded_AD_tree)
        if self.debug >= 2:
            self.AD_tree.debug = self.debug - 1
            self.AD_tree.debug_prepare__querying()
        if self.debug >= 1: print('Using preloaded AD-tree.')


    def load_AD_tree(self, adtree_load_path):
        if self.debug >= 1: print("Loading the AD-tree from {} ...".format(adtree_load_path))
        super().load_AD_tree(adtree_load_path)

        if self.debug >= 2:
            self.AD_tree.debug = self.debug - 1
            self.AD_tree.debug_prepare__querying()
        if self.debug >= 1: print('AD-tree loaded.')


    def build_AD_tree(self):
        if self.debug >= 1: print("Building the AD-tree...")
        super().build_AD_tree(self)
        if self.debug >= 1: print("AD-tree built in {:>10.4f}s".format(self.AD_tree_build_duration))


    def save_AD_tree(self):
        super().save_AD_tree()
        adtree_save_path = self.parameters.get('ci_test_ad_tree_path__save', None)
        if self.debug >= 1: print("AD-tree saved to", adtree_save_path)


    def G_test_conditionally_independent(self, X, Y, Z):
        result = super().G_test_conditionally_independent(X, Y, Z)

        try:
            if self.AD_tree.debug >= 1:
                result.extra_info = (
                    '\nAD-Tree:'
                    ' total of {n_pmf} contingency tables; currently {n_pmf_ll} contingency tables from leaf-lists;'
                    ' queries {n_queries},'
                    ' of which leaf-list queries {n_queries_ll}'
                ).format(**self.AD_tree.__dict__)
        except AttributeError:
            pass

        return result
