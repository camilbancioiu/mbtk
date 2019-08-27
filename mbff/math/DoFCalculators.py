import collections


class UnadjustedDoF:

    def __init__(self, G_test):
        self.requires_pmfs = False
        self.requires_cpmfs = False

        self.G_test = G_test

        if self.G_test is not None:
            self.parameters = self.G_test.parameters
            self.datasetmatrix = self.G_test.datasetmatrix
            self.matrix = self.G_test.matrix
            self.column_values = self.G_test.column_values
            self.N = self.G_test.N

        self.debug_values = dict()

        self.reset()


    def reset(self):
        self.PrXYcZ = None
        self.PrXcZ = None
        self.PrYcZ = None
        self.PrZ = None
        self.PrXYZ = None
        self.PrXZ = None
        self.PrYZ = None
        self.PrZ = None
        self.X = None
        self.Y = None
        self.Z = None


    def set_context_variables(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z


    def set_context_cpmfs(self, PrXYcZ, PrXcZ, PrYcZ, PrZ):
        self.PrXYcZ = PrXYcZ
        self.PrXcZ = PrXcZ
        self.PrYcZ = PrYcZ
        self.PrZ = PrZ


    def set_context_pmfs(self, PrXYZ, PrXZ, PrYZ, PrZ):
        self.PrXYZ = PrXYZ
        self.PrXZ = PrXZ
        self.PrYZ = PrYZ
        self.PrZ = PrZ


    def calculate_DoF(self, X, Y, Z):
        X_val = len(self.column_values[X])
        Y_val = len(self.column_values[Y])

        Z_val = 1
        for z in Z:
            Z_val *= len(self.column_values[z])
        DoF = (X_val - 1) * (Y_val - 1) * Z_val

        if DoF == 0:
            DoF = 1

        return DoF



class StructuralDoF(UnadjustedDoF):

    def __init__(self, G_test):
        super().__init__(G_test)
        self.requires_pmfs = False
        self.requires_cpmfs = True

    def calculate_DoF(self, X, Y, Z):
        dkey = frozenset({X} | {Y} | set(Z))
        self.debug_values[dkey] = dict()

        DoF = 0
        for (z, pz) in self.PrZ.items():
            if pz == 0:
                continue

            PrX = self.PrXcZ.given(z)
            PrY = self.PrYcZ.given(z)

            X_val = len(list(filter(None, PrX.values())))
            Y_val = len(list(filter(None, PrY.values())))

            self.debug_values[dkey][z] = (X_val, Y_val)

            structural_dof_xycz = (X_val - 1) * (Y_val - 1)
            DoF += structural_dof_xycz

        if DoF == 0:
            DoF = 1

        return DoF



class CachedStructuralDoF(UnadjustedDoF):

    def __init__(self, G_test):
        super().__init__(G_test)
        self.DoF_cache = dict()
        self.requires_pmfs = True
        self.requires_cpmfs = False


    def set_context_pmfs(self, PrXYZ, PrXZ, PrYZ, PrZ):
        # All PMFs received as arguments are expected to be PMFs of
        # JointVariables. If PrZ is the PMF of a single variable, it is skipped
        # from caching.
        if PrXYZ is not None:
            self.cache_DoFs_for_pmf(PrXYZ, PrXYZ.variable.variableIDs)
        if PrXZ is not None:
            self.cache_DoFs_for_pmf(PrXZ, PrXZ.variable.variableIDs)
        if PrYZ is not None:
            self.cache_DoFs_for_pmf(PrYZ, PrYZ.variable.variableIDs)

        # Handle PrZ separately, to check if was instantiated with a single
        # variable (when a JointVariables object is expected).
        if PrZ is not None:
            ZvariableIDs = None
            try:
                ZvariableIDs = PrZ.variable.variableIDs
            except AttributeError:
                pass

            if ZvariableIDs is not None:
                self.cache_DoFs_for_pmf(PrZ, ZvariableIDs)


    def calculate_DoF(self, X, Y, Z):
        key = {X, Y}
        key.update(Z)
        key = frozenset(key)
        (variables, pairwise_dofs) = self.DoF_cache[key]

        ix = variables.index(X)
        iy = variables.index(Y)

        DoF = pairwise_dofs[(ix, iy)]
        return DoF


    def cache_DoFs_for_pmf(self, pmf, variables):
        if len(variables) <= 1:
            return

        key = frozenset(variables)

        if key not in self.DoF_cache:
            pairwise_dofs = self.calculate_pairwise_DoFs(pmf, len(variables))
            self.DoF_cache[key] = (variables, pairwise_dofs)


    def calculate_pairwise_DoFs(self, pmf, keysize):
        # Remove any 0 entry from the PMF, as it they will increase DoF
        # artificially.
        pmf.remove_zeros()

        pairwise_dofs = dict()

        for ix in range(keysize):
            for iy in range(ix + 1, keysize):

                x_values_per_z = collections.defaultdict(set)
                y_values_per_z = collections.defaultdict(set)
                zs = set()
                for key in pmf.keys():
                    x = key[ix]
                    y = key[iy]
                    z = tuple(key[iz] for iz in range(keysize) if iz != ix and iz != iy)
                    x_values_per_z[z].add(x)
                    y_values_per_z[z].add(y)
                    zs.add(z)

                DoF = 0
                for z in zs:
                    X_val = len(x_values_per_z[z])
                    Y_val = len(y_values_per_z[z])
                    DoF += (X_val - 1) * (Y_val - 1)

                if DoF == 0:
                    DoF = 1
                pairwise_dofs[(ix, iy)] = pairwise_dofs[(iy, ix)] = DoF

        return pairwise_dofs
