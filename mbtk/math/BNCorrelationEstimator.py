from typing import Union

from mbtk.math.HeuristicResult import HeuristicResult


class BNCorrelationEstimator:

    def __init__(self, _, parameters) -> None:
        self.parameters = parameters
        self.source_bn = self.parameters.get('source_bayesian_network', None)
        if self.source_bn is None:
            raise NotImplementedError

        self.heuristic_results: list[HeuristicResult] = []


    def compute(self, X: int, Y: int, Z: Union[set[int], list[int]]) -> float:
        Zs = list(Z)
        assert isinstance(Zs, list)
        bn = self.source_bn

        result = HeuristicResult()
        result.start_timing()
        if len(Z) > 0 and bn.conditionally_independent(X, Y, Z):
            return 0

        paths = bn.find_all_undirected_paths(X, Y)
        shortest_path = min(paths, key=len)
        value = 1.0 / len(shortest_path)
        result.end_timing()

        result.index = len(self.heuristic_results)
        result.set_variables(X, Y, Z)
        result.set_heuristic('BNCorrelationEstimator', value)
        self.heuristic_results.append(result)
        return value
