from models.criterions.GRASP import GRASP


class EarlyGRASP(GRASP):
    """
    Original creation from our paper:  https://arxiv.org/abs/2006.00896
    Implements SNIP-it (before training)
    """

    def __init__(self, *args, limit=0.0, steps=100, **kwargs):
        self.limit = limit
        steps = 5
        super(EarlyGRASP, self).__init__(*args, **kwargs)
        ratio = 0.5
        # self.steps = []
        self.steps = [limit - (limit - ratio) * (ratio ** i) for i in range(steps + 1)] + [limit]

        # TODO: DELETE THIS
        self.steps = [limit]
        # n = 100
        # for k in range(1, 101):
        #     self.steps.append(limit*(k/n))

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage=0.0, *args, **kwargs):
        while len(self.steps) > 0:
            # determine k_i
            percentage = self.steps.pop(0)

            # prune
            super().prune(percentage=percentage, *args, **kwargs)
            if len(self.steps) != 0:
                while self.model.pruned_percentage > self.steps[0]:
                    self.steps.pop(0)
