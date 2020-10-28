import math

class EarlyStopping(object):
    
    def __init__(self, patience=5, delta=1e-2, less_is_better=True):

        self.patience = patience
        self.delta = delta
        self.less_is_better = less_is_better
        self.best = math.inf if less_is_better else -math.inf
        self.counter = 0

    def update(self, loss):
        if self.best is None:
            self.best = loss
            return

        if self.less_is_better:
            # Best loss updated.
            if loss < self.best - self.delta:
                self.best = loss
                self.counter = 0
            else:
                self.counter += 1
        else:
            # Best loss updated.
            if loss > self.best + self.delta:
                self.best = loss
                self.counter = 0
            else:
                self.counter += 1

    def step(self, loss):
        self.update(loss)

        # Return True if the counter exceeded patience.
        return self.counter >= self.patience

    def get_best_score(self):
        return self.best
