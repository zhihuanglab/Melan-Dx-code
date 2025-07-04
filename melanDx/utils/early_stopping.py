import numpy as np


class EarlyStopping:

    def __init__(
        self,
        patience: int = 10,
        mode: str = 'max',
        min_delta: float = 0.0,
        verbose: bool = True
    ):

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if mode not in ['min', 'max']:
            raise ValueError(f"mode {mode} is unknown, only 'min' or 'max' are supported")
        

        self.monitor_op = np.greater if self.mode == 'max' else np.less
        

        self.best_score = float('-inf') if self.mode == 'max' else float('inf')
    
    def __call__(self, epoch: int, current_score: float) -> bool:

        score = current_score

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self.monitor_op(score - self.min_delta, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                print(f'Best score: {self.best_score:.6f} at epoch {self.best_epoch}')
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'\nEarly stopping triggered after epoch {epoch}')
                    print(f'Best score was {self.best_score:.6f} at epoch {self.best_epoch}')
                return True
        
        return False
    
    def reset(self):

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0