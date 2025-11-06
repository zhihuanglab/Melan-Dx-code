import random
import numpy as np
import torch
import os
import logging

def set_seed(seed: int = 42) -> None:

    logger = logging.getLogger(__name__)
    
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Python hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed set to {seed}") 