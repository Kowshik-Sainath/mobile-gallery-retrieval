import torch

def check_tensor(name, t):
    if t is None:
        return
    if torch.isnan(t).any():
        print(f"!!! NAN detected in {name} !!!")
    if torch.isinf(t).any():
        print(f"!!! INF detected in {name} !!!")
