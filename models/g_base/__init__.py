from .model import Glow

def get_model(args: list):
    n_blocks, n_flows = map(int, args) if args else (4, 32)

    model = Glow(3, n_flows, n_blocks)

    return model