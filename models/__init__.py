
def calc_z_shapes(n_channel, input_size, n_blocks):
    z_shapes = []

    for _ in range(n_blocks - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes