

def shorten_param_name(name):
    """Heuristic sort of parsing based on conventions in the networks.
    (maybe incomplete)
    """
    short = ""
    if "conv" in name:
        short += "Conv"
    elif "output" in name:
        short += "Output"
    else:
        short += "FC"  # (assumes only conv or FC layers)
    if "Val" in name:
        short += "Val"
    if "Adv" in name:
        short += "Adv"
    if "output" not in name:
        for num in [str(i) for i in range(100, -1, -1)]:
            if num in name:
                short += num
                break
    for alpha in ["W", "b"]:
        if alpha in name:
            short += alpha
    return short
