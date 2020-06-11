"""
Helper utilities.
"""


def read_labels(file_path):
    """
    Helper for loading labels.txt
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        ret = {}
        for line in lines:
            pair = line.strip().split(maxsplit=1)
            ret[int(pair[0])] = pair[1].strip()
    return ret
