"""
Helper utilities.
"""


def read_coco_labels(file_path):
    """
    Helper for loading coco_labels.txt
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        ret = {}
        for line in lines:
            pair = line.strip().split(maxsplit=1)
            ret[int(pair[0])] = pair[1].strip()
    return ret
