from typing import Any, Dict, List


def update_dict_of_lists(d1: Dict[Any, List[Any]],
                         d2: Dict[Any, Any]):
    for key, value in d2.items():
        values = d1.get(key)
        if values is None:
            d1[key] = [value]
        else:
            values.append(value)
