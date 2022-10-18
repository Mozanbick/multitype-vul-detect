from typing import Dict


class DefDict:
    """
    A class for def-dict structure
    ### be used in Method while adding def--use relations
    ### the structure is like
        {
            1: {
                    "data": 21,
                    "data_buffer": 25,
                },
            2: {
                    "tmp_variable": 37,
                }
        }
    """

    def __init__(self):
        self._def_dict: Dict[int, Dict[str, int]] = {}
        self._depth = 1
