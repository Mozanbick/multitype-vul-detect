import xml.sax
from typing import Dict, List


class LabelHandler(xml.sax.handler.ContentHandler):
    """
    Handler to process labels in the SARD dataset
    """

    def __init__(self, label_dict: Dict[str, Dict[str, List]]):
        super().__init__()
        self.cur_testID = None
        self.cur_filename = None
        self._label_dict: Dict[str, Dict[str, List]] = label_dict

    def startElement(self, name, attrs):
        if name == "file":
            path: str = attrs["path"]
            if not path.startswith("000/") or not path.endswith(".c"):
                return
            items = path.split("/")
            testID = items[1] + "-" + items[2]
            filename = items[-1]
            self.cur_testID = testID
            self.cur_filename = filename
            if testID in self._label_dict:
                if filename not in self._label_dict[testID]:
                    self._label_dict[testID][filename] = []
            else:
                self._label_dict[testID] = {filename: []}
        elif name == "flaw" or name == "mixed":
            line = attrs["line"]
            self._label_dict[self.cur_testID][self.cur_filename].append(int(line))
