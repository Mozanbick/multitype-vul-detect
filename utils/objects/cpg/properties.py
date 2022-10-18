from typing import Dict


class Properties:
    """
    A class for nodes' properties
    """

    def __init__(self, props: Dict):
        self.size = len(props)
        self.properties: Dict = props

    def get(self):
        """
        Returns the properties dict
        """
        return self.properties

    @property
    def id(self):
        """
        Returns the 'id' property of the node
        """
        return self.properties["id"]

    @property
    def label(self):
        """
        Returns the label(node type) of the node;
        Corresponding to the value of '_label' in properties dict
        """
        return self.properties["_label"]

    @property
    def code(self):
        """
        Returns the code property of the node;
        Actually every node has 'code' property, while some of them are '<global>'
            and '<empty>', which are meaningless.
        """
        if "code" in self.properties:
            if self.properties["code"] != "<global>" and self.properties["code"] != "<empty>":
                return self.properties["code"]
        return None

    @property
    def name(self):
        """
        Returns the 'name' property of the node;
        Most nodes have this property.
        """
        if "name" in self.properties:
            if self.properties["name"] != "<global>" and self.properties["name"] != "<empty>":
                return self.properties["name"]
        return None

    @property
    def line_number(self):
        """
        Returns the 'lineNumber' property of the node
        """
        return self.properties["lineNumber"] if "lineNumber" in self.properties else None

    @property
    def column_number(self):
        """
        Returns the 'columnNumber' property of the node
        """
        return self.properties["columnNumber"] if "columnNumber" in self.properties else None

    def get_property(self, name: str):
        """
        Returns one of the properties of this node
        """
        return self.properties[name] if name in self.properties else None
