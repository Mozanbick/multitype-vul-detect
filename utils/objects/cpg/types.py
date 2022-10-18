class TYPES:

    NONE = 0b0000
    AST = 0b0001
    CFG = 0b0010
    CDG = 0b0100
    DDG = 0b1000

    @classmethod
    def items(cls):
        return {"AST", "CFG", "CDG", "DDG"}
