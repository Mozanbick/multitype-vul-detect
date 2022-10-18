from typing import Dict
from .properties import Properties
from .types import TYPES

node_labels = ["Block", "Call", "Comment", "ControlStructure", "File", "Identifier", "FieldIdentifier", "Literal",
               "Local", "Member", "MetaData", "Method", "MethodInst", "MethodParameterIn", "MethodParameterOut",
               "MethodReturn", "Namespace", "NamespaceBlock", "Return", "Type", "TypeDecl", "Unknown"]

operators = ['addition', 'addressOf', 'and', 'arithmeticShiftRight', 'assignment',
             'assignmentAnd', 'assignmentArithmeticShiftRight', 'assignmentDivision',
             'assignmentMinus', 'assignmentMultiplication', 'assignmentOr', 'assignmentPlus',
             'assignmentShiftLeft', 'assignmentXor', 'cast', 'conditionalExpression',
             'division', 'equals', 'fieldAccess', 'greaterEqualsThan', 'greaterThan',
             'indirectFieldAccess', 'indirectIndexAccess', 'indirection', 'lessEqualsThan',
             'lessThan', 'logicalAnd', 'logicalNot', 'logicalOr', 'minus', 'modulo', 'multiplication',
             'not', 'notEquals', 'or', 'postDecrement', 'plus', 'postIncrement', 'preDecrement',
             'preIncrement', 'shiftLeft', 'sizeOf', 'subtraction']

node_labels += operators


class Operators:

    unary_ops = [
        'addressOf', 'logicalNot', 'not', 'postDecrement', 'postIncrement', 'minus', 'plus',
        'sizeOf', 'preDecrement', 'preIncrement'
    ]

    binary_ops = [
        'addition', 'and', 'arithmeticShiftRight', 'assignment', 'assignmentAnd', 'assignmentArithmeticShiftRight',
        'assignmentDivision', 'assignmentMinus', 'assignmentMultiplication', 'assignmentOr', 'assignmentPlus',
        'assignmentShiftLeft', 'assignmentXor', 'division', 'equals', 'fieldAccess', 'greaterEqualsThan',
        'greaterThan', 'indirectFieldAccess', 'indirectIndexAccess', 'indirection', 'lessEqualsThan', 'lessThan',
        'logicalAnd', 'logicalOr', 'modulo', 'multiplication', 'notEquals', 'or', 'shiftLeft', 'subtraction'
    ]

    conditional_ops = ['conditional']


class Node:
    """
    A class for the node structure
    Contains: properties
    """

    def __init__(self, node_info: Dict):
        self.properties = Properties(node_info)
        # indicate if a node is AST node(1, 0001), CFG node(2, 0010), CDG node(4, 0100) or DDG node(8, 1000)
        self.type = TYPES.NONE
        self.attr = 'Unknown'

    @property
    def id(self):
        return self.properties.id

    @property
    def label(self) -> str:
        return self.properties.label

    @property
    def code(self) -> str:
        return self.properties.code

    @property
    def name(self) -> str:
        return self.properties.name

    @property
    def line_number(self):
        return self.properties.line_number

    @property
    def column_number(self):
        return self.properties.column_number

    @property
    def control_type(self):
        return self.get_property("controlStructureType")

    @property
    def node_type(self) -> int:
        return self.type

    def add_type_attr(self, node_type):
        if node_type == "AST":
            self.type = self.type | TYPES.AST
        elif node_type == "CFG":
            self.type = self.type | TYPES.CFG
        elif node_type == "CDG":
            self.type = self.type | TYPES.CDG
        elif node_type == "DDG":
            self.type = self.type | TYPES.DDG

    def get_property(self, name):
        return self.properties.get_property(name)

    @property
    def node_attr(self):
        return self.attr

    def set_attr(self, parent=None):
        """
        Set node ast attribute
        """
        if self.label == "METHOD":
            self.attr = "FunctionDefinition"
        elif self.label == "METHOD_PARAMETER_IN":
            self.attr = "ParameterDeclaration"
        elif self.label == "BLOCK":
            self.attr = "CompoundStatement"
        elif self.label == "METHOD_RETURN":
            self.attr = "NamedTypeSpecifier"
        elif self.label == "RETURN":
            self.attr = "ReturnStatement"
        elif self.label == "IDENTIFIER" or self.label == "FIELD_IDENTIFIER":
            self.attr = "IdExpression"
        elif self.label == "CALL":
            if not self.name.startswith("<operator>."):  # common function call
                self.attr = "FunctionCallExpression"
            else:  # operators
                name = self.name.removeprefix("<operator>.")
                # if an assignment is the suffix of a local
                # this assignment should be "EqualsInitializer"
                if name == "assignment" and parent and parent.label == "LOCAL" and self.line_number == parent.line_number:
                    self.attr = "EqualsInitializer"
                    return
                if name == "fieldAccess" or name == "indirectFieldAccess":
                    self.attr = "FieldReference"
                    return
                if name == "cast":
                    self.attr = "CastExpression"
                    return
                if name == "indexAccess" or name == "indirectIndexAccess":
                    self.attr = "ArraySubscriptExpression"
                    return
                # other condition
                if name in Operators.unary_ops:  # unary operators
                    self.attr = "UnaryExpression"
                elif name in Operators.binary_ops:  # binary operators
                    self.attr = "BinaryExpression"
                elif name in Operators.conditional_ops:  # conditional expression
                    self.attr = "ConditionalExpression"
                else:
                    self.attr = "Unknown"
        elif self.label == "LOCAL":
            self.attr = "DeclarationStatement"
        elif self.label == "LITERAL":
            self.attr = "LiteralExpression"
        elif self.label == "CONTROL_STRUCTURE":
            tp = self.get_property("controlStructureType")
            if tp == "IF":
                self.attr = "IfStatement"
            elif tp == "FOR":
                self.attr = "ForStatement"
            elif tp == "SWITCH":
                self.attr = "SwitchStatement"
            elif tp == "BREAK":
                self.attr = "BreakStatement"
            elif tp == "GOTO":
                self.attr = "GotoStatement"
            elif tp == "WHILE":
                self.attr = "WhileStatement"
            elif tp == "DO":
                self.attr = "DoStatement"
            elif tp == "CONTINUE":
                self.attr = "ContinueStatement"
            else:
                self.attr = "Unknown"
        elif self.label == "JUMP_TARGET":
            tp = self.get_property("parserTypeName")
            if tp == "CASTCaseStatement":
                self.attr = "CaseStatement"
            elif tp == "CASTDefaultStatement":
                self.attr = "DefaultStatement"
            elif tp == "CASTLabelStatement":
                self.attr = "LabelStatement"
            else:
                self.attr = "Unknown"
        elif self.label == "UNKNOWN":
            tp = self.get_property("parserTypeName")
            if tp == "CASTTypeId":
                self.attr = "TypeId"
            elif tp == "CASTProblemStatement":
                self.attr = "ProblemStatement"
            elif tp == "CASTProblemExpression":
                self.attr = "ProblemExpression"
            elif tp == "CASTProblemDeclaration":
                self.attr = "ProblemDeclaration"
            else:
                self.attr = "Unknown"
        else:
            self.attr = "Unknown"
