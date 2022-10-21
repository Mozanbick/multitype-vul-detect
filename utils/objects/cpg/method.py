
from .node import Node
from .edge import Edge
from .types import TYPES
from typing import List, Dict, Set


class Method:
    """
    A class for methods' structure
    """

    def __init__(self, entry: int, ret: int, testID: str, name: str, node_list: List[Node], edge_list: List[Edge]):
        self._entry = entry
        self._ret = ret
        self._name = name
        self._testID = testID
        self.nodes: Dict[int, Node] = {node.id: node for node in node_list}
        self.edges = edge_list
        self.ast_edges: Dict[int, Set] = {}
        self.cfg_edges: Dict[int, Set] = {}
        self.cdg_edges: Dict[int, Set] = {}
        self.ddg_edges: Dict[int, Set] = {}
        self._node_id_set = self.nodes.keys()
        self._local_param: Dict[str, List] = {}
        self._func_calls: Dict[str, List] = {}
        self.nodes_in_line: Dict[int, List] = {}
        self.use_line_to_def_line: Dict[str, Set] = {}
        self.init_graph()
        # modify v2
        # no need
        # self._replenish_def_use_relations_new()

    def parseConnections(self):
        """
        Parse edges<Edge> into connections<Map[int, Tuple(int, str)]>
        """
        if self.edges:
            cons = {}
            for edge in self.edges:
                cons[edge.node_in] = (edge.node_out, edge.type)
            return cons
        return None

    def parseGraph(self):
        """
        Parse edges<Edge> into sub-graph<Map[int, int]>(i.e. ast, cfg, cdg, ddg)
        """
        if self.edges:
            for item in TYPES.items():
                sub_graph = {}
                for edge in self.edges:
                    if edge.type == item and edge.node_in in self.nodes and edge.node_out in self.nodes:
                        # add node type attribute
                        self.nodes[edge.node_in].add_type_attr(item)
                        self.nodes[edge.node_out].add_type_attr(item)
                        # map edges
                        if edge.node_in not in sub_graph:
                            sub_graph[edge.node_in] = {edge.node_out}
                        else:
                            sub_graph[edge.node_in].add(edge.node_out)
                if item == "AST":
                    self.ast_edges = sub_graph
                elif item == "CFG":
                    self.cfg_edges = sub_graph
                elif item == "CDG":
                    self.cdg_edges = sub_graph
                elif item == "DDG":
                    self.ddg_edges = sub_graph
            del self.edges
        return None

    def _divide_nodes_in_line(self):
        """
        Sorts nodes in line number
        """
        line_node_dict = {}
        cur_line = 1
        cur_nodes = []
        for nid, node in self.nodes.items():
            if node.line_number and node.line_number > cur_line:
                line_node_dict[cur_line] = cur_nodes
                cur_line = node.line_number
                cur_nodes = []
            cur_nodes.append(nid)
        if cur_nodes:
            line_node_dict[cur_line] = cur_nodes
        return line_node_dict

    def init_graph(self):
        """
        Init a graph of this method
        # Do: divide nodes in line number;
              collect local and param information;
              collect function call information;
              replenish ast relationships due to BLOCK nodes;
        """
        line_node_dict = {}
        cur_line = 1
        cur_nodes = []
        # draw edges
        self.parseGraph()
        # draw nodes
        last_node = None
        variable_idx = 0
        function_idx = 0
        for nid, node in self.nodes.items():
            if not node:
                del self.nodes[nid]
                continue
            # collect local and param information
            # info represents for (node's id, node's line number, if a node is pointer type, node's alias)
            # only the condition of '*' and '['
            p_local = False
            if node.label == "METHOD_PARAMETER_IN":
                if "*" in node.code or "[" in node.code:
                    p_local = True
                self._local_param[node.name] = [node.id, node.line_number, p_local, f"var_{variable_idx}"]
                variable_idx += 1
            elif node.label == "LOCAL":
                if "*" in node.code or "[" in node.code:
                    p_local = True
                self._local_param[node.name] = [node.id, node.line_number, p_local, f"var_{variable_idx}"]
                variable_idx += 1
            # collect function call information
            # info represents for (node's id, node's alias)
            elif node.label == "CALL" and not node.name.startswith("<operator>."):
                if node.name not in self._func_calls:
                    self._func_calls[node.name] = [node.id, f"func_{function_idx}"]
                    function_idx += 1
            # replenish ast relationships
            elif node.label == "BLOCK":
                fid = self.get_father_node(nid)
                if fid in self.ast_edges and nid in self.ast_edges:
                    self.ast_edges[fid].update(self.ast_edges[nid])
            # initialize node attribution
            node.set_attr(last_node)
            last_node = node
            # divide nodes in line number
            if node.line_number and node.line_number > cur_line:
                line_node_dict[cur_line] = cur_nodes
                cur_line = node.line_number
                cur_nodes = []
            cur_nodes.append(nid)
        if cur_nodes:
            line_node_dict[cur_line] = cur_nodes
        self.nodes_in_line = line_node_dict

    def _get_ast_tree_nodes_no_call(self, root: int):
        """
        Get the id of all the nodes in ast subtree start from root node, but except the call type node and its children
        """
        children = []
        node_not_traverse = {root}
        while node_not_traverse:
            pid = node_not_traverse.pop()
            if self.nodes[pid].label == "CALL" and not self.nodes[pid].name.startswith("<operator>"):
                continue
            if pid in children:
                continue
            children.append(pid)
            node_not_traverse.update(self.ast_edges[pid] if pid in self.ast_edges else [])
        return children

    def _get_ast_tree_nodes_parent(self, root: int):
        """
        Get the id of all the CALL and CONTROL_STRUCTURE nodes in ast subtree start from root node
        """
        children = set()
        node_not_traverse = self.ast_edges[root] if root in self.ast_edges else {}
        while node_not_traverse:
            pid = node_not_traverse.pop()
            if self.nodes[pid].label != "CALL" and self.nodes[pid].label != "CONTROL_STRUCTURE":
                if self.nodes[pid].label == "BLOCK":
                    node_not_traverse.update(self.ast_edges[pid] if pid in self.ast_edges else [])
                continue
            children.add(pid)
        return children

    def get_ast_subtree_nodes(self, root: int):
        return self._get_ast_subtree_nodes(root)

    def _get_ast_subtree_nodes(self, root: int):
        """
        Get the id of all nodes in ast subtree start from the root node
        """
        children = set()
        node_not_traverse = {root}
        while node_not_traverse:
            pid = node_not_traverse.pop()
            children.add(pid)
            node_not_traverse.update(self.ast_edges[pid] if pid in self.ast_edges else [])
        return sorted(list(children))

    def _get_access_identifier(self, access):
        while self.nodes[access].label != "IDENTIFIER":
            access += 1
        return access

    def _add_use_data_dependency(self, def_dict: Dict[str, int], name: str, node_id: int):
        if name not in def_dict:
            return
        if def_dict[name] in self.ddg_edges:
            self.ddg_edges[def_dict[name]].add(node_id)
        else:
            self.ddg_edges[def_dict[name]] = {node_id}

    def _replenish_def_use_relations_new(self):
        """
        Replenish def-to-use relationships in a function, which are not included in cpg
            but are useful in slice generation
        ### def-to-use relationships refers to the edges connecting identifier definition node
            to identifier use node
        ### re-definition nodes are taken into consideration
        """
        for nid, node in self.nodes.items():
            try:
                # a node with label "IDENTIFIER" is an identifier use node
                if node.label == "IDENTIFIER":
                    if node.name in self._local_param and node.line_number:
                        # add ddg edge from definition node to use node
                        tid = self.get_parent_node_new(node.id)
                        # TODO: 加不加这条边在生成切片图的时候十分关键
                        if self._local_param[node.name][0] in self.ddg_edges:
                            self.ddg_edges[self._local_param[node.name][0]].add(tid)
                        else:
                            self.ddg_edges[self._local_param[node.name][0]] = {tid}
                        self._local_param[node.name][3] = True
                # re-definition situations to deal
                elif node.label == "CALL" and node.name == "<operator>.assignment":
                    # definition with initial value, pass
                    if self.nodes[node.id - 1].label == "LOCAL":
                        continue
                    # commonly '=' ast node has two children, the left and the right
                    if len(self.ast_edges[node.id]) != 2:
                        continue
                    left, right = self.ast_edges[node.id]
                    # first to add dependency from the definition to the right and from the right to the left
                    right_nodes = self._get_ast_tree_nodes_no_call(right)
                    for pid in right_nodes:
                        if self.nodes[pid].label == "IDENTIFIER":
                            if self.nodes[pid].name in self._local_param:
                                tid = self.get_parent_node_new(pid)
                                # self._add_use_data_dependency(def_dict, self.nodes[pid].name, tid)
                                if tid in self.ddg_edges:
                                    self.ddg_edges[tid].add(node.id)
                                else:
                                    self.ddg_edges[tid] = {node.id}
                    # then to deal the left, corresponding to the re-def situation
                    # add use dependency first
                    # common variable
                    left_nodes = self._get_ast_tree_nodes_no_call(left)
                    for pid in left_nodes:
                        if self.nodes[pid].label == "IDENTIFIER":
                            if self.nodes[pid].name in self._local_param:
                                tid = self.get_parent_node_new(pid)
                                if node.id in self.ddg_edges:
                                    self.ddg_edges[node.id].add(tid)
                                else:
                                    self.ddg_edges[node.id] = {tid}
                # Parameters of a function should be taken into consideration too
                # Commonly, a function's parameters won't change when the function returns, so there is no dependency
                # between parameters and other identifiers after the function call
                # But the reality is exactly the opposite for address type(i.e. array, pointer) variables in functions
                # such as scanf, fgets etc., in which the contents the address points to would be changed and triggers
                # a re-definition condition
                elif node.label == "CALL" and not node.name.startswith("<operator>"):
                    if node.id not in self.ast_edges:
                        continue
                    # traverse all its child nodes
                    child_not_traverse = self.ast_edges[node.id]
                    while child_not_traverse:
                        pid = child_not_traverse.pop()
                        if pid in self.nodes and self.nodes[pid].label == "IDENTIFIER":
                            if self.nodes[pid].name not in self._local_param:
                                continue
                            # deal use situation
                            tid = self.get_parent_node_new(pid)
                            # self._add_use_data_dependency(def_dict, self.nodes[pid].name, tid)
                            # add dependency from parameter_in to function call
                            if tid in self.ddg_edges:
                                self.ddg_edges[tid].add(node.id)
                            else:
                                self.ddg_edges[tid] = {node.id}
                            # deal re-def situation
                            # if a parameter is array or pointer type
                            if self.nodes[pid].name in self._local_param and self._local_param[self.nodes[pid].name][2]:
                                # re-definition
                                self._local_param[self.nodes[pid].name][0] = tid
                                # add dependency from function call to parameter_out
                                if node.id in self.ddg_edges:
                                    self.ddg_edges[node.id].add(tid)
                                else:
                                    self.ddg_edges[node.id] = {tid}
                            # if a parameter has the option of '&'
                            elif self.nodes[pid - 1].name == "<operator>.addressOf":
                                # re-definition
                                self._local_param[self.nodes[pid].name][0] = tid
                                # add dependency from function call to parameter_out
                                if node.id in self.ddg_edges:
                                    self.ddg_edges[node.id].add(tid)
                                else:
                                    self.ddg_edges[node.id] = {tid}
                        if pid in self.ast_edges:
                            child_not_traverse.update(self.ast_edges[pid])
                # add dependency from CONTROL statement(i.e. if and for) to its child statements
                elif node.label == "CONTROL_STRUCTURE":
                    # if structure with identifier condition
                    if node.control_type == "IF" and self.nodes[node.id + 1].label == "IDENTIFIER":
                        if self.nodes[node.id + 1].name in self._local_param:
                            if self._local_param[self.nodes[node.id + 1].name][0] in self.ddg_edges:
                                self.ddg_edges[self._local_param[self.nodes[node.id + 1].name][0]].add(node.id)
                            else:
                                self.ddg_edges[self._local_param[self.nodes[node.id + 1].name][0]] = {node.id}
                    child_nodes = self._get_ast_tree_nodes_parent(node.id)
                    if node.id in self.ddg_edges:
                        self.ddg_edges[node.id].update(child_nodes)
                    else:
                        self.ddg_edges[node.id] = child_nodes
            except (KeyError, IndexError, AttributeError) as e:
                # print(e)
                continue

    def _replenish_def_use_relations(self):
        """
        Replenish def-to-use relationships in a function, which are not included in cpg
            but are useful in slice generation
        ### def-to-use relationships refers to the edges connecting identifier definition node
            to identifier use node
        ### re-definition nodes are taken into consideration
        """
        c_basic_type = ["int", "char", "short", "long", "float", "double", "unsigned int", "size_t",
                        "unsigned char", "unsigned short", "unsigned long", "long long", "unsigned long long"]
        def_dict = {}
        local_pos = {}
        pointer_local = set()
        for nid, node in self.nodes.items():
            # a node with label "LOCAL" is an identifier definition node
            if node.label == "LOCAL":
                def_dict[node.name] = nid
                if node.line_number:
                    local_pos[node.name] = node.line_number
                if "*" in node.code or "[" in node.code:
                    pointer_local.add(node.name)
                continue
            # a node with label "IDENTIFIER" is an identifier use node
            elif node.label == "IDENTIFIER":
                if node.name in local_pos and node.line_number:
                    if node.line_number in self.use_line_to_def_line:
                        self.use_line_to_def_line[node.line_number].add(local_pos[node.name])
                    else:
                        self.use_line_to_def_line[node.line_number] = {local_pos[node.name]}
            # re-definition situations to deal
            elif node.label == "CALL" and node.name == "<operator>.assignment":
                # definition with initial value, no action
                if self.nodes[node.id-1].label == "LOCAL":
                    continue
                # commonly '=' ast node has two children, the left and the right
                if len(self.ast_edges[node.id]) != 2:
                    continue
                left, right = self.ast_edges[node.id]
                # first to add dependency from the definition to the right and from the right to the left
                right_nodes = self._get_ast_tree_nodes_no_call(right)
                for pid in right_nodes:
                    if self.nodes[pid].label == "IDENTIFIER":
                        if self.nodes[pid].name in def_dict:
                            tid = self.get_parent_node(pid)
                            self._add_use_data_dependency(def_dict, self.nodes[pid].name, tid)
                            if tid in self.ddg_edges:
                                self.ddg_edges[tid].add(node.id)
                            else:
                                self.ddg_edges[tid] = {node.id}
                # then to deal the left, corresponding to the re-def situation
                # add use dependency first
                # common variable
                if self.nodes[left].label == "IDENTIFIER":
                    if self.nodes[left].name in def_dict:
                        self._add_use_data_dependency(def_dict, self.nodes[left].name, node.id)
                        def_dict[self.nodes[left].name] = node.id
                # array access condition
                elif self.nodes[left].name and "IndexAccess" in self.nodes[left].name:
                    pid = self._get_access_identifier(left+1)
                    if self.nodes[pid].name in def_dict:
                        self._add_use_data_dependency(def_dict, self.nodes[pid].name, node.id)
                        def_dict[self.nodes[pid].name] = node.id
                # pointer access condition
                elif self.nodes[left].name and "indirection" in self.nodes[left].name:
                    pid = self._get_access_identifier(left+1)
                    if self.nodes[pid].name in def_dict:
                        self._add_use_data_dependency(def_dict, self.nodes[pid].name, node.id)
                        def_dict[self.nodes[pid].name] = node.id
                # field access condition
                elif self.nodes[left].name and "fieldAccess" in self.nodes[left].name:
                    pid = self._get_access_identifier(left + 1)
                    if self.nodes[pid].name in def_dict:
                        self._add_use_data_dependency(def_dict, self.nodes[pid].name, node.id)
                        def_dict[self.nodes[pid].name] = node.id
            # Parameters of a function should be taken into consideration too
            # Commonly, a function's parameters won't change when the function returns, so there is no dependency
            # between parameters and other identifiers after the function call
            # But the reality is exactly the opposite for address type(i.e. array, pointer) variables in functions
            # such as scanf, fgets etc., in which the contents the address points to would be changed and triggers
            # a re-definition condition
            elif node.label == "CALL" and not node.name.startswith("<operator>"):
                if node.id not in self.ast_edges:
                    continue
                # traverse all its child nodes
                child_not_traverse = self.ast_edges[node.id]
                while child_not_traverse:
                    pid = child_not_traverse.pop()
                    if pid in self.nodes and self.nodes[pid].label == "IDENTIFIER":
                        if self.nodes[pid].name not in def_dict:
                            continue
                        # deal use situation
                        tid = self.get_parent_node(pid)
                        self._add_use_data_dependency(def_dict, self.nodes[pid].name, tid)
                        # add dependency from parameter_in to function call
                        if tid in self.ddg_edges:
                            self.ddg_edges[tid].add(node.id)
                        else:
                            self.ddg_edges[tid] = {node.id}
                        # deal re-def situation
                        # if a parameter is array or pointer type
                        if self.nodes[pid].name in pointer_local:
                            # re-definition
                            def_dict[self.nodes[pid].name] = tid
                            # add dependency from function call to parameter_out
                            if node.id in self.ddg_edges:
                                self.ddg_edges[node.id].add(tid)
                            else:
                                self.ddg_edges[node.id] = {tid}
                        # if a parameter has the option of '&'
                        elif self.nodes[pid-1].name == "<operator>.addressOf":
                            # re-definition
                            def_dict[self.nodes[pid].name] = tid
                            # add dependency from function call to parameter_out
                            if node.id in self.ddg_edges:
                                self.ddg_edges[node.id].add(tid)
                            else:
                                self.ddg_edges[node.id] = {tid}
                    if pid in self.ast_edges:
                        child_not_traverse.update(self.ast_edges[pid])
            # add dependency from CONTROL statement(i.e. if and for) to its child statements
            elif node.label == "CONTROL_STRUCTURE":
                # if structure with identifier condition
                if node.control_type == "IF" and self.nodes[node.id+1].label == "IDENTIFIER":
                    if self.nodes[node.id+1].name in def_dict:
                        self._add_use_data_dependency(def_dict, self.nodes[node.id+1].name, node.id)
                child_nodes = self._get_ast_tree_nodes_parent(node.id)
                if node.id in self.ddg_edges:
                    self.ddg_edges[node.id].update(child_nodes)
                else:
                    self.ddg_edges[node.id] = child_nodes

    @property
    def name(self):
        """
        Returns the filename of this method's source file
        """
        return self._name

    @property
    def node_id_set(self):
        """
        Returns the set of nodes' ids which are involved in this method
        """
        return self._node_id_set

    @property
    def entry(self):
        """
        Returns the entry node id of this method
        """
        return self._entry

    @property
    def ret(self):
        return self._ret

    @property
    def testID(self):
        """
        Returns the testID this method belongs to
        """
        return self._testID

    @property
    def filename(self):
        """
        Returns the filename of this method
        """
        return self.nodes[self._entry].get_property("filename")

    @property
    def local_param(self):
        return self._local_param

    @property
    def func_calls(self):
        return self._func_calls

    def get_params(self):
        """
        Returns the id set of the parameter(s) of this method
        """
        params = []
        for node in self.ast_edges[self._entry]:
            if self.nodes[node].label == "METHOD_PARAMETER_IN":
                params.append(node)
        return params

    def get_returns(self):
        """
        Returns the id set of the return statements of this method
        """
        rets = []
        for node in self.ddg_edges:
            if self._ret in self.ddg_edges[node] and self.nodes[node].label == "RETURN":
                rets.append(node)
        return rets

    def get_local_def_lines(self) -> List[int]:
        """
        Return the local or param definition lines
        """
        lp_list = []
        for name, info in self._local_param.items():
            if info[3]:
                lp_list.append(info[1])
        return lp_list

    def get_father_node(self, nid):
        """
        Returns purely the ast parent node
        """
        pid = nid - 1
        while pid > self._entry:
            if pid in self.ast_edges and nid in self.ast_edges[pid]:
                break
            pid -= 1
        if pid <= self._entry:
            pid = self._entry
        return pid

    def get_parent_node(self, nid):
        """
        Returns the parent node id of this node, specially the parent node is "CALL" type (decided by joern tool)
        ### specially, the parent type would not be of the types:
            <operator>.indirectIndexAccess, <operator>.addressOf, <operator>.indirection, <operator>.fieldAccess
        """
        not_include = {"<operator>.indirectIndexAccess", "<operator>.addressOf", "<operator>.indirection", "<operator>.fieldAccess"}
        if self.nodes[nid].label == "IDENTIFIER":
            while self.nodes[nid].label != "CALL" and self.nodes[nid].label != "CONTROL_STRUCTURE" or self.nodes[nid].name in not_include:
                pid = nid - 1
                while pid > self._entry:
                    if pid in self.ast_edges and nid in self.ast_edges[pid]:
                        break
                    pid -= 1
                nid = pid
                if nid == self._entry:
                    break
        return nid

    def get_parent_node_new(self, nid) -> int:
        """
        Updated
        We have marked every node with property "type", a node with "CDG" or "DDG" type could be a parent node
        """
        while self.nodes[nid].type & TYPES.DDG == 0 and self.nodes[nid].type & TYPES.CDG == 0:
            pid = nid - 1
            while pid > self._entry:
                if pid in self.ast_edges and nid in self.ast_edges[pid]:
                    break
                pid -= 1
            nid = pid
            if nid == self._entry:
                break
        return nid

    def get_ast_entrance(self, nid: int):
        """
        Actually a code statement usually corresponds to an ast subtree, and "CALL", "PARAMETER_IN" and "RETURN" nodes
        in subtree may carry control or data dependencies. Thus, for a focus point node in the subtree,
        we should extract it's "CALL" parent nodes as the slice generation entrance.
        """
        # entrance = set()
        # node_not_traverse = {nid}
        # # child nodes
        # while node_not_traverse:
        #     pid = node_not_traverse.pop()
        #     if pid not in entrance and self.nodes[pid].label == "CALL":
        #         entrance.add(pid)
        #     node_not_traverse.update(self.ast_edges[pid] if pid in self.ast_edges else [])
        # # parent nodes
        # entrance.add(self.get_parent_node(nid))
        # return entrance

        return [self.get_parent_node_new(nid)]
