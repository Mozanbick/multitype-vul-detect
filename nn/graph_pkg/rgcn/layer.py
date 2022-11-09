import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import TypedLinear


class RGCNLayer(nn.Module):
    """
    Relational Graph Conv Network Layer
    ++  Computing outgoing message using node representation and weight matrix
        associated with edge type (message function)
    ++  Aggregate incoming messages and generate new node representations (reduce
        and reply function)
    """

    def __init__(self,
                 in_feature, out_feature, num_relations, num_bases=-1,
                 bias=None, activation=None, is_input_layer=None,
                 regularizer=None, self_loop=True, dropout=0.0, layer_norm=False):
        """

        :param num_bases: In order to reduce model parameter size and prevent over-fitting,
            the R-GCN proposes to use basis decomposition. The weight W_r^{(l)} in weight matrix
            is a linear combination of basis transformation V_b^{(l)} with coefficients
            a_{rb}^{(l)} ---- reference to the equation (3) in original paper.
            The number of bases B is much smaller than the number of relations in the knowledge base.
        """

        super(RGCNLayer, self).__init__()

        self.linear_r = TypedLinear(in_feature, out_feature, num_relations, regularizer, num_bases)
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        self.in_feature = in_feature  # input feature dimension
        self.out_feature = out_feature  # output feature dimension
        self.num_relations = num_relations  # number of edge relations
        self.num_bases = num_bases  # number of bases

        self.is_input_layer = is_input_layer  # if is the input layer (the first layer)

        # sanity check
        # the number of bases should in [1, num_relations]
        if self.num_bases <= 0 or self.num_bases > self.num_relations:
            self.num_bases = self.num_relations

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feature,
                                                self.out_feature))  # trainable Parameter
        if self.num_bases < self.num_relations:
            # linear combination coefficient in equation (3)
            # 对应于公式3中的a_{rb},
            # 每个edge relation对应一个W_r, 每个W_r的shape是(num_bases)
            # 所以a_{rb}的shape就是(num_relations, num_bases)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_relations, self.num_bases))

        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feature))

        # init trainable parameters
        # 这里用的是xavier_uniform, 此外还有使用glorot_uniform的实例
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_relations:
            nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_relations:
            # generate all weights from bases (equation 3)
            # |E| = num_relations, 代表关系个数, 或者说边的种类
            # 首先weight的shape调整为(in, B, out), a_{rb}的shape是(|E|, B)
            # 那么a_{rb}和weight的matmul得到的shape就是(in, |E|, out)
            # 最后将weight的shape调整为(|E|, in, out)
            weight = self.weight.view(self.in_feature, self.num_bases, self.out_feature)
            weight = torch.matmul(self.w_comp, weight).view(self.num_relations, self.in_feature, self.out_feature)
        else:
            # shape (|E|, in, out), |E| == B
            weight = self.weight

        if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                # rel_type是边类型的编码, 从0开始依次递增
                # 首先把weight的shape从(|E|, in, out)展开为(|E|*in, out)
                embed = weight.view(-1, self.out_feature)
                # 所以rel_type*in+src['id']这个索引对应的就是编号为'id'的node对应的shape为(out)的张量
                index = edges.data['rel_type'] * self.in_feature + edges.src['id']
                # 信息的汇聚就变成了矩阵相乘
                # 这时候embed[index]表示的就是edges对应的weight
                msg = embed[index] * edges.data['norm']
                return {'msg': msg}
        else:
            def message_func(edges):
                # 首先提取出rel_type边类型对应的weight矩阵
                # 有趣的是这里的shape是(rel_type, in, out)
                w = weight[edges.data['rel_type'].long()]
                # src['h']是edges的源结点的隐层特征, shape是(rel_type, in)
                # edges.src['h'].unsqueeze(1): (rel_type, in) -> (rel_type, 1, in)
                # bmm矩阵乘法, (rel_type, 1, in) * (rel_type, in, out) -> (rel_type, 1, out)
                # squeeze()后shape变为(rel_type, out)
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}

        def apply_func(nodes):
            # 提取出需要的hidden units特征以及node feature
            # h 的shape是(in, out)
            h = nodes.data['h']
            feat = nodes.data['feat']
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)
