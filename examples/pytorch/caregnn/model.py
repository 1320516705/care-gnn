import dgl.function as fn
import numpy as np
import torch as th
import torch.nn as nn


class CAREConv(nn.Module):
    """One layer of CARE-GNN."""

    def __init__(
        self,
        in_dim,
        out_dim,
        num_classes,
        edges,
        activation=None,
        step_size=0.02,
    ):
        super(CAREConv, self).__init__()

        self.activation = activation
        self.step_size = step_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.edges = edges
        self.dist = {}

        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.MLP = nn.Linear(self.in_dim, self.num_classes)

        self.p = {}
        self.last_avg_dist = {}
        self.f = {}
        self.cvg = {}
        for etype in edges:
            self.p[etype] = 0.5 # 之后在强化学习时，改变的就是这个概率p值
            self.last_avg_dist[etype] = 0
            self.f[etype] = []
            self.cvg[etype] = False

    def _calc_distance(self, edges):
        # formula 2
        d = th.norm( # d={Tensor:(e_N,)}
            th.tanh(self.MLP(edges.src["h"]))# 整个式子的shape={Tensor:(N,2)}，其中N为节点数
            - th.tanh(self.MLP(edges.dst["h"])),
            1, # 表示使用 L1 范数进行计算
            1, # 表示对第一个维度求和
        )
        return {"d": d}

    def _top_p_sampling(self, g, p):
        # this implementation is low efficient
        # optimization requires dgl.sampling.select_top_p requested in issue #3100
        dist = g.edata["d"] # dist={Tensor:(e_N,)}
        neigh_list = []
        for node in g.nodes():
            edges = g.in_edges(node, form="eid") # 获得节点 node 的入边列表，指定返回边 ID，而不是边的其他属性。
            num_neigh = th.ceil(g.in_degrees(node) * p).int().item() #  计算节点 node 的入度（in_degrees），与参数 p 相乘并向上取整，然后转换为整数类型，并将结果赋值给变量 num_neigh。
            neigh_dist = dist[edges]#dist[edges] 的作用是从距离列表 dist 中选择与 入边edges（是索引） 对应的距离特征
            if neigh_dist.shape[0] > num_neigh:
                neigh_index = np.argpartition( # 找到数组中最小的num_neigh个元素的索引
                    neigh_dist.cpu().detach(), num_neigh
                )[:num_neigh]
            else:
                neigh_index = np.arange(num_neigh)
            neigh_list.append(edges[neigh_index])#edges[neigh_index] 的作用是从入边列表 edges 中选择与 neigh_index 对应的边标识
        return th.cat(neigh_list)# 将所有选中的边标识拼接成一个张量（），并返回

    def forward(self, g, feat):
        with g.local_scope():
            g.ndata["h"] = feat # feat={Tensor:(N,25)}，即g.ndata["h"]={Tensor:(N,25)}
# 分关系,（g.etypes）遍历所有关系，对每个关系，计算节点特征之间的距离，并使用top-p采样选择一部分邻居节点。
            hr = {}
            for i, etype in enumerate(g.canonical_etypes): # num_edges={('user', 'net_upu', 'user'): 351216, ('user', 'net_usu', 'user'): 7132958, ('user', 'net_uvu', 'user'): 2073474}
                g.apply_edges(self._calc_distance, etype=etype)# apply_edges用提供的函数更新指定边的特征
                self.dist[etype] = g.edges[etype].data["d"] #g.edges是用于获取图中的边索引（即图的结构），而g.edata用于存储图中边的特征数据。g.edges[etype] 是获取图 g 中特定类型（etype）的边对象，可以访问这些边的特征（属性）数据，保存在 g.edges[etype].data 中
                sampled_edges = self._top_p_sampling(g[etype], self.p[etype])

                # formula 8
                g.send_and_recv(
                    sampled_edges,
                    fn.copy_u("h", "m"),
                    fn.mean("m", "h_%s" % etype[1]),#这是接收端节点的特征存储位置的键名；其中 %s 是一个占位符，表示一个字符串要插入的位置；etype[1]是一个关系类型的中间部分，例如 ('user', 'net_upu', 'user') 中的 'upu'
                    etype=etype,#指定在什么类型的关系上执行这个消息传递和接收操作
                )
                hr[etype] = g.ndata["h_%s" % etype[1]]
                if self.activation is not None:
                    hr[etype] = self.activation(hr[etype])

            # formula 9 using mean as inter-relation aggregator
            p_tensor = (#将字典 self.p 中的值转换为张量形式
                th.Tensor(list(self.p.values())).view(-1, 1, 1).to(g.device)
            )
            h_homo = th.sum(th.stack(list(hr.values())) * p_tensor, dim=0) #: 使用 th.sum 函数计算跨不同关系类型的节点特征加权平均，并加上输入特征 feat
            h_homo += feat
            if self.activation is not None:
                h_homo = self.activation(h_homo)

            return self.linear(h_homo)


class CAREGNN(nn.Module):
    def __init__(
        self,
        in_dim,
        num_classes,
        hid_dim=64,
        edges=None,
        num_layers=2,
        activation=None,
        step_size=0.02,
    ):
        super(CAREGNN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.edges = edges
        self.activation = activation
        self.step_size = step_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        if self.num_layers == 1:
            # Single layer
            self.layers.append(
                CAREConv(
                    self.in_dim,
                    self.num_classes,
                    self.num_classes,
                    self.edges,
                    activation=self.activation,
                    step_size=self.step_size,
                )
            )

        else:
            # Input layer
            self.layers.append(
                CAREConv(
                    self.in_dim,
                    self.hid_dim,
                    self.num_classes,
                    self.edges,
                    activation=self.activation,
                    step_size=self.step_size,
                )
            )

            # Hidden layers with n - 2 layers
            for i in range(self.num_layers - 2):
                self.layers.append(
                    CAREConv(
                        self.hid_dim,
                        self.hid_dim,
                        self.num_classes,
                        self.edges,
                        activation=self.activation,
                        step_size=self.step_size,
                    )
                )

            # Output layer
            self.layers.append(
                CAREConv(
                    self.hid_dim,
                    self.num_classes,
                    self.num_classes,
                    self.edges,
                    activation=self.activation,
                    step_size=self.step_size,
                )
            )

    def forward(self, graph, feat):
        # For full graph training, directly use the graph
        # formula 4
        sim = th.tanh(self.layers[0].MLP(feat)) # sim={Tensor:(N,2)}

        # Forward of n layers of CARE-GNN
        for layer in self.layers:
            feat = layer(graph, feat) # 进入了CAREConv的forward方法

        return feat, sim

    def RLModule(self, graph, epoch, idx):
        for layer in self.layers:
            for etype in self.edges:
                if not layer.cvg[etype]:#自定义的，表示当前层中特定边类型是否已经收敛
                    # formula 5
                    eid = graph.in_edges(idx, form="eid", etype=etype)
                    avg_dist = th.mean(layer.dist[etype][eid])

                    # formula 6
                    if layer.last_avg_dist[etype] < avg_dist:
                        if layer.p[etype] - self.step_size > 0:
                            layer.p[etype] -= self.step_size
                        layer.f[etype].append(-1)# 累加奖励，达到条件（加和<=2并且epoch>=10），也就是RL模块终止后，过滤阈值固定为最优阈值
                    else:
                        if layer.p[etype] + self.step_size <= 1:
                            layer.p[etype] += self.step_size
                        layer.f[etype].append(+1)
                    layer.last_avg_dist[etype] = avg_dist

                    # formula 7
                    if epoch >= 9 and abs(sum(layer.f[etype][-10:])) <= 2:
                        layer.cvg[etype] = True
