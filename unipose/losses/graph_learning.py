from math import log10 as log
from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F


def SparsePlus(x, epsilon=1e-8):
    return (x + torch.sqrt(x * x + epsilon)) / 2.0


class GLES(nn.Module):
    r"""
    ====== Graph LEarning by Smoothness (GLES) =======
    Pairwise Graph Smoothness:
        $ tr(X^TLX) + \lambda \|W\|_F^2 $
        i.e.
        $ \sum_{i,j} W_{ij}\|(X_i-X_j)\|^2 + \lambda \sum_{i,j} W_{ij}^2 $

        When $\lambda$ is larger, the model shows a greater tolerance to the pairwise distance.

    Settings of the layer:
    >   num_nodes: number of nodes $N$ of the graph. Defaultly set to 'None'.
                   if set to 'None', GLES Layer can be used for any number of nodes.
    >   expect_edges: a function of the relationsip between expected number of edges in the graph 
                            and the number of nodes. Defaultly set to 'None'.
                      if set to 'None', automatically set to $f$,
                            where $f(N) = 2N/(\lfloor log(N)\rfloor+1)$.
    >   lambd: normalization factor $\lambda$.
    >   max_iter: the largest number of iterations in Newton's Method. Defaultly set to 10.
    >   mode: whether the learned graph should be directed or not.
              must be 'undirected' or 'directed'. Defaultly set to 'undirected'.
    """

    def __init__(self, num_nodes=None, expect_edges=None, lmbda=1, max_iter=5, mode="undirected"):
        super(GLES, self).__init__()
        self.mode = mode
        self.nodes = num_nodes
        self.lmbda = lmbda
        self.edges = expect_edges
        self.max_iter = max_iter
        
        if self.edges is None:
            self.edges = lambda x: x * (int(log(x)) + 1)

        if self.mode != "undirected" and self.mode != "directed":
            raise ValueError("GLES Error: mode must be 'undirected' or 'directed'.")
        if self.nodes is not None and (self.edges(self.nodes) > self.nodes * (self.nodes - 1)):
            raise ValueError(f"GLES Error: #(edges)>#(all possible edges when there are {self.nodes} nodes).")

    def forward(self, x):
        # input: (num_nodes, num_features)
        # output: (num_nodes, num_nodes)
        nodes = x.shape[0]
        if (self.nodes is not None) and (self.nodes != nodes):
            raise ValueError("GLES Error: #(nodes) must be consistent with the layer's setting.")
        edges = self.edges(nodes)
        if (edges>nodes*(nodes-1)):
            raise ValueError(f"GLES Error: #(edges)>#(all possible edges when there are {nodes} nodes).")
        
        theta = torch.norm(x[:, None, :] - x[None, :, :], dim=-1)
        gamma = torch.min(theta).requires_grad_()
        graph = torch.ones(nodes, nodes).requires_grad_(False)
        for _ in range(self.max_iter):
            g = theta + gamma
            g = g - torch.diag_embed(torch.diag(g))
            equ = torch.sum(SparsePlus(-g / (2.0 * self.lmbda))) - edges
            grad = torch.autograd.grad(equ.sum(), gamma, create_graph=True)[0]
            gamma = gamma - (equ / grad)
            graph = F.relu(-g / (2.0 * self.lmbda)).detach().requires_grad_(False)
            graph = graph - torch.diag_embed(torch.diag(graph))

        if self.mode == "undirected":
            graph = (graph + graph.T) / 2.0

        return graph


class GLES_fc(nn.Module):
    def __init__(self, num_features=32*32, num_nodes=None, expect_edges=None, lmbda=1, max_iter=5, mode="undirected"):
        super(GLES_fc, self).__init__()
        self.mode = mode
        self.nodes = num_nodes
        self.lmbda = lmbda
        self.edges = expect_edges
        self.max_iter = max_iter
        
        self.fc = nn.Linear(num_features, int(sqrt(num_features)))
        self.gles = GLES(num_nodes, expect_edges, lmbda, max_iter, mode)
        
        nn.init.xavier_normal_(self.fc.weight)
        
    def forward(self, x):
        x = self.fc(x)
        return self.gles(x)


if __name__ == "__main__":
    print("A Test of Correctness on Graph_Learning Layer.")
    lmbda = 2.0
    # x = [
    #         [1.0, 1.0, 1.0],  # feature of node 0
    #         [1.0, 1.1, 1.0],
    #         [10.0, 10.0, 10.0],
    #         [1.0, 1.0, 0.7],
    #         [11.0, 12.0, 9.0],
    #         [8.0, 10.0, 10.0],
    #         [-7.0, -7.0, -7.0],
    #         [-6.9, -7.1, -7.3],
    #     ]
    x = torch.randn((13, 32 * 32))
    # x = torch.tensor(x)
    print(x)

    import time

    start = time.perf_counter()
    model = GLES(lmbda=lmbda)
    graph = model(x)
    print(graph)
    print("Time: ", time.perf_counter() - start)
