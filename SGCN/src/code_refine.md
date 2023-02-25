
(1) param_parser
input, log, feature, weight
layers
reduction-dimensions

(2) utils.py
bug:

    p_edges = positive_edges + [[edge[1], edge[0]] for edge in positive_edges]
    n_edges = negative_edges + [[edge[1], edge[0]] for edge in negative_edges]
    train_edges = p_edges + n_edges
    index_1 = [edge[0] for edge in train_edges]
    index_2 = [edge[1] for edge in train_edges]
    values = [1]*len(p_edges) + [-1]*len(n_edges)
    shaping = (node_count, node_count)
    signed_A = sparse.csr_matrix(sparse.coo_matrix((values, (index_1, index_2)),
                                                   shape=shaping,
                                                   dtype=np.float32))
(Pdb) max(index_1)
6055
(Pdb) max(index_2)
6055
(Pdb) len(values)
379398
(Pdb) node_count
5543

refine def read_graph(args)
edges["ncount"] = len(set([edge[0] for edge in dataset]+[edge[1] for edge in dataset]))
-->
edges["ncount"] = np.max(np.array(dataset))+1

(3) sgcn.py 
self.y
setup_dataset
        #self.y = np.array([0 if i < int(self.ecount/2) else 1 for i in range(self.ecount)]+[2]*(self.ecount*2))
        self.y = np.array([0]*self.positive_edges.size(1) + [1]*self.negative_edges.size(1) +[2]*(self.ecount*2))
