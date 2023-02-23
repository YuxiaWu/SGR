1、utils--> sgcn_utils
class SignedGraphConvolutionalNetwork(torch.nn.Module):
    def __init__(self, device, args, X):

2、del the X in __init__
class SignedGraphConvolutionalNetwork(torch.nn.Module):
    def __init__(self, device, args, node_count):

3、set_layers

4、def forward(self, positive_edges, negative_edges, target, X):