import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
import math
from torch_geometric.nn import MLP
from typing import Optional, Union, Callable, Literal, List


def glorot(tensor):
    """
    Glorot (Xavier) initialization for weights.

    Initializes the input tensor with values using a uniform distribution
    scaled according to the fan-in and fan-out of the tensor.

    Args:
        tensor (torch.Tensor): The tensor to initialize
    """
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    """
    Zero initialization for bias terms.

    Fills the input tensor with zeros.

    Args:
        tensor (torch.Tensor): The tensor to initialize
    """
    if tensor is not None:
        tensor.data.fill_(0)


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class SpectConv(MessagePassing):
    """
    Spectral Convolution layer for Graph Neural Networks.

    This layer implements spectral graph convolution using edge attributes as the
    spectral coefficients. It supports depthwise separable convolutions and self-connections.

    Args:
        in_channels (int): Size of each input sample
        out_channels (int): Size of each output sample
        K (int): Number of spectral coefficients (default: 1)
        selfconn (bool): Whether to include self-connections (default: True)
        depthwise (bool): Whether to use depthwise separable convolution (default: False)
        bias (bool): If set to False, the layer will not learn an additive bias (default: True)
        **kwargs: Additional arguments for the MessagePassing base class
    """

    def __init__(self, in_channels, out_channels, K=1, selfconn=True,
                 depthwise=False, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SpectConv, self).__init__(**kwargs)

        assert K > 0, "Number of spectral coefficients must be positive"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise = depthwise
        self.selfconn = selfconn

        # Adjust K if self-connection is used
        if self.selfconn:
            K = K + 1

        # Set up weights based on convolution type
        if self.depthwise:
            self.DSweight = Parameter(torch.Tensor(K, in_channels))
            self.nsup = K
            K = 1

        # Main filter weights
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        # Optional bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Glorot initialization and set biases to zero."""
        glorot(self.weight)
        if hasattr(self, 'bias') and self.bias is not None:
            zeros(self.bias)
        if self.depthwise:
            zeros(self.DSweight)

    def forward(self, x, edge_index, edge_attr, edge_weight=None,
                batch=None, lambda_max=None):
        """
        Forward pass of the spectral convolution layer.

        Args:
            x (Tensor): Node feature matrix
            edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges]
            edge_attr (Tensor): Edge feature matrix with shape [num_edges, K]
            edge_weight (OptTensor, optional): One-dimensional edge weight tensor
            batch (OptTensor, optional): Batch vector
            lambda_max (OptTensor, optional): Largest eigenvalue for each graph

        Returns:
            Tensor: Updated node features
        """
        Tx_0 = x
        out = 0

        if not self.depthwise:
            end_itr = self.weight.size(0)

            # Handle self-connection if enabled
            if self.selfconn:
                out = torch.matmul(Tx_0, self.weight[-1])
                end_itr -= 1

                # Process each spectral coefficient
            for i in range(0, end_itr):
                h = self.propagate(edge_index, x=Tx_0, norm=edge_attr[:, i], size=None)
                out = out + torch.matmul(h, self.weight[i])
        else:
            # Depthwise separable convolution path
            end_itr = self.nsup

            # Handle self-connection if enabled
            if self.selfconn:
                out = Tx_0 * self.DSweight[-1]
                end_itr -= 1

                # First coefficient has special handling
            out = out + (1 + self.DSweight[0:1, :]) * self.propagate(
                edge_index, x=Tx_0, norm=edge_attr[:, 0], size=None)

            # Process remaining coefficients
            for i in range(1, end_itr):
                out = out + self.DSweight[i:i + 1, :] * self.propagate(
                    edge_index, x=Tx_0, norm=edge_attr[:, i], size=None)

            # Final projection
            out = torch.matmul(out, self.weight[0])

            # Add bias if enabled
        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, norm):
        """
        Message function that defines how node features are aggregated.

        Args:
            x_j (Tensor): Node features of neighbors
            norm (Tensor): Normalization/weight for the edges

        Returns:
            Tensor: The message passed along each edge
        """
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        """String representation of the layer."""
        return '{}({}, {}, K={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0))


class ML3Layer(torch.nn.Module):
    def __init__(self, learnedge=True, nedgeinput=1, nedgeoutput=1, ninp=1, nout1=30, nout2=2):
        """
        ML3Layer: A message-passing layer for GNNs with higher-order expressivity

        Args:
            learnedge (bool): Whether to learn edge features transformations
            nedgeinput (int): Dimension of input edge features
            nedgeoutput (int): Dimension of output edge features
            ninp (int): Dimension of input node features
            nout1 (int): Dimension of convolution output features
            nout2 (int): Dimension of skip connection output features (set to 0 to disable)
        """
        super(ML3Layer, self).__init__()

        self.learnedge = learnedge
        self.nout2 = nout2

        # Edge feature transformation network (if enabled)
        if self.learnedge:
            self.edge_transform = nn.ModuleDict({
                'linear1': torch.nn.Linear(nedgeinput, 2 * nedgeinput, bias=False),
                'linear2': torch.nn.Linear(nedgeinput, 2 * nedgeinput, bias=False),
                'linear3': torch.nn.Linear(nedgeinput, 2 * nedgeinput, bias=False),
                'linear4': torch.nn.Linear(4 * nedgeinput, nedgeoutput, bias=False)
            })
        else:
            nedgeoutput = nedgeinput

        # Main spectral convolution
        self.conv1 = SpectConv(ninp, nout1, nedgeoutput, selfconn=False)

        # Node feature skip connection (if enabled)
        if nout2 > 0:
            self.skip_connection = nn.ModuleDict({
                'linear1': torch.nn.Linear(ninp, nout2),
                'linear2': torch.nn.Linear(ninp, nout2)
            })

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the ML3Layer

        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Graph connectivity in COO format
            edge_attr (torch.Tensor): Edge features

        Returns:
            torch.Tensor: Updated node features
        """
        # Transform edge features if enabled
        if self.learnedge:
            linear_part = F.relu(self.edge_transform['linear1'](edge_attr))
            gated_part = torch.tanh(self.edge_transform['linear2'](edge_attr)) * \
                         torch.tanh(self.edge_transform['linear3'](edge_attr))
            tmp = torch.cat([linear_part, gated_part], dim=1)
            edge_attr = F.relu(self.edge_transform['linear4'](tmp))

        # Apply convolution
        conv_output = F.relu(self.conv1(x, edge_index, edge_attr))

        # Apply skip connection if enabled
        if self.nout2 > 0:
            skip_output = torch.tanh(self.skip_connection['linear1'](x)) * \
                          torch.tanh(self.skip_connection['linear2'](x))
            return torch.cat([conv_output, skip_output], dim=1)
        else:
            return conv_output


class GNNML3(torch.nn.Module):
    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            global_dim: int,
            hidden_dim: int = 30,
            num_layers: int = 4,
            num_encoder_layers: int = 2,
            output_dim: int = 2,
            shift_predictor_hidden_dim: Union[int, List[int]] = 32,
            shift_predictor_layers: int = 1,
            embedding_type: Literal["node", "global", "combined"] = "node",
            act: Union[str, Callable] = "relu",
            norm: Optional[str] = "batch_norm",
            dropout: float = 0.0,
            **kwargs
    ):
        super().__init__()

        # Store model configuration parameters
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_encoder_layers = num_encoder_layers
        self.embedding_type = embedding_type
        self.act = act
        self.norm = norm
        self.dropout = dropout

        # Initialize node and edge encoders as None (will be initialized in forward)
        self.node_encoder = None
        self.edge_encoder = None

        # First layer has different input dimension
        self.layers = torch.nn.ModuleList()
        self.layers.append(
            ML3Layer(
                learnedge=True,
                nedgeinput=edge_dim,
                nedgeoutput=edge_dim,
                ninp=node_dim,
                nout1=self.hidden_dim,
                nout2=self.output_dim
            )
        )

        # Remaining layers take output of previous layer as input
        nin = self.hidden_dim + self.output_dim  # Combined dimension of layer outputs
        for _ in range(num_layers - 1):
            self.layers.append(
                ML3Layer(
                    learnedge=True,
                    nedgeinput=edge_dim,
                    nedgeoutput=edge_dim,
                    ninp=nin,
                    nout1=self.hidden_dim,
                    nout2=self.output_dim
                )
            )

        # Determine input dimension for shift predictor
        if embedding_type == "node":
            shift_predictor_in_dim = nin  # Use node embedding dimension
        elif embedding_type == "global":
            shift_predictor_in_dim = global_dim
        else:  # combined
            shift_predictor_in_dim = nin + global_dim

        # Create channel list for shift predictor MLP
        if isinstance(shift_predictor_hidden_dim, int):
            channel_list = [shift_predictor_in_dim] + [shift_predictor_hidden_dim] * (shift_predictor_layers - 1) + [1]
        else:
            channel_list = [shift_predictor_in_dim] + list(shift_predictor_hidden_dim) + [1]

        # Shift predictor MLP
        self.shift_predictor = MLP(
            channel_list=channel_list,
            act=act,
            norm=None,  # norm,
            dropout=dropout
        )

    def forward(self, x, edge_index, edge_attr, batch=None, u=None):
        """
        Forward pass of the GNNML3 model.

        Args:
            x: Node features
            edge_index: Graph connectivity in COO format
            edge_attr: Edge features
            batch: Batch vector
            u: Global features (optional)

        Returns:
            torch.Tensor: Graph-level prediction
            dict: Empty dictionary (for compatibility)
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initialize node encoder if not already initialized
        if self.node_encoder is None:
            node_in_channels = x.size(1)  # Infer input dimension from node features
            self.node_encoder = MLP(
                in_channels=node_in_channels,
                hidden_channels=self.hidden_dim,
                out_channels=self.node_dim,
                num_layers=self.num_encoder_layers,
                act=self.act,
                norm="BatchNorm",  # self.norm,
                dropout=self.dropout
            ).apply(init_weights).to(x.device)  # Move node_encoder to the same device as x

        # Initialize edge encoder if not already initialized
        if self.edge_encoder is None:
            edge_in_channels = edge_attr.size(1)  # Infer input dimension from edge features
            self.edge_encoder = MLP(
                in_channels=edge_in_channels,
                hidden_channels=self.hidden_dim,
                out_channels=self.edge_dim,
                num_layers=self.num_encoder_layers,
                act=self.act,
                norm="BatchNorm",  # self.norm,
                dropout=self.dropout
            ).apply(init_weights).to(x.device)  # Move edge_encoder to the same device as x

        # Encode node and edge features
        x = self.node_encoder(x)  # Encode node features
        edge_attr = self.edge_encoder(edge_attr)  # Encode edge features

        # Apply each layer sequentially
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        return self.shift_predictor(x), {}

        # Pooling strategy based on embedding type
        #if self.embedding_type == "node":
        #    # Use node features for prediction (pooled)
        #    pooled = global_add_pool(x, batch)
        #    return self.shift_predictor(pooled), {}
        #elif self.embedding_type == "global":
        #    # Use global features for prediction
        #    if u is not None:
        #        return self.shift_predictor(u), {}
        #    else:
        #        # Fallback to node features if no global features are provided
        #        pooled = global_add_pool(x, batch)
        #        return self.shift_predictor(pooled), {}
        #else:  # combined
        #    # Combine node and global features
        #    pooled = global_add_pool(x, batch)
        #    if u is not None:
        #        combined = torch.cat([pooled, u], dim=1)
        #        return self.shift_predictor(combined), {}
        #    else:
        #        return self.shift_predictor(pooled), {}
