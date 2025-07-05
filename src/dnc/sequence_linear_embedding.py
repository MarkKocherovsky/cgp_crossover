import torch
import torch.nn as nn

class SequenceLinearEmbedding(nn.Module):
    """
    Projects a sequence of fully continuous (float) genes using a single linear layer.
    Suitable for cases where all fields are real-valued (even if originally categorical).
    """

    def __init__(self, input_dim, embedding_dim):
        """
        :param input_dim: Number of fields per gene (e.g., 6 for CGP: NodeType, Value, Operator, Operand1, Operand2, Active)
        :param embedding_dim: Output dimension of the embedding
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        """
        :param x: Tensor of shape (..., input_dim) with float values
        :return: Tensor of shape (..., embedding_dim)
        """
        if not torch.is_floating_point(x):
            x = x.float()
        #print("Embedding input shape:", x.shape)
        #print("Expected input_dim:", self.linear.in_features)
        return self.linear(x)

