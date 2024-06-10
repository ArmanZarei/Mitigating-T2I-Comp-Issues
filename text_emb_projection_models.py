import torch


class CLIPTextEmbeddingLinearProjector(torch.nn.Module):
    def __init__(self, dim, initialization_type):
        super().__init__()

        self.linear = torch.nn.Linear(dim, dim)

        if initialization_type == 'zeros':
            self.linear.weight.data.zero_()
            self.linear.bias.data.zero_()
        elif initialization_type == 'eye':
            torch.nn.init.eye_(self.linear.weight)
            torch.nn.init.zeros_(self.linear.bias)
        elif initialization_type == 'default':
            pass
        elif initialization_type == 'xavier':
            torch.nn.init.xavier_uniform_(self.linear.weight)
            torch.nn.init.zeros_(self.linear.bias) # or 0.01?
        else:
            raise Exception('Invalid initialization type')
    
    def forward(self, x):
        return self.linear(x)


class CLIPTextEmbeddingLinearSkipProjector(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.linear = torch.nn.Linear(dim, dim)

        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
    
    def forward(self, x):
        return self.linear(x) + x


class CLIPTextEmbeddingMLPProjector(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim),
        )

        # initialize to zero
        for param in self.network.parameters():
            torch.nn.init.constant_(param, 0)
    
    def forward(self, x):
        return self.network(x) + x


class WindowAwareLinearProjection(torch.nn.Module):
    def __init__(self, text_embeddings_dim: int, window_size: int):
        super().__init__()

        self.emb_dim = text_embeddings_dim

        self.projection = torch.nn.Conv1d(
            in_channels=text_embeddings_dim,
            out_channels=text_embeddings_dim,
            kernel_size=window_size, 
            padding='same',
            padding_mode='zeros'
        )

        self.projection.weight.data.zero_()
        self.projection.bias.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[2] == self.emb_dim

        return x + self.projection(x.permute(0, 2, 1)).permute(0, 2, 1)
