"""Minimal Gemma + PyTorch Transformer text classifier for 3-way decisions."""

import hashlib
from typing import Iterable, Optional

import torch
from torch import Tensor, nn
from sentence_transformers import SentenceTransformer

# Define a function to automatically get the best available device
def get_best_device():
    # Check for CUDA (NVIDIA GPU) availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # Check for MPS (Apple Silicon GPU) availability
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon MPS device")
    # Fallback to CPU
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


class SimpleGemmaTransformerClassifier(nn.Module):
    """Wraps Google Gemma embeddings with a tiny Transformer encoder head."""
    def __init__(
        self,
        num_classes: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_cache = {}
        self.device = get_best_device()
        self.embedding_model = SentenceTransformer("google/embeddinggemma-300m", device=str(self.device))
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.project = nn.Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid()
        )

        self.to(self.device)

    def embedding(self, text: str) -> Tensor:
        """Compute or retrieve cached Gemma embedding for a single text."""
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key not in self.embedding_cache:
            features = self.embedding_model.tokenize([text])
            features = {name: tensor.to(self.device) for name, tensor in features.items()}

            with torch.no_grad():
                outputs = self.embedding_model(features)

            token_embeddings = outputs["token_embeddings"]
            attention_mask = features["attention_mask"]

            mask = attention_mask.unsqueeze(-1)
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            self.embedding_cache[key] = pooled.squeeze(0)

        return self.embedding_cache[key]

    def forward(self, texts: Iterable[str]) -> Tensor:
        # Use cached embeddings for each text
        embeddings = torch.stack([self.embedding(text) for text in texts])

        # Project embeddings to hidden dimension
        hidden = self.project(embeddings).unsqueeze(1)  # Add sequence dimension

        # Pass through transformer (sequence length = 1)
        encoded = self.transformer(hidden)

        # Remove sequence dimension
        pooled = encoded.squeeze(1)

        return self.classifier(pooled)


def example_forward() -> None:
    model = SimpleGemmaTransformerClassifier()
    texts = ["Buy the dip?", "Market is flat."]
    logits = model(texts)
    print("Logits:", logits.detach().cpu())
    print("Probabilities:", logits.softmax(dim=-1).detach().cpu())


def example_train(epochs: int = 2, lr: float = 1e-4) -> None:
    """
    Tiny illustrative training loop. Replace ``texts`` and ``labels`` with
    real data when integrating into your pipeline.
    """
    model = SimpleGemmaTransformerClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    texts = [
        "Strong earnings beat expectations.",
        "Regulatory concerns weigh on the sector.",
        "Momentum indicators point to consolidation.",
    ]
    labels = torch.tensor([2, 0, 1], device=model.device)  # class ids: buy, sell, hold

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        probs = logits.softmax(dim=-1).detach().cpu()
        print(f"Epoch {epoch + 1}: loss={loss.item():.4f} probs={probs}")


if __name__ == "__main__":
    example_forward()
