import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

# 구현하세요!


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        print("check point")
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        tokenized_corpus = [tokenizer.encode(sentence, add_special_tokens=False) for sentence in corpus]
        print("num_epochs: ", num_epochs)
        print("tokenized: ",len(tokenized_corpus))
        for epoch in range(num_epochs):
            total_loss = 0.0
            for tokens in tokenized_corpus:
                if len(tokens) < 2 * self.window_size + 1:
                    continue
                if self.method == "cbow":
                    loss = self._train_cbow(tokens, criterion, optimizer)
                else:
                    loss = self._train_skipgram(tokens, criterion, optimizer)
                total_loss += loss
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    def _train_cbow(
        self,
        tokens: list[int],
        criterion: nn.CrossEntropyLoss,
        optimizer: torch.optim.Adam
        # 구현하세요!
    ) -> float:
        loss = 0
        for i in range(self.window_size, len(tokens) - self.window_size):
            context = tokens[i - self.window_size:i] + tokens[i+1:i + 1 + self.window_size]
            target = tokens[i]
            
            context_tensor = LongTensor(context).unsqueeze(0)
            target_tensor = LongTensor([target])
            
            optimizer.zero_grad()
            context_embeds = self.embeddings(context_tensor).mean(dim=1)
            output = self.weight(context_embeds)
            
            batch_loss = criterion(output, target_tensor)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
        return loss

    def _train_skipgram(
        self,
        tokens: list[int],
        criterion: nn.CrossEntropyLoss,
        optimizer: torch.optim.Adam
    ) -> float:
        loss = 0
        for i in range(self.window_size, len(tokens) - self.window_size):
            target = tokens[i]
            context = tokens[i - self.window_size:i] + tokens[i + 1:i + 1 + self.window_size]
            
            target_tensor = LongTensor([target])
            context_tensor = LongTensor(context)
            
            optimizer.zero_grad()
            target_embed = self.embeddings(target_tensor)
            output = self.weight(target_embed)
            
            output = output.expand(len(context_tensor), -1)
            batch_loss = criterion(output, context_tensor)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
        
        return loss