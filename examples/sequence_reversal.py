"""Example: Decoder-only transformer on sequence-reversal task."""

import torch
import torch.nn.functional as F
from torch.utils import data
from typing import Union

from yati.model.decoder_only_transformer import DecoderOnlyTransformer
from yati.params.decoder_only_transformer_params import DecoderOnlyTransformerParams
from yati.params.encoder_decoder_params import EncoderDecoderParams


class SequenceReversalDataset(data.Dataset):
    """Sequence-reversal dataset."""

    def __init__(  # noqa: DCO010
        self, size: int, seq_len: int, num_classes: int, device: Union[torch.device, str]
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.data = torch.randint(num_classes, size=(size, seq_len))
        self.data = self.data.to(device)

    def __len__(self):  # noqa: DCO010
        return self.data.shape[0]

    def __getitem__(self, idx: int):  # noqa: DCO010
        x = self.data[idx]
        y = torch.flip(x, dims=(0,))
        return x, y


def main():
    """Train decoder-only transformer on sequence-reversal task."""
    # Set device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")

    # Create sequence-reversal dataset
    dataset = SequenceReversalDataset(50000, 16, 10, device)
    train_loader = data.DataLoader(
        dataset, batch_size=128, shuffle=True, drop_last=True, pin_memory=True
    )

    # Create decoder-only transformer parameters
    d_model = dataset.num_classes

    decoder_params = EncoderDecoderParams(
        d_input=d_model,
        h=1,
        d_k=16,
        d_v=16,
        d_ff=32,
        p_dropout=0.0,
    )

    params = DecoderOnlyTransformerParams(
        d_model=d_model,
        max_seq_len=dataset.seq_len,
        input_num_embeddings=dataset.num_classes,
        output_num_embeddings=dataset.num_classes,
        decoder_stack_num_layers=1,
        decoder_params=decoder_params,
        p_dropout=0.0,
        tie_weight_matrices=False,
    )

    # Create decoder-only transformer
    transformer = DecoderOnlyTransformer(params)
    transformer = transformer.to(device)

    # Set up for training loop
    num_epochs = 10
    num_steps = num_epochs * len(dataset)
    update_steps = 100
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-5)

    # Run basic training loop
    for step in range(num_steps):
        # Get next batch
        x, y = next(iter(train_loader))
        y_hat_logits = transformer(x)

        # Compute loss on batch
        y_hat_logits = y_hat_logits.view(-1, y_hat_logits.shape[-1])
        loss = F.cross_entropy(y_hat_logits, y.view(-1))

        # Gradient descent update
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Print current loss to console
        if (step + 1) % update_steps == 0:
            print(f"{step + 1}/{num_steps} loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
