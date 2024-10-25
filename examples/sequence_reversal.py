"""Example: Decoder-only transformer on sequence-reversal task."""

import torch
import torch.nn.functional as F
from torch.utils import data

from yati.model.decoder_only_transformer import DecoderOnlyTransformer
from yati.params.decoder_only_transformer_params import DecoderOnlyTransformerParams
from yati.params.encoder_decoder_params import EncoderDecoderParams


class SequenceReversalDataset(data.Dataset):
    """Sequence-reversal dataset."""

    def __init__(self, size: int, seq_len: int, num_classes: int) -> None:  # noqa: DCO010
        super().__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.data = torch.randint(num_classes, size=(size, seq_len))

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
    dataset = SequenceReversalDataset(100000, 16, 10)
    train_loader = data.DataLoader(
        dataset, batch_size=128, shuffle=True, drop_last=True, pin_memory=True
    )

    # Create decoder-only transformer parameters
    d_model = dataset.num_classes

    # fmt: off
    decoder_params = EncoderDecoderParams(
        d_input=d_model,  # Input dimension; equal to embedding dimension
        h=1,              # Use single-head attention
        d_k=16,           # Number of rows in {Q, K}-matrix
        d_v=16,           # Number of rows in V-matrix
        d_ff=32,          # Hidden layer dimension
        p_dropout=0.0,    # Disable dropout
    )
    # fmt: on

    # fmt: off
    params = DecoderOnlyTransformerParams(
        d_model=d_model,                            # Embedding dimension
        max_seq_len=dataset.seq_len,                # Fixed input sequence length
        input_num_embeddings=dataset.num_classes,   # Input vocabulary size
        output_num_embeddings=dataset.num_classes,  # Output vocabulary size
        decoder_stack_num_layers=1,                 # Number of layers in decoder stack
        decoder_params=decoder_params,              # Decoder layer params
        p_dropout=0.0,                              # Disable dropout
        tie_weight_matrices=False,                  # Do not tie weight matrices
        apply_causal_mask=False,                    # Do not apply causal mask
                                                    # -- Reversal operation is non-causal
    )
    # fmt: on

    # Create decoder-only transformer
    transformer = DecoderOnlyTransformer(params)
    transformer = transformer.to(device)

    # Set up for training loop
    num_epochs = 20
    num_batches_per_epoch = len(dataset) // train_loader.batch_size
    update_steps = 100
    optimizer = torch.optim.Adam(transformer.parameters(), lr=5e-4)
    save_model = True
    save_filename = "sequence_reversal_model.pt"

    # Run super-simple training loop
    for epoch in range(num_epochs):
        for batch in range(num_batches_per_epoch):
            # Get next batch
            x, y = next(iter(train_loader))
            x, y = x.to(device), y.to(device)

            # Compute model output, interpreted as logits
            y_hat_logits = transformer(x)

            # Compute loss on batch
            # - Tensors are reshaped since cross_entropy() expects shape (b, n)
            y_hat_logits = y_hat_logits.view(-1, y_hat_logits.shape[-1])
            loss = F.cross_entropy(y_hat_logits, y.view(-1))

            # Gradient descent update
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Print current loss to console
            if (batch + 1) % update_steps == 0:
                print(
                    f"epoch={epoch}, "
                    f"batch={batch + 1}/{num_batches_per_epoch}, "
                    f"loss: {loss.item():.4f}"
                )

        # Save model checkpoint after each epoch
        if save_model:
            print(f"epoch={epoch}, saving model checkpoint to {save_filename}")
            torch.save(transformer, save_filename)


if __name__ == "__main__":
    main()
