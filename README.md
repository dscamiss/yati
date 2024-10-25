# yati (Yet Another Transformer Implementation)

This project is an implementation of the (encoder/decoder) transformer
architecture, as described in "Attention is All You Need" (AIAYN).
There is also an implementation of the decoder-only architecture.

The main goal of the project (aside from being an interesting learning
exercise) is to be as faithful as possible to the description in AIAYN.
Any assumptions are clearly documented.

# Installation using `uv`
```shell
uv venv my_env
source my_env/bin/activate
git clone https://github.com/dscamiss/yati && cd yati
uv pip install .
```

# Example usage

See `examples` directory.

# TODO
- Add more training examples
