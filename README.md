# yati (Yet Another Transformer Implementation)

This project is an implementation of the (encoder/decoder) transformer
architecture, as described in "Attention is All You Need" (AIAYN).
There is also an implementation of the decoder-only architecture.

Aside from being an interesting learning exercise, the main goal of the project
is to be as faithful as possible to the description in AIAYN.  This includes the
choice of terminology and notation/variable names.  Any assumptions are clearly 
documented.

# Installation 

## Using `uv`
```shell
git clone https://github.com/dscamiss/yati && cd yati
uv pip install .
```

## Using `pip`
```shell
git clone https://github.com/dscamiss/yati && cd yati
pip install -r requirements.txt
```

# Example usage

See `examples` directory.

# TODO
- Add more training examples
