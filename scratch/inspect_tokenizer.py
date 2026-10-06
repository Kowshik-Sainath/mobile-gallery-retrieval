import mobileclip
import inspect

tok = mobileclip.get_tokenizer("mobileclip_s1")
print(f"Tokenizer class: {tok.__class__.__name__}")
print(f"Tokenizer module: {tok.__class__.__module__}")
print(f"Attributes: {dir(tok)}")

# Check vocab size and files
if hasattr(tok, 'vocab_size'):
    print(f"Vocab size: {tok.vocab_size}")
if hasattr(tok, 'encoder'):
    print(f"Encoder dict len: {len(tok.encoder)}")
if hasattr(tok, 'bpe_ranks'):
    print(f"BPE ranks len: {len(tok.bpe_ranks)}")

# Test sample encoding
sample_text = "a dog playing in the park"
tokens = tok([sample_text])
print(f"Sample: '{sample_text}' -> shape {tokens.shape}")
print(f"Tokens: {tokens[0, :10]}")
