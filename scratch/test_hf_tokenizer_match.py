import mobileclip
from transformers import CLIPTokenizer

tok_mc = mobileclip.get_tokenizer("mobileclip_s1")
tok_hf = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

test_sentences = [
    "a dog playing in the park",
    "a red sports car parked in front of a modern house",
    "two cats sleeping on a blue sofa with a striped pillow",
    "a clock on the top of the building",
    "a kitchen with stainless steel appliances and granite countertops"
]

all_match = True
for s in test_sentences:
    t_mc = tok_mc([s])[0].numpy().tolist()
    t_hf = tok_hf(s, padding="max_length", max_length=77, truncation=True)["input_ids"]
    match = (t_mc == t_hf)
    print(f"Match: {match} for '{s[:35]}...'")
    if not match:
        all_match = False
        print(f"  MC: {t_mc[:10]}")
        print(f"  HF: {t_hf[:10]}")

print(f"All sentences token-for-token identical: {all_match}")
