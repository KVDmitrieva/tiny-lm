# TinyStories
Checkpoint with training pipeline

Data was downloaded from [HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main) with
```bash
wget "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt?download=true" -O train_raw.txt
wget "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt?download=true" -O val_raw.txt
```
and then processed with function:
```python
def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        texts = f.readlines()

    all_texts = []
    tmp_text = []
    for t in texts:
        if "endoftext" in t:
            all_texts.append(' '.join(tmp_text))
            tmp_text = []
        elif len(t) > 1:
            tmp_text.append(t[:-1])

    if len(tmp_text) > 0:
        all_texts.append(' '.join(tmp_text))
        tmp_text = []

    with open(output_file, 'w') as f:
        f.writelines('\n'.join(all_texts))
```

training script is working with processed files. For final model processing will be implemented in `main.py`.