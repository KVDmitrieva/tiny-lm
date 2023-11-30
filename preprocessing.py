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

    with open(output_file, 'w') as f:
        f.writelines('\n'.join(all_texts))


if __name__ == '__main__':
    process_file("data/train_raw.txt", "data/train.txt")
    process_file("data/val_raw.txt", "data/val.txt")
