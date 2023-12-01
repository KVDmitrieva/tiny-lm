def process_file(input_file, output_file):
    with open(input_file, 'r', encoding="utf-8") as f:
        texts = f.readlines()

    with open(output_file, 'w', encoding="utf-8") as f:
        new_story = True
        for t in texts:
            if "endoftext" in t:
                f.write('\n')
                new_story = True
            else:
                if not new_story:
                    f.write(' ')
                f.write(t[:-1])
                new_story = False

def main():
    process_file("data/train_raw.txt", "data/train.txt")
    process_file("data/val_raw.txt", "data/val.txt")


if __name__ == '__main__':
    main()
