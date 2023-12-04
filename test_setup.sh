#!/bin/bash

echo "Download model checkpoint"
gdown "https://drive.google.com/u/0/uc?id=1q8Uuce7J0eqpRk1nqaOBdaWvIVDUgv6u" -O model.pth
gdown "https://drive.google.com/u/0/uc?id=1ythD4rgjT5a_xaAvZQdrxMDE4G0fHmXs" -O tiny_stories.model
gdown "https://drive.google.com/u/0/uc?id=1XWZ1WDEd9RDLMQl6XxWelz2uG0iKICTh" -O tiny_stories.vocab
