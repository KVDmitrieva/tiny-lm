# TinyStories
Data was downloaded from [HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main). Run 
```bash
chmod +x setup.sh
./setup.sh
```
in order to download and preprocess data. Then run 

```python
python3 main.py -k WANDB_KEY
```
to start trainig.

To test model download checkpoints with `test_setup.sh` (after `setup.sh`) and run

```python
python3 test.py -p YOUR_PROMPT
```
to get model output.
