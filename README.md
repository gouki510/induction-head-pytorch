# Induction-head

This implementation is a PyTorch-based reproduction of the following paper.  
THE MECHANISTIC BASIS OF DATA DEPENDENCE AND ABRUPT LEARNING IN AN IN-CONTEXT CLASSIFICATION TASK:https://arxiv.org/abs/2312.03002  


## Environment
```bash
pip install -r requirements.txt
```
## Notebook

You can easily check the implementation in a notebook. `notebooks/icl_induction_head.ipynb`  

## Run
```bash
python train.py --ways B --num_class K --eps Epsilon --alpha Alpha --gpu GPU
```

The results can be viewed on Weights & Biases (wandb).  


### Reproduce
You can reproduce figures in the paper. 
```bash
sh scripts/hoge.sh
```

