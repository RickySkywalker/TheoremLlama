# TheoremLlama

This is the official repository for all the code of TheoremLlama for uploading the training and inference code of the paper. The code will be available soon. You can now try our model checkpoints and dataset at:
 - TheoremLlama model at: https://huggingface.co/RickyDeSkywalker/TheoremLlama
 - *Open Bootstrapped Theorem* (OBT) dataset at: https://huggingface.co/datasets/RickyDeSkywalker/OpenBootstrappedTheorem

[\[ArXiv\]](https://arxiv.org/abs/2407.03203)

## Updates:
- 10th Oct 2024: update MiniF2F evaluation code


## Test on MiniF2F dataset
Setup the environment:
```bash
pip install -r requirements.txt
```

Setup the env variable in `eval_MiniF2F.py`. Default settings are:
```python
CUDA_DEVICE_ID=1
BATCH_SIZE=4
PROOF_NUM_PER_THEOREM=32
MODEL_ID = "RickyDeSkywalker/TheoremLlama"
CKPT_PATH = "./Generated_proof_ckpts/MiniF2F_Valid/test_output"
SAVE_PATH = './Generated_proof/MiniF2F_Valid/test_output'
dataset_split = "test"
```

Run the evaluation code:
```bash
python eval_MiniF2F.py
```

You can find the generated proof in `SAVE_PATH` and the ckpts for the proof in `CKPT_PATH`.



## Citation:
```
@misc{wang2024theoremllamatransforminggeneralpurposellms,
      title={TheoremLlama: Transforming General-Purpose LLMs into Lean4 Experts}, 
      author={Ruida Wang and Jipeng Zhang and Yizhen Jia and Rui Pan and Shizhe Diao and Renjie Pi and Tong Zhang},
      year={2024},
      eprint={2407.03203},
      archivePrefix={arXiv},
      primaryClass={cs.FL},
      url={https://arxiv.org/abs/2407.03203}, 
}
```

## Acknowledgement
This research used both the DeltaAI advanced computing and data resource, which is supported by the National Science Foundation (award OAC 2320345) and the State of Illinois, and the Delta advanced computing and data resource which is supported by the National Science Foundation (award OAC 2005572) and the State of Illinois. Delta and DeltaAI are joint efforts of the University of Illinois Urbana-Champaign and its National Center for Supercomputing Applications.
