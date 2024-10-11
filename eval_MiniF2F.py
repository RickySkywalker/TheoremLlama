from Prove_writer import Prove_writer
import utils
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
CUDA_DEVICE_ID=1
BATCH_SIZE=4
PROOF_NUM_PER_THEOREM=32
MODEL_ID = "RickyDeSkywalker/TheoremLlama"
CKPT_PATH = "./Generated_proof_ckpts/MiniF2F_Valid/test_output"
SAVE_PATH = './Generated_proof/MiniF2F_Valid/test_output'

dataset_split = "test"      # test or vaild, which to use for testing


def main_generate_proof():
    utils.check_folder_exit(SAVE_PATH)
    utils.check_folder_exit(CKPT_PATH)
    
    example_ls = utils.load_data_ls("./eval_dataset/MiniF2F_valid_partial_withProof_commented.json", has_proof=True, has_comment=True)
    if dataset_split == "test":
        data_ls_to_test = utils.load_data_ls("./eval_dataset/MiniF2F_test_Lean4.json")
    elif dataset_split == "valid":
        data_ls_to_test = utils.load_data_ls("./eval_dataset/MiniF2F_valid_Lean4.json")
    else:
        raise NotImplementedError("dataset split {dataset_split} not implemented")

    

    print(f"current example list have {len(example_ls)} examples")

    model_id = MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:"+str(CUDA_DEVICE_ID),
    )
    terminators = [tokenizer.eos_token_id, 
                tokenizer.convert_tokens_to_ids("<|eot_id|>"), 
                tokenizer.convert_tokens_to_ids("<|reserved_special_token_26|>")
    ]
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    prove_writer = Prove_writer(model, tokenizer, terminators, example_ls, example_num=14)

    test_proof_ls = prove_writer.generate_proof_dataset(data_ls_to_test, 
                                                        proof_num=PROOF_NUM_PER_THEOREM, 
                                                        temperature=0.9,
                                                        variable_tempreature=0.6,
                                                        top_p=0.9,
                                                        repetition_penalty=1.1,
                                                        batch_size=BATCH_SIZE, 
                                                        ckpt_path=CKPT_PATH)
    # validation_proof_ls = prove_writer.generate_proof_dataset(validation_data_ls,
    #                                                           proof_num=300,
    #                                                           temperature=1.0,
    #                                                           variable_tempreature=0.6)
    utils.write_to_json(f"{SAVE_PATH}/MiniF2F_{data_ls_to_test}_Lean4_proof.json", test_proof_ls)
    # utils.write_to_json(validation_proof_ls, "./Generated_proof/Llama-3-8B-Instruct_unfintuned/MiniF2F_validation_Lean4_proof.json")

if __name__ == "__main__":
    main_generate_proof()