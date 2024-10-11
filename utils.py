import json
import torch
from datasets import Dataset
import random
import re
import os

def check_folder_exit(folder_path):
    if not os.path.exists(folder_path):
        # 如果文件夹不存在，则创建它
        os.makedirs(folder_path)
        print(f"creating '{folder_path}'")
    else:
        print(f"folder '{folder_path}' exists")


def write_to_json(filePath, data):
    with open(filePath, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def write_to_file(filePath, txt_to_write):
    with open(filePath, 'w', encoding='utf-8') as file:
        file.write(txt_to_write)

# This function will read all json files in the folder and return a list of all json file
def read_json_in_folder(folder_path):
    json_list = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否是 JSON 文件
        if filename.endswith(".json"):
            # 构造文件的完整路径
            file_path = os.path.join(folder_path, filename)
            curr_file = read_from_json(file_path)
            json_list += [curr_file]
    return json_list

def read_from_json(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def get_best_device(i=0):
    if torch.cuda.is_available():
        return torch.device("cuda:" + str(i))
    elif torch.backends.mps.is_built():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def load_dataset(dataset_path,
                 begin_dix=0,
                 protion_dataset_to_take=1.0):
    data = read_from_json(dataset_path)
    if protion_dataset_to_take < 1.0:
        data = [curr for curr in data if random.random() < protion_dataset_to_take]
    dataset_keys = list(data[0].keys())
    dataset_dic = {key: [] for key in dataset_keys}
    for curr_record in data[begin_dix:]:
        for key in dataset_keys:
            dataset_dic[key] += [curr_record[key]]
    return Dataset.from_dict(dataset_dic)

def extract_thm_name(lean_code: str) -> str:
    # 定义正则表达式匹配theorem名称
    match = re.search(r'theorem\s+([a-zA-Z0-9_]+)', lean_code)
    if match:
        return match.group(1)
    return None

def _test():
    dataset = [
        {
            'thmName': 'recC_intro',
            'thmStatement': "theorem recC_intro {motive : (a : α) → Acc r a → Sort v}\n    (intro : (x : α) → (h : ∀ (y : α), r y x → Acc r y) →\n     ((y : α) → (hr : r y x) → motive y (h y hr)) → motive x (intro x h))\n    {a : α} (h : ∀ (y : α), r y a → Acc r y) :\n    recC intro (Acc.intro _ h) = intro a h (fun y hr => recC intro (h y hr)) :=",
            'thmProof': 'rfl'
        },
        {
            'thmName': 'rec_eq_recC',
            'thmStatement': "theorem recC_intro {motive : (a : α) → Acc r a → Sort v}\n    (intro : (x : α) → (h : ∀ (y : α), r y x → Acc r y) →\n     ((y : α) → (hr : r y x) → motive y (h y hr)) → motive x (intro x h))\n    {a : α} (h : ∀ (y : α), r y a → Acc r y) :\n    recC intro (Acc.intro _ h) = intro a h (fun y hr => recC intro (h y hr)) :=",
            'thmProof': 'rfl'
        }
    ]
    write_to_json("test.json", dataset)
    readed_data = read_from_json("test.json")
    print(readed_data)

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 将每行转换为字典并添加到列表中
            data.append(json.loads(line.strip()))
    return data


def load_data_ls(data_path, 
                 has_proof=False, 
                 has_comment=False, 
                 llm_type="Meta-Llama3-8B", 
                 natural_language_statement_key="Informal_statement", 
                 natural_language_proof_key="Informal_proof"):
    data = read_from_json(data_path)
    to_return = []
    for i in range(len(data)):
        curr_record = data[i]
        if has_proof:
            if "sorry" in data[i]["Proof"]:
                continue
            curr_record["FL"] = curr_record["Proof"]    
            if has_comment:
                curr_record["FL"] = curr_record["Commented_proof"]
        if "deepseek" in llm_type.lower() and 'prover' in llm_type.lower():
            curr_record["NL"] = curr_record[natural_language_statement_key]
        else:
            curr_record["NL"] = f"{curr_record[natural_language_statement_key]}\n\n{curr_record[natural_language_proof_key]}"
        
        if "DSTrain" in llm_type:
            curr_record["FL_statement"] = curr_record["Statement"] + " by"
        else:
            curr_record["FL_statement"] = curr_record["Statement"]
        to_return += [curr_record]
    return to_return





def load_data_ls_pureLean(data_path, has_proof=False):
    data = read_from_json(data_path)
    to_return = []
    for i in range(len(data)):
        curr_record = data[i]
        if has_proof:
            if "sorry" in data[i]["Proof"]:
                continue
            curr_record["FL"] = curr_record["Proof"]    
        curr_record["NL"] = ""
        curr_record["FL_statement"] = curr_record["Statement"]
        to_return += [curr_record]
    return to_return



# This function extracts the lean4 code that have the given theorem_name
def extract_theorem_proof(input_str, theorem_name):
    # Regular expression to match theorems and proofs
    pattern = re.compile(
        r'(?P<theorem>theorem\s+' + re.escape(theorem_name) + r'\s*.*?[^:])\s*:=\s*(?P<proof>by\s+.*?)(?=\n\n|\Z)', 
        re.DOTALL
    )
    
    match = pattern.search(str(input_str))
    if match:
        return match.group('theorem') + ' := ' + match.group('proof')
    else:
        return None
    
# This function extracts the theorem name from the FL_statement
def find_theorem_name(input_str):
    # Regular expression to match the theorem name
    pattern = re.compile(r'theorem\s+(\w+)')
    
    match = pattern.search(input_str)
    if match:
        return match.group(1)
    else:
        return None

if __name__ == "__main__":
    _test()
