from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import random
from tqdm.auto import tqdm
import utils


class Prove_writer:

    # The model and tokenizer should be a HuggingFace model, ModelForCausalLM is used in all our testing cases
    # example_ls should be a list of dict that has form:
    #      [{"NL": <Natural Language version of theorem statement and prove>,
    #        "Name": <Name of current theorem>,
    #        "FL": <Formal Language version of theorem statement and prove>
    #        "FL_Statement": <Formal Language version of theorem statement>},
    #       ...
    #      ]
    def __init__(self,
                 model,
                 tokenizer,
                 terminators,
                 example_ls,
                 example_num=3):
        self.model = model
        self.tokenizer = tokenizer
        self.terminators = terminators
        self.example_ls = example_ls
        self.example_num = example_num

    # This function extract the first lean code block from the text provided for normal LLMs, if using DeepSeek-Prover-1.5, we should
    # extract the last code block
    def _extract_lean_code_blocks(self, text, llm_type="Llama3-Instruct"):
        lines = text.split('\n')
        inside_code_block = False
        code_blocks = []
        current_block = []

        for line in lines:
            if line.strip().startswith('```lean'):
                inside_code_block = True
                continue  
            elif "```" in line.strip() and inside_code_block:
                code_blocks.append('\n'.join(current_block))
                current_block = []
                inside_code_block = False
                continue  

            if inside_code_block:
                current_block.append(line)

        try:
            if "deepseek" in llm_type.lower() and 'prover' in llm_type.lower():
                if current_block != []:
                    code_blocks.append('\n'.join(current_block))
                return code_blocks[-1]
            else:
                return code_blocks[0]
        except Exception:
            print(f"current result has unexcepted problem, generated text is:{text}")
            return -1

    # This function test whether there is a lean code block inside the text
    def _contains_lean_code_block(self, text):
        pattern = r'```lean.*?```'
        match = re.search(pattern, text, re.DOTALL)
        return match is not None

    # This function query the model with given prompt and return the result string, we assume all format of instructions
    # are already inside the prompt
    # If we are using deepseek-prover-1.5, return the whole response without any processing
    def _query_model(self,
                     prompt,
                     max_new_tokens=1024,
                     do_sample=True,
                     temperature=0.9,
                     top_p=0.9,
                     repetition_penalty=1.0,
                     batch_size=4, 
                     llm_type="Llama3-Instruct"):
        inputs = self.tokenizer([prompt] * batch_size, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs,
                                      max_new_tokens=max_new_tokens,
                                      eos_token_id=self.terminators,
                                      do_sample=do_sample,
                                      temperature=temperature,
                                      top_p=top_p, 
                                      repetition_penalty=repetition_penalty,)
        if "deepseek" in llm_type.lower() and 'prover' in llm_type.lower():
            responses = [self.tokenizer.decode(outputs[i], skip_special_tokens=True) for i in range(len(outputs))]
        else:
            responses = [self.tokenizer.decode(outputs[i])[len(prompt):] for i in range(len(outputs))]
        return responses

    # This function formulate prompt in the form of Llama3-instruct style
    def _formulate_prompt_llama3Instruct(self,
                                         NL,
                                         thm_name,
                                         FL_statement,
                                         system_prompt="You are a Lean4 expert who can write good Lean4 code based on natural language mathematical theorem and proof"):

        if FL_statement[-len("sorry"):] == "sorry":
            FL_statement = FL_statement[:-len("sorry")]

        selected_examples = random.sample(self.example_ls, self.example_num)
        prompt = f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id>user<|end_header_id|>\n\n"
        for curr_example in selected_examples:
            prompt += (f"Natural language version of theorem and proof:\n"
                       f"{curr_example['Name']}\n{curr_example['NL']}\n\n"
                       f"### Lean4 version of theorem statement:\n"
                       f"```lean\n{curr_example['FL_statement']}\n```\n\n"
                       f"### Lean4 version of theorem and proof:\n"
                       f"```lean\n{curr_example['FL']}\n```<|reserved_special_token_26|>\n\n"
                       f"---\n\n")
        prompt += (f"Natural language version of theorem and proof:\n"
                   f"{thm_name}\n{NL}\n\n"
                   f"### Lean4 version of theorem statement:\n"
                   f"```lean\n{FL_statement}\n```\n\n"
                   f"### Lean4 version of theorem and proof:\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>")
        return prompt

    # generate_proof_singleThm() function ask the LLM in this class to write some proves for the provided thm_record
    # this is most of the model-specific setting take place
    # The thm_record should be a dictionary in form:
    # {"NL": <Natural Language version of theorem statement and prove>,
    #  "Name": <Name of current theorem>,
    #  "FL_Statement": <Formal Language version of theorem statement>, ...}
    # The return will be a list of complete lean4 theorem proofs in the format
    # ['theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n  -- aesop?\n  ring',
    #  'theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n  rfl', ...]
    def generate_proof_singleThm(self,
                                 thm_record,
                                 proof_num=200,
                                 do_sample=True,
                                 temperature=0.9,
                                 top_p=0.9,
                                 variable_top_p = (-1.0),
                                 variable_tempreature=(-1.0),
                                 batch_size=4,
                                 repetition_penalty=1.0,
                                 print_result=True,
                                 llm_type="Llama3-Instruct", 
                                 system_prompt="You are a Lean4 expert who can write good Lean4 code based on natural language mathematical theorem and proof"):
        proof_ls = []
        
        for i in range(int(proof_num / batch_size)):
            if llm_type == "Llama3-Instruct":
                prompt = self._formulate_prompt_llama3Instruct(thm_record["NL"],
                                                            thm_record["Name"],
                                                            thm_record["FL_statement"], 
                                                            system_prompt=system_prompt)

            else:
                raise NotImplementedError

            if print_result:
                print("current prompt is")
                print(prompt)
            
            if variable_tempreature != -1.0:
                curr_temperature = random.uniform(variable_tempreature, temperature)
            else:
                curr_temperature = temperature
            
            if variable_top_p != -1.0:
                curr_top_p = random.uniform(variable_top_p, top_p)
            else:
                curr_top_p = top_p
            
            curr_responses = self._query_model(prompt,
                                               do_sample=do_sample,
                                               temperature=curr_temperature,
                                               top_p=curr_top_p,
                                               batch_size=batch_size, 
                                               repetition_penalty=repetition_penalty, 
                                               llm_type=llm_type)
            if print_result:
                for curr_response in curr_responses:
                    print(curr_response)
            for curr_response in curr_responses:
                if self._contains_lean_code_block(curr_response):
                    curr_proof = self._extract_lean_code_blocks(curr_response, llm_type=llm_type)
                    proof_ls += [curr_proof]
        return proof_ls

    # generate_proof_dataset() function ask the LLM in this class to write some proves for the theorems in provided set_to_prove
    # The set_to_prove should be a dictionary in form:
    # [{"NL": <Natural Language version of theorem statement and prove>,
    #   "Name": <Name of current theorem>,
    #   "FL_Statement": <Formal Language version of theorem statement>, ...
    #  }, ...
    # ]
    # The return will be a list of complete lean4 theorem proofs in the format
    # ['theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n  -- aesop?\n  ring',
    #  'theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n  rfl', ...]
    # it will be in the "Generated_proof" section of each thm record
    # The parameter of variable tempreature is used to set the range of random temperature, if it is set to -1.0, the temperature 
    # is a fixed number, if not, the temperature parameter indicates the upper bound of temperature and variable_temperature is the lower bound
    def generate_proof_dataset(self,
                               set_to_prove,
                               proof_num=200,
                               do_sample=True,
                               temperature=0.9,
                               top_p=0.9,
                               variable_top_p = (-1.0),
                               variable_tempreature=(-1.0),
                               batch_size=4,
                               repetition_penalty=1.0,
                               ckpt_path="./Generated_proof_ckpt", 
                               llm_type="Llama3-Instruct", 
                               system_prompt="You are a Lean4 expert who can write good Lean4 code based on natural language mathematical theorem and proof"):
        for i in tqdm(range(len(set_to_prove))):
            curr_thm_record = set_to_prove[i]
            curr_prove_list = self.generate_proof_singleThm(curr_thm_record,
                                                            proof_num=proof_num,
                                                            do_sample=do_sample,
                                                            temperature=temperature,
                                                            top_p=top_p,
                                                            variable_top_p=variable_top_p,
                                                            variable_tempreature=variable_tempreature,
                                                            batch_size=batch_size, 
                                                            llm_type=llm_type, 
                                                            system_prompt=system_prompt, 
                                                            repetition_penalty=repetition_penalty,)
            curr_thm_record["Generated_proof"] = curr_prove_list
            set_to_prove[i] = curr_thm_record

            if ckpt_path != None:
                utils.write_to_json(f"{ckpt_path}/Generated_proof_{curr_thm_record['Name']}.json", curr_thm_record)

        return set_to_prove
