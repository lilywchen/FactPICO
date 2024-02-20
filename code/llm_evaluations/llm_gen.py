
from openai import OpenAI
import time
from tqdm import tqdm
import os
import pandas as pd
import together


from dotenv import load_dotenv


system_prompt1 = "You are given an abstract and a summary. <PICOInfo> Find the <PICOElem> in accordance with PICO in both the abstract and the summary and use it rate the summary between 1 to 5.\n\n\nThe ratings are as follows.\n\n1 - The <PICOElem> is mentioned in the model summary and described accurately.\n2 - The <PICOElem> is mentioned in the model summary but described vaguely or somewhat inaccurately.\n3 - The <PICOElem> is mentioned in the model summary but described inaccurately or is missing critical descriptors.\n4 - The <PICOElem> is missing in the model summary.\n5 - N/A\n\nPlease provide only the rating and the rationale for the rating. Provide the rating after stating \"Rating:\"."

pop_info = "Population in PICO describes the type of subjects involved in the trial. Critical descriptors for population include important demographic information and any specific shared conditions."
inter_info = "Intervention in PICO describes the treatments considered in the trial."
comp_info = "Comparator in PICO describes the alternative treatment to which the intervention is being compared against."
out_info = "Outcome in PICO describes the outcome measures used to determine results of the trial. If the primary outcome measures are not be mentioned, then the summary is critically flawed."

info_label = {
    'population': pop_info,
    'interventions': inter_info,
    'comparator': comp_info,
    'outcome': out_info,
}

def load_api_keys():
    load_dotenv(override=True)
    together.api_key = os.getenv("TOGETHER_API_KEY")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client


def pico_gen_gpt(row, pico_elem, client):
    
    sys_prompt = system_prompt1.replace("<PICOElem>", pico_elem)
    
    if pico_elem == 'interventions':
        sys_prompt = system_prompt1.replace("<PICOElem> is", "interventions are")
        sys_prompt = sys_prompt.replace("<PICOElem>", pico_elem)

    sys_prompt = sys_prompt.replace("<PICOInfo>", info_label[pico_elem])
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {'role':'system', "content": sys_prompt},
            {'role': 'user', 'content': "Abstract:\n{}\n\nSummary:\n{}\n".format(row["Abstract"], row["generation"])}
        ]
    )
    response_content = response.choices[0].message.content
    return response_content


def pico_gen_together(row, pico_elem, model, debug=False):
    
    sys_prompt = system_prompt1.replace("<PICOElem>", pico_elem)
    
    if pico_elem == 'interventions':
        sys_prompt = system_prompt1.replace("<PICOElem> is", "interventions are")
        sys_prompt = sys_prompt.replace("<PICOElem>", pico_elem)
    
    sys_prompt = sys_prompt.replace("<PICOInfo>", info_label[pico_elem])        
    
    
    user_msg = "Abstract:\n{}\n\nSummary:\n{}\n".format(row["Abstract"], process_gen(row['generation']))
    
    prompt = ""
       
    if 'llama' in model:
        prompt = f"<s>[INST] <<SYS>>{sys_prompt}<</SYS>>\\n\\n{user_msg}[/INST]"
    elif 'mistral' in model:
        prompt = f"<s>[INST] {sys_prompt}\n\n{user_msg}[/INST]"
    elif 'alpaca' in model:
        prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{sys_prompt}\n\n### Input:\n{user_msg}\n\n### Response:"

    
    output = together.Complete.create(
        prompt,
        model = model,
        max_tokens = 256,
        temperature = 0.6,
        top_k = 90,
        top_p = 0.8,
        repetition_penalty = 1.1,
        stop = ['</s>']
    )
    
    time.sleep(5)
    return output['output']['choices'][0]['text']
    

models = [
    "gpt-4",
    "togethercomputer/llama-2-7b-chat",
    "togethercomputer/alpaca-7b",
    "mistralai/Mistral-7B-Instruct-v0.1",
]

#Replace with processing file
filename = ""

df = pd.read_csv(filename)

gpt_client = load_api_keys()


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

for model in tqdm(models):
    print(model)
    for ele in tqdm(info_label.keys(), leave=False):
        print(ele)
        if model == 'gpt-4':
            if f'gpt-4_{ele}' not in df.columns:
                df[f'gpt-4_{ele}'] = "did not generate"
        else:            
            m_type = model.split("/")[1]
            if f'{m_type}_{ele}' not in df.columns:
                df[f'{m_type}_{ele}'] = 'did not generate'
        inc_time = 3
        for i, row in tqdm(df.iterrows(), total=df.shape[0], leave=False):

            m_type = 'gpt-4' if model == 'gpt-4' else model.split("/")[1]
            
            if not pd.isna(df.at[i, f'{m_type}_{ele}']):
                text = df.at[i, f'{m_type}_{ele}']
                if is_float(text) or "Rating: " in text:
                    continue
            
            processing = True
            while processing:
                try: 
                    if model == 'gpt-4':
                        print("skip")
                    else:
                        m_type = model.split("/")[1]
                        df.at[i, f'{m_type}_{ele}'] = pico_gen_together(row, ele, model)
                    processing = False
                except Exception as error:
                    time.sleep(5)                            
            df.to_csv(filename)
