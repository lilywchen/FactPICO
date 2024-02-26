from openai import OpenAI
import time
from tqdm import tqdm
import os
import pandas as pd
import together


from dotenv import load_dotenv

result_prompt = """
You are given a result inference span from an abstract, and you are given a summary. A result inference span corresponds to an inferred result in an experiment. Find the corresponding result inference in the summary and use it to rate the summary between 1 to 4.

The ratings are as follows:

1 - The result inference is mentioned and described accurately.
2 - The result inference is mentioned but is described vaguely or is slightly inaccurate.
3 - The result inference is critically inaccurate.
4 - The result inference is missing in the model summary.

Please provide only the rating and the rationale for the rating. Provide the rating after stating "Rating:".

"""

def load_api_keys():
    load_dotenv(override=True)
    together.api_key = os.getenv("TOGETHER_API_KEY")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client


def result_gen_gpt(row, client):

    sys_prompt = result_prompt
        
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {'role':'system', "content": sys_prompt},
            {'role': 'user', 'content': "Result Inference Span:\n{}\n\nSummary:\n{}\n".format(row["results_span"], row["generation"])}
        ]
    )

    response_content = response.choices[0].message.content
    return response_content


def result_gen_together(row, model, debug=False):
    
    sys_prompt = result_prompt
    
    
    user_msg = "Result Inference Span:\n{}\n\nSummary:\n{}\n".format(row["results_span"], process_gen(row['generation']))
    
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

    time.sleep(1)
    return output['output']['choices'][0]['text']

filename = ""

df = pd.read_csv(filename)

models = [
    "gpt-4",
    "togethercomputer/llama-2-7b-chat",
    "togethercomputer/alpaca-7b",
    "mistralai/Mistral-7B-Instruct-v0.1",
]

gpt_client = load_api_keys()

for model in models:
    
    m_type = 'gpt-4' if model == 'gpt-4' else model.split("/")[1]
    ele = "result"
    col_name = f'{m_type}_{ele}'
    
    if col_name not in df.columns:
        df[col_name] = 'did not generate'

    for i, row in tqdm(df.iterrows(), total=df.shape[0], leave=False):
        processing = True
        
        if df.at[i, col_name] != 'did not generate':
            continue
        
        while processing:
            try: 
                if model == 'gpt-4':
                    df.at[i, col_name] = result_gen_gpt(row, gpt_client)
                else:
                    m_type = model.split("/")[1]
                    df.at[i, col_name] = result_gen_together(row, model)
                processing = False
                df.to_csv(filename)
            except Exception as error:
                print(error)
                time.sleep(5) 