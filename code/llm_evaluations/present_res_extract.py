from openai import OpenAI
from tqdm import tqdm
import os
import pandas as pd
from dotenv import load_dotenv

result_ex_prompt = """
You are given a result inference span from both an abstract and a summary. A result span corresponds to an inferred result in an experiment. Use the result inference spans from the abstract and the summary to rate the summary between 1 to 4.

The ratings are as follows:

1 - The result inference is mentioned and described accurately.
2 - The result inference is mentioned but is described vaguely or is slightly inaccurate.
3 - The result inference is critically inaccurate.
4 - The result inference is missing in the model summary.

Please provide only the rating and the rationale for the rating. Provide the rating after stating "Rating:".

"""

extract_res_prompt = """
A result inference span corresponds to an inferred result in an experiment.

Identify result inference spans in the following passage. Pull direct quotes from the passage:
"""

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def res_extract(abs):
    
    response = client.chat.completions.create(
        model='gpt-4',
        messages=[
            {'role':'system', "content":extract_res_prompt},
            {'role':'user', 'content':abs}
        ]
    )

    
    response_content = response.choices[0].message.content
    return response_content

def res_gen_gpt(abs_res, sim_res):
    sys_prompt = result_ex_prompt
    response = client.chat.completions.create(
        model='gpt-4',
        messages=[
            {'role':'system', "content": sys_prompt},
            {'role': 'user', 'content': "Abstract:\n{}\n\nSummary:\n{}\n".format(abs_res, sim_res)}
        ]
    )
    response_content = response.choices[0].message.content
    return response_content

filename = ""

df = pd.read_csv(filename)
 
 
for i, row in tqdm(df.iterrows(), total=df.shape[0], leave=False):
    
    if df.at[i, 'gpt4_result_eval'] != "did not generate":
        continue
    abs_res = row['results_span']
    sim_res = res_extract(row['generation'])
    
    df.at[i, "gpt4_result_extract"] = sim_res
    df.at[i, "gpt4_result_eval"] = res_gen_gpt(abs_res, sim_res)

df.to_csv("res_combined_extract.csv")    
    
    
