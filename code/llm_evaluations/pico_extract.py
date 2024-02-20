from openai import OpenAI
import time
from tqdm import tqdm
import os
import pandas as pd


from tqdm import tqdm

from dotenv import load_dotenv


gpt_pico_extract_prompt = """
Definition of each PICO element:

Population: The types of patients involved in the trial
Intervention: The treatments considered
Comparator: The alternative treatment to which the intervention is being compared to.
Outcome: The thing measured. NOT what was found in the study (“result”). For example, if the study finds a drug reduces the duration of headache, the outcome here is just the “duration of headache”, not that it reduced it.  

Identify the PICO elements in the following passage. Pull direct quotes from the passage:

"""

pop_info = "Population in PICO describes the type of subjects involved in the trial. Critical descriptors for population include important demographic information and any specific shared conditions."
inter_info = "Intervention in PICO describes the treatments considered in the trial."
comp_info = "Comparator in PICO describes the alternative treatment to which the intervention is being compared against."
out_info = "Outcome in PICO describes the outcome measures used to determine results of the trial. If the primary outcome measures are not be mentioned, then the summary is critically flawed."


gpt_system_prompt = """
You are given a list of PICO elements from an abstract and a summary. <PICOInfo> Find the <PICOElem> in accordance with PICO in both the abstract and the summary and use it rate the summary between 1 to 5.


The ratings are as follows.

1 - The <PICOElem> is mentioned in the model summary and described accurately.
2 - The <PICOElem> is mentioned in the model summary but described vaguely or somewhat inaccurately.
3 - The <PICOElem> is mentioned in the model summary but described inaccurately or is missing critical descriptors.
4 - The <PICOElem> is missing in the model summary.
5 - N/A

Please provide only the rating and the rationale for the rating. Provide the rating after stating "Rating:".
"""


load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

info_label = {
    'population': pop_info,
    'interventions': inter_info,
    'comparator': comp_info,
    'outcome': out_info,
}

def pico_gen_gpt(abs_pico, sim_pico, pico_elem):
    
    sys_prompt = gpt_system_prompt.replace("<PICOElem>", pico_elem)
    
    if pico_elem == 'interventions':
        sys_prompt = gpt_system_prompt.replace("<PICOElem> is", "interventions are")
        sys_prompt = sys_prompt.replace("<PICOElem>", pico_elem)

    sys_prompt = sys_prompt.replace("<PICOInfo>", info_label[pico_elem])
    
    response = client.chat.completions.create(
        model='gpt-4',
        messages=[
            {'role':'system', "content": sys_prompt},
            {'role': 'user', 'content': "Abstract:\n{}\n\nSummary:\n{}\n".format(abs_pico, sim_pico)}
        ]
    )
    time.sleep(1)
    response_content = response.choices[0].message.content
    return response_content

def pico_extract(abs):
    
    response = client.chat.completions.create(
        model='gpt-4',
        messages=[
            {'role':'system', "content":gpt_pico_extract_prompt},
            {'role':'user', 'content':abs}
        ]
    )
    
    time.sleep(1)
    
    response_content = response.choices[0].message.content
    return response_content    
    
filename = ""

df = pd.read_csv(filename)

data = []
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    print(i, len(data))
    if i < len(data):
        continue
    data_dict = {}
    data_dict['abs'] = row['Abstract']
    data_dict['gen'] = row['generation']
    data_dict['pico_abs'] = pico_extract(row['Abstract'])
    data_dict['pico_sim'] = pico_extract(row['generation'])
    print("got pico")
    data_dict['Rating_pop'] = pico_gen_gpt(data_dict['pico_abs'], data_dict['pico_sim'], 'population')
    data_dict['Rating_inter'] = pico_gen_gpt(data_dict['pico_abs'], data_dict['pico_sim'], 'interventions')
    data_dict['Rating_comp'] = pico_gen_gpt(data_dict['pico_abs'], data_dict['pico_sim'], 'comparator')
    data_dict['Rating_out'] = pico_gen_gpt(data_dict['pico_abs'], data_dict['pico_sim'], 'outcome')
    print("got rest")
    data.append(data_dict)
    
new_df = pd.DataFrame.from_dict(data)

new_filename = ""

new_df.to_csv(new_filename)