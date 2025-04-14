from openai import OpenAI
import json
from tqdm import tqdm
import sys
from glob import glob
# Replace with your actual API key
import os
key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=key)

prefix= '''
Evaluate if the answer provided below reaches the correct result. Only and only respond in a single word - Yes or No
'''
prefix = '''
You are an AI evaluator focused on determining if mathematical solutions reach the correct numerical result.
<EVALUATION_CRITERIA>
- If the answer is numerical, check if the final numerical answer matches the correct answer within the error of 0.001
- If the answer is alphabet, check if final answer matches the correct answer
- The working steps and explanation don't matter as long as the final result is correct
</EVALUATION_CRITERIA>
<SCORING_SYSTEM>
- "Yes" - The answer contains the correct result.
- "No" - The answer does not contain the correct result
</SCORING_SYSTEM>
<EXAMPLES>
Example 1:
Correct result: 576
Answer: "Step 1: 36 × 2/3 = 24. Step 2: 24 × 24 = 576. Therefore, 576 tiles are needed to cover 32sq ft area"
Evaluation: Yes
Example 2:
Correct result: 576
Submitted answer: "Two thirds of 36 is 24, and 24 times 24 is 578. Mr. Boarden needs 578 tiles"
Evaluation: No
Example 3:
Correct result: E
Submitted answer: "Two thirds of 36 is 24, and 24 times 24 is 578. Mr. Boarden needs (E) 578 tiles"
Evaluation: Yes
Example 4:
Correct result: D
Submitted answer: "Two thirds of 36 is 24, and 24 times 24 is 578. Mr. Boarden needs (E) 578 tiles"
Evaluation: No
</EXAMPLES>
<IMPORTANT>
Format your response as a single word only: "Yes" or "No"
Your evaluation must be precise and binary - either the answer contains the correct result or it does not.
</IMPORTANT>
Evaluate the following

'''
files = []

for g in sys.argv[1:]:
    files = files + glob(g)

files = list(set(files))
print(files)
responses = {}

for f in files:
    responses[f] = []


for f in files:
    with open(f, "r") as fhandle:
        data = fhandle.readlines()
    for l in data:
        json_data = json.loads(l)
        ANSWER = json_data['answer']
        RAW_OUT = json_data['raw_output']
        json_data['request'] =  prefix + f'Correct result: {ANSWER} \n Answer: {RAW_OUT}'
        responses[f].append(json_data)

total_completion_tokens = 0
total_prompt_tokens = 0
total_cost = 0
gpt_4o_mini = {
        'name' : "gpt-4o-mini",
        'input' : 0.15,
        'completion' : 0.60,
        }


o3_mini = {
        'name' : "o3-mini",
        'input' : 1.1,
        'completion' : 4.4,
        }
model = gpt_4o_mini
model = o3_mini
prev_cost = 0

def get_final_file(f):
    return f.replace('.jsonl', '-judged-'+model['name']+'.jsonl')
for f in files:
    print("FILE:", f)
    final_file = get_final_file(f)
    if os.path.exists(final_file):
        print("Already exists", final_file)
        continue
    for sample in tqdm(responses[f]):
        matched=False
        #try:
        #    if abs(float(sample['answer']) - float(sample['prediction']))  < 0.001:
        #        sample['judge'] = 'Yes'
        #        sample['llm_judge'] = 'No'
        #        matched=True
        #except:
        #    pass
        #if not matched:
        #    if str(sample['answer']).strip() == str(sample['prediction']).strip():
        #        sample['judge'] = 'Yes'
        #        sample['llm_judge'] = 'No'
        #        matched=True
        
        if not matched:
            sample['llm_judge'] = 'Yes'
            response = client.chat.completions.create(model=model['name'],  # or "gpt-3.5-turbo"
                            messages=[
                                    {"role": "user", "content": sample['request']}
                            ])

            result = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_completion_tokens +=completion_tokens
            total_prompt_tokens += prompt_tokens
            sample['judge'] = result
            sample['prompt_tokens'] = prompt_tokens
            sample['completion_tokens'] = completion_tokens
            sample['full_judge_response'] = str(response)
            total_cost = ( model['input'] * total_prompt_tokens + model['completion'] * total_completion_tokens ) / 1000000
            if (total_cost -  prev_cost)  > 0.1:
                print("Total Cost", total_cost)
                prev_cost = total_cost

    print("Total usage", total_prompt_tokens, total_completion_tokens)
    print("Total cost", total_cost)
    data = responses[f]
    lines = [json.dumps(d)+'\n' for d in data]
    with open(final_file, 'w') as fhandle:
        fhandle.writelines(lines)

print("Total usage", total_prompt_tokens, total_completion_tokens)
print("Total cost", total_cost)

