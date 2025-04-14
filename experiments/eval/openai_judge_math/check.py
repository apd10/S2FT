import json
from tqdm import tqdm
import sys
from glob import glob

files = []

for g in sys.argv[1:]:
    files = files + glob(g)

files = list(set(files))
print(files)
responses = {}

for f in files:
    with open(f, "r") as fhandle:
        data = fhandle.readlines()

    legacy_correct=0
    judge_correct=0
    total = 0
    for l in data:
        json_data = json.loads(l)
        ANSWER = json_data['answer']
        PREDICTION = json_data['prediction']
        judge = json_data['judge']
        matched=False
        try:
            if abs(float(ANSWER) - float(PREDICTION))  < 0.001:
                matched=True
        except:
            pass
        if not matched:
            if str(ANSWER).strip() == str(PREDICTION).strip():
                matched=True
        if matched:
            legacy_correct += 1
            if judge == "No":
                print("*********************")
                print(json_data['instruction'])
                print("---------------------")
                print("Judge-NO Legacy-Yes", json_data['answer'], json_data['prediction'], json_data['raw_output'])
        if judge == "Yes":
            judge_correct += 1

            if not matched:
                print("*********************")
                print(json_data['instruction'])
                print("---------------------")
                print("Judge-Yes Legacy=No", json_data['answer'], json_data['prediction'], json_data['raw_output'])

        total+=1
    print(f"{f}: legacy:{legacy_correct}/{total} ({(legacy_correct/total):.2f}) judge:{judge_correct}/{total} ({(judge_correct/total):.2f}) ", )
