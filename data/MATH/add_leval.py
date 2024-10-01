from datasets import load_dataset
import json


with open("/apdcephfs_qy3/share_301812049/brightxwang/timurhe-tmp/W2S/data/MATH/MATH_test.jsonl") as f:
    data = [json.loads(l) for l in f]

id2level = {}
for d in data:
    id2level[d["idx"]] = d["level"]

with open("/apdcephfs_qy3/share_301812049/brightxwang/timurhe-tmp/W2S/data/MATH/id2level.jsonl", "w") as f:
    json.dump(id2level, f, indent=4)
