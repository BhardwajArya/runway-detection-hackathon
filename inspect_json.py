import json, sys
p = "/Users/anandjha/Downloads/archive/labels/labels/lines/train_labels_640x360.json"
data = json.load(open(p,'r'))
print("TYPE:", type(data))
if isinstance(data, dict):
    print("TOP-LEVEL KEYS:", list(data.keys())[:50])
else:
    print("Top-level is list with length", len(data))
# show a small sample to inspect shapes
def show_sample(x, depth=0, limit=3):
    t = type(x)
    print("  " * depth + f"- type={t}, repr_snip={repr(x)[:200]}")
    if isinstance(x, dict):
        for k in list(x.keys())[:limit]:
            print("  " * (depth+1) + f"KEY: {k} -> type={type(x[k])}")
    elif isinstance(x, list):
        for i, item in enumerate(x[:limit]):
            print("  " * (depth+1) + f"ITEM[{i}] type={type(item)}")
            if isinstance(item, (dict, list)):
                show_sample(item, depth+2, limit=2)
# print first 5 top-level entries (or keys->values) for manual inspection
if isinstance(data, dict):
    for i,k in enumerate(list(data.keys())[:10]):
        print(f"\nENTRY KEY {i} = {k}")
        show_sample(data[k], depth=1, limit=2)
else:
    for i,item in enumerate(data[:10]):
        print(f"\nENTRY INDEX {i}")
        show_sample(item, depth=1, limit=2)
print("\\nDone inspection. Paste the printed output here so I can write the exact converter.")