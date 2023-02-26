
import pandas as pd
import json
import pickle
from collections import OrderedDict
import nltk
import nltk.stem as ns
import numpy as np
from tqdm import tqdm
import re

slot_value2item = {}
with open('item_dict.json', 'r') as f:
    item_dict = json.load(f)

def save_json(data, save_name):
    with open(save_name+'.json', 'w', encoding="utf-8") as f:
        f.write(json.dumps(data, indent=4))
    
    
for item, slot_value_list in item_dict.items():
    slot_value = list(slot_value_list.values())
    slot_value = [j  for i in slot_value[2:] for j in i]
    for k in slot_value:
        if k not in slot_value2item.keys():
            slot_value2item[k] = []
        slot_value2item[k].append(int(item))

save_json(slot_value2item, 'slot_value2item')                               
 