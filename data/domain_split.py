# food,  hotel, nightlife, shopping  mall, sightseeing
import os
import json

data_path = './'
dial_path = os.path.join(data_path,'final_dials_bs_accumulated')

with open(os.path.join(data_path,'DATA_SPLIT_Feb_5.json'), "r", encoding="utf-8") as f:
    data_split = json.load(f) 

venue_cat = []
dial_list = data_split['test'] 
print('len_dial: ', len(dial_list))
cat_dict = {'food': ['restaurant', 'food', 'bbq','caf', 'buffet', 'bakery', 
                    'tea','snack','coffee', 'dessert','wings','soup'],
            'hotel': ['hotel','hostel'],
            'nightlife': ['nightlife','bar','movie'],
            'shopping mall': ['shopping mall','shopping', 'mall','plaza'],
            'sightseeing':['sightseeing','beach','lake','harbor','building',
                            'lookout','museum','golf','art', 'park','resort',
                            'and historic site', 'outdoor', 'garden']
            }

with open(os.path.join(data_path,'dial2user.json'), "r", encoding="utf-8") as f:
    dial2user = json.load(f) 
uid_all = []
for json_file_name in dial_list:
    dial_idx = json_file_name
    uid = dial2user[dial_idx]
    uid_all.append(uid)
    with open(os.path.join(dial_path, json_file_name + '.json'),"r", encoding="utf-8") as f:
        data = json.load(f)

    cat = data['goal']['content'][0]['venueCat']

    if isinstance(cat, list):
        cat = [i.lower() for i in cat]
    else:
        cat = cat.lower()

    cat_new = []
    for cls, values in cat_dict.items():
        #print('cat:', cat)
        for val in values:
            if val in cat or val in cat[0]:
                cat = cls
                cat_new.append(cat)
                #break
    cat_new = list(set(cat_new)) 
    if len(cat_new)>0:           
        venue_cat.extend(cat_new)
    else:
        print(cat)

    '''
    if len(cat_new)>1:
        print(cat_new)
        cat_i_all = []
        for cat_i in cat:
            cat_i = cat_i.lower()
            for cls, values in cat_dict.items():
                for val in values:
                    if val in cat_i:
                        cat_i = cls
            cat_i_all.append(cat_i)
        cat_i_all = list(set(cat_i_all))
        venue_cat.append(cat_i_all)
    else:
        venue_cat.extend(cat_new)
    '''
#print(venue_cat)

uid_domain = dict(zip(uid_all, venue_cat))
with open('uid_domain.json', 'w', encoding="utf-8") as f:
    f.write(json.dumps(uid_domain, indent=4))

venue_cat_list = venue_cat
print(len(venue_cat_list))

from collections import Counter
print(Counter(venue_cat_list))
#Counter({
# 'food': 567, 
# 'sightseeing': 149, 
# 'hotel': 111, 
# 'nightlife': 102, 
# 'shopping mall': 71})
venue_cat = list(set(venue_cat))
print(len(venue_cat))
print(venue_cat)