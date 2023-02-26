
import pandas as pd
import json
import pickle
from collections import OrderedDict
import nltk
import nltk.stem as ns
import numpy as np
from tqdm import tqdm
import re
lemmatizer = ns.WordNetLemmatizer()

pattern = re.compile(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fa5\uac00-\ud7ff\u00e8-\u00e9]')
def value_re(s,v):
    #s=repr(s)
    #print(type(s))
    #s2 = re.sub(r'\\u....','',s)
    #s2 = re.sub(pattern,"",s)
    #s2 = s2.strip()

    #s2= re.findall('[a-zA-Z0-9]+',s, re.S)
    #s2 = ' '.join(s2)
    s = re.sub(u'\u00e9', "e", s) 
    s = re.sub(u'\u00f4', "o", s) 
    
    str_enc = s.encode('ascii','ignore')
    s2 = str_enc.decode()
   
    return s2

def save_json(data, save_name):
    with open(save_name+'.json', 'w', encoding="utf-8") as f:
        f.write(json.dumps(data, indent=4))

with open("./venues_db_new.txt","r", encoding="utf-8") as f:
    venue_db = json.load(f)

with open("./corrected_315_classes_Nov_30.txt","r", encoding="utf-8") as f:
    image_classes = json.load(f)

######################### image ################################
with open('./data_final_with_images.txt',"r", encoding="utf-8") as f:
    dial_with_image_list = json.load(f)

dial_turn_id_with_image = OrderedDict()
"""
dial_idx: {turn_idx: image_name}
"""
image_name_user_provide = set()
for dial in dial_with_image_list:
    dial_idx = dial['dialogue_idx']
    dial_turn_id_with_image[dial_idx] = {}
    dial_content = dial['dialogue'] # list
    for dial_i in dial_content:
        turn_idx = dial_i['turn_idx']
        dial_user = dial_i['user']
        if 'img_gts' in dial_user:
            if len(dial_user['img_gts'])>0:
                img_url = dial_user['imgs'] 
                img_name = img_url[0].split('/')[-1].split('.')[0]
                image_name_user_provide.add(img_name)
                dial_turn_id_with_image[dial_idx][turn_idx] = img_name

save_json(dial_turn_id_with_image, 'dial_turn_id_with_image')
save_json(list(image_name_user_provide), 'image_name_user_provide')
print('dial_turn_id_with_image done')
print('len images from user provide {}'.format(len(image_name_user_provide)))


image2id = OrderedDict()
'''
{
”xxxx.jpg”: {
    "id": 0,
    "class_name": " "
    "venueName": " "       
    }
},
'''
image_idx = -1
for idx, (className, images) in enumerate(image_classes.items()):
    for img in images:
        image_idx+=1
        split_list = img.split('/')
        img = split_list[-1][:-4]
        if img in image_name_user_provide:
            continue
        venueName = split_list[2]
        image2id[img] = {}
        image2id[img]['venueName'] = venueName
        image2id[img]['className'] = className
        image2id[img]['class_id'] = idx
        image2id[img]['image_id'] = image_idx
save_json(image2id, 'image_map')
print('the num of images in db not in user provide: {}'.format(len(image2id)))

'''
venue_db[0].keys()
dict_keys(['requestable', 'informable'])
venue_db[0]['requestable'].keys()
dict_keys(['venueName', 'venueAddress', 'desc', 'url', 'telephone', 'timeFrame', 'tips', 'images'])
venue_db[0]['informable'].keys()
dict_keys(['venueScore', 'Credit Cards', 'Menus', 'venueNeigh', 'terms', 'Parking',  \
    'Dining Options', 'Restroom', 'Wi-Fi', 'Music', 'Outdoor Seating', 'Wheelchair Accessible', \
         'Smoking', 'Reservations', 'Drinks', 'venueCat', 'price'])
'''

######################### slot ################################
# 构建graph时，venue也是节点，所以id map 要考虑venueName
slot2id = OrderedDict()
slots = ['venueName','venueAddress', 'telephone', 'images', 'venueScore', 'Credit Cards', 'Menus', 'venueNeigh', 'open span', 'Parking',  \
        'Dining Options', 'Restroom', 'Wi-Fi', 'Music', 'Outdoor Seating', 'Wheelchair Accessible', \
        'Smoking', 'Reservations', 'Drinks', 'price' ]

for idx, slot in enumerate(slots):
    slot = slot.lower()
    slot2id[slot] = idx
save_json(slot2id, 'slot_id_map')

######################### open span ################################
open_span_value_map = OrderedDict()
open_span_list = []
with open("./open_span_candidates_frequencies_filtered_1063.txt","r", encoding="utf-8") as f:
    for ind, line in enumerate(f.readlines()):
        #print(line)
        word = line.split('\t')[0]
        word = value_re(word, v=False)
        open_span_list.append(word)
        slot_value = 'open span'+'_'+word
        open_span_value_map[slot_value] = ind
len_open_sapn = len(open_span_value_map)
print('len open_span: ', len(open_span_list))

venuescore = ["poor", "good", "average", "excellent", "fair"]
for indx, score in enumerate(venuescore):
    slot_value = 'venuescore'+'_'+score
    open_span_value_map[slot_value] = int(len_open_sapn+indx)
len_venue_score = len(open_span_value_map) 

slot_value_id0 = len_open_sapn+len_venue_score
'''
venue_db_dict: 
{
”0”:  # venueName id
    {
        "0": [0,1,2,]  # slot_id: slot_value_id
        "1": [3,4,5]

},
'''
venue_db_dict = {}
slot_value2id = open_span_value_map
slot_value_index = slot_value_id0-1
print('slot_value_id0: ',slot_value_id0)
venuenames = set()
# mustafa centre
venueID = -1
for venue_data in venue_db:
    data_request = venue_data['requestable']
    data_inform = venue_data['informable']
    data = {**data_request,**data_inform}
        
    venueName = (data['venueName']).lower()

    '''
    venueName = re.findall('[a-zA-Z0-9]+',venueName, re.S)
    venueName = ' '.join(venueName)

    '''
    
    venueName = value_re(venueName, v=False)
    if len(venueName)==0:
        print(venueName)

    if venueName in venuenames:
        print(venueName)
        continue
    venueID+=1
    venue_db_dict[venueID] = {}
    venue_db_dict[venueID]["venuename"] = venueName
    venuenames.add(venueName)
    #venue_db_dict[i]["feature_index"] = []

    for slot, value in data.items():
        slot = slot.lower()
        #if slot in set(['venueName','desc', 'url', 'timeFrame','venueCat', 'tips']):
        if slot in set(['desc', 'url', 'timeframe','venuecat', 'tips']):
            continue
        if slot=='terms':
            slot='open span'      

        slot_id = slot2id[slot]
        if len(value)>0:  # 有的value是空。 
            venue_db_dict[venueID][slot_id] = []

            #print(type(value))
            # value 转化为小写
            try:
                # 字符串
                value = value.lower()
                value = value.strip()
                value = [value]  # 字符串放到list
                #print(len(value))
            except:
                # list中每个字符串都转化为小写
                #长度1: ['happy hour']
                #长度>1,有多个value: ['beer', ' full bar', ' cocktails'] ['sunsets', 'beach', 'casual']
                try:
                    value = [i.lower().strip() for i in value]
                except:
                    value = [i[0] for i in value]
                    #print(value)
            
            for value_i in value:

                if slot=='images': 
                    try:                   
                        #value_i = str(image2id[value_i]['image_id'])
                        value_i = str(image2id[value_i]['className'])
                    except:
                        continue
                        #print(value_i)
                        #print(venueName)                    
                else:
                    value_i = value_i.strip()
                    value_i = value_i.replace('-', ' ')
                    value_i = value_i.replace('\n', ' ')
                    #Yes (incl. American Express & MasterCard)
                    if 'yes' in value_i:
                        value_i = 'yes'
                    if slot in ['open span','menus' , 'drinks','wi-fi']:
                        # value转化为单数形式
                        #'Menus_bar snacks', 'Drinks_cocktails', 名词
                        # 'Wi-Fi_paid': 动词
                        new_word = []
                        for word_i in value_i.split(' '):
                            word_i2 = lemmatizer.lemmatize(word_i,'n')
                            if word_i2==word_i:
                                word_i2 = lemmatizer.lemmatize(word_i,'v')
                            new_word.append(word_i2)
                        value_i = ' '.join(new_word)
                    
                    if slot=='open span':    
                        if not value_i in set(open_span_list):
                                continue           
                
                value_i = value_re(value_i, v=False)
                slot_value = slot + '_' + value_i #.strip() 
                #if add .strip(), node count is 6049

                if slot_value not in slot_value2id.keys():
                    slot_value_index+=1
                    slot_value2id[slot_value] = slot_value_index
                                
                if not slot_value2id[slot_value] in set(venue_db_dict[venueID][slot_id]): 
                    venue_db_dict[venueID][slot_id].append(slot_value2id[slot_value])


print('number of slot_value in db',slot_value_index)

slot_value2id = {**open_span_value_map,**slot_value2id}

'''
save_json(image2id, 'image_map')
save_json(venue_db_dict, 'item_dict')
save_json(slot_value2id, 'slot_value_map')
'''
# reindex
'''
    "0": [0,1,2,]
    "1": [3,4,5]
'''
print('reindex: put together the values belonging to the same slot.....................')
slot_has_values_id = {}
for slot_id in range(len(slot2id)):
    slot_has_values_id[slot_id] = []

newid_dict = OrderedDict()
id_old = []
for slot_value, idx in slot_value2id.items():
    split_s = slot_value.split('_') 
    slot = split_s[0]  
    value = split_s[1]
    slot_id = slot2id[slot]
    slot_has_values_id[slot_id].append(idx)
    #id_old.append(idx)
#value_len = np.array([len(set(value)) for i, value in slot_has_values.items()])
#cumsum = np.cumsum(value_len)
id_old = [idx for slot_id in range(len(slot2id)) for idx in slot_has_values_id[slot_id]]
id_new = list(range(len(id_old)))
for old, new in zip(id_old, id_new):
    newid_dict[old] = new

new_slot_value2id = OrderedDict()
for slot_value, idx in slot_value2id.items():
    new_slot_value2id[slot_value] = newid_dict[idx]

new_slot_has_values_id = OrderedDict()
for slot_id, value_id in slot_has_values_id.items():
    new_value_id = [newid_dict[i] for i in value_id]
    new_slot_has_values_id[slot_id] = new_value_id

new_venue_db_dict = OrderedDict()
for venue_id, slot_value in venue_db_dict.items():
    new_venue_db_dict[venue_id] = {}
    for slot_id, value_id in slot_value.items():
        if slot_id=='venuename':
            new_venue_db_dict[venue_id][slot_id] = value_id
        else:
            new_value_id = [newid_dict[i] for i in value_id]
            new_venue_db_dict[venue_id][slot_id] = new_value_id

sort = sorted(new_slot_value2id.items(), key=lambda x: x[1])
new_slot_value2id = OrderedDict()
for item in sort:
    new_slot_value2id[item[0]] = item[1]    

save_json(new_venue_db_dict, 'item_dict')
save_json(new_slot_has_values_id, 'slot_values_id')
save_json(new_slot_value2id, 'slot_value_map')

print('graph constructing.....................')
# 定义包含互斥关系的slot
slot_with_neg = set(['venueScore', 'Credit Cards', 'Menus', 'venueNeigh', 'Parking', \
                     'Dining Options', 'Restroom', 'Wi-Fi', 'Music', 'Outdoor Seating', \
                     'Wheelchair Accessible', 'Smoking', 'Reservations', 'Drinks', 'price'])

slot_id_with_neg = set([slot2id[i.lower()] for i in slot_with_neg])
sign_data = pd.DataFrame(columns=('from','to','sign'))

for venue_id, slot_value in new_venue_db_dict.items():
    for slot_id, slot_value_ids in slot_value.items():
        if slot_id=='venuename':
            continue

        if slot_value_ids==[venue_id]:
            continue

        for slot_value_id in slot_value_ids:
            sign_data = sign_data.append(pd.DataFrame({'from':[venue_id],'to':slot_value_id,'sign':[1]}), ignore_index=True)
        if slot_id in slot_id_with_neg:
            neg_values = set(new_slot_has_values_id[slot_id])-set(slot_value_ids)
            for value_i in neg_values:
                sign_data = sign_data.append(pd.DataFrame({'from':[venue_id],'to':value_i,'sign':[-1]}), ignore_index=True)

sign_data.to_csv('mmdial_signed_new.csv', index=0)

'''
venuename_the star vista
venuename_star vista
'''