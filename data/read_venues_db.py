import pandas as pd
import json
import pickle
from collections import OrderedDict
import nltk
import nltk.stem as ns
#nltk.download('wordnet')

lemmatizer = ns.WordNetLemmatizer()
#lemma = lemmatizer.lemmatize('some images','n')
with open("./venues_db_new.txt","r", encoding="utf-8") as f:
    venue_db = json.load(f)

with open("./slots_values.txt","r", encoding="utf-8") as f:
    slots_values = json.load(f)

open_span_list = []
with open("./open_span_candidates_frequencies_filtered_1063.txt","r", encoding="utf-8") as f:
    for line in f.readlines():
        #print(line)
        word = line.split('\t')[0]
        '''
        new_word = []
        for word_i in word.split(' '):
            word_i = lemmatizer.lemmatize(word_i,'n')
            new_word.append(word_i)
        word_combine = ' '.join(new_word)       
        open_span_list.append(word_combine)
        '''
        open_span_list.append(word)

'''
venue_db[0].keys()
dict_keys(['requestable', 'informable'])
venue_db[0]['requestable'].keys()
dict_keys(['venueName', 'venueAddress', 'desc', 'url', 'telephone', 'timeFrame', 'tips', 'images'])
venue_db[0]['informable'].keys()
dict_keys(['venueScore', 'Credit Cards', 'Menus', 'venueNeigh', 'terms', 'Parking', 'Dining Options', 'Restroom', 'Wi-Fi', 'Music', 'Outdoor Seating', 'Wheelchair Accessible', 'Smoking', 'Reservations', 'Drinks', 'venueCat', 'price'])
'''

# 定义包含互斥关系的slot
slot_with_neg = set(['venueScore', 'Credit Cards', 'Menus', 'venueNeigh', 'Parking', \
                     'Dining Options', 'Restroom', 'Wi-Fi', 'Music', 'Outdoor Seating', \
                     'Wheelchair Accessible', 'Smoking', 'Reservations', 'Drinks', 'price'])
                     
slot_value_with_neg = OrderedDict()
slot_value2id = OrderedDict()
slot_value_id = -1  # id从0开始
for slot in slot_with_neg:
    value_list = slots_values[slot]
    slot_value_with_neg[slot] =  set(value_list)
    for value in value_list:
        slot_value_id +=1
        slot_value = slot + '_' + value
        slot_value2id[slot_value] = int(slot_value_id)

slot_value2id_df = pd.DataFrame.from_dict(slot_value2id, orient='index')
slot_value2id_df.to_csv('slot_value2id.csv', header=0)

venueName2id = OrderedDict()
sign_data = pd.DataFrame(columns=('from','to','sign'))
venue_db_dict = {}
for venue_idx, venue_data in enumerate(venue_db):
    data_request = venue_data['requestable']
    data_inform = venue_data['informable']
    venueName = data_request['venueName']
    venue_db_dict[venueName] = {}
    for slot, value in data_inform.items():
        venue_db_dict[venueName][slot] = value

    if venueName not in venueName2id.keys():
        venueName2id[venueName] = venue_idx
    for slot, value in data_inform.items():
        if slot =='venueCat':
            continue
        #slot_value
        #print('slot: ', slot)
        if slot=='terms':
            slot='OpenSpan'
        
        if len(value)>0:  # 有的value是空。          
            #print(type(value))
            try:
                # 字符串，转化为小写
                value = value.lower()
                value = value.strip()
                value = [value]  # 字符串放到list
                #print(len(value))
            except:
                #长度1: ['happy hour']
                #长度>1,有多个value: ['beer', ' full bar', ' cocktails'] ['sunsets', 'beach', 'casual']
                # list中每个字符串都转化为小写
                value = [i.lower() for i in value]
            new_value = [] # for neg
            for value_i in value:
                value_i = value_i.strip()
                value_i = value_i.replace('-', ' ')
                 #Yes (incl. American Express & MasterCard)
                if 'yes' in value_i:
                    value_i = 'yes'

                if slot=='OpenSpan' or slot=='Menus' or slot=='Drinks' or slot=='Wi-Fi':
                    # value转化为单数形式
                    #'Menus_bar snacks', 'Drinks_cocktails', 'Wi-Fi_paid': 动词
                    new_word = []
                    for word_i in value_i.split(' '):
                        word_i2 = lemmatizer.lemmatize(word_i,'n')
                        if word_i2==word_i:
                            word_i2 = lemmatizer.lemmatize(word_i,'v')
                        new_word.append(word_i2)
                    value_i = ' '.join(new_word)
                    if not value_i in set(open_span_list):
                        continue
                new_value.append(value_i)
                slot_value = slot + '_' + value_i 
                # for debug
                try:
                    slot_value_id = slot_value2id[slot_value]
                except:
                    if slot!='OpenSpan':
                        print(slot_value)
                
                sign_data = sign_data .append(pd.DataFrame({'from':[venue_idx],'to':slot_value_id,'sign':[1]}), ignore_index=True)
                
            if slot in slot_with_neg:
                neg_values = slot_value_with_neg[slot]-set(new_value)
                for value_n in neg_values:
                    slot_value = slot + '_' + value_n
                    slot_value_id = slot_value2id[slot_value]                
                    sign_data = sign_data.append(pd.DataFrame({'from':[i],'to':slot_value_id,'sign':[-1]}), ignore_index=True)


sign_data.to_csv('mmdial_signed.csv', index=0)

venueName2id_df = pd.DataFrame.from_dict(venueName2id, orient='index')
venueName2id_df.to_csv('venueName2id.csv', header=0)


'''

手动改掉了
slot-value: "central region , singapore",  # region后面有个空格
db: "central region, singapore",

slot-value: "central region , singapore",  # region后面有个空格
db: "central region, singapore",

slot_value txt的改动
Reservations 增加 groups only    
    "Reservations": [
        "yes",
        "no",
        "groups only"
    ],

venueScore 增加两个
venueScore_6.9/10
venueScore_5.4/10

'''