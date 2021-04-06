#https://akbaritabar.netlify.app/how_to_use_a_wikidata_dump
import bz2
import json
import pandas as pd
import pydash
'''
处理策略：1. 排除items，which is a subclass or instance of the 
most common wikimedia-internal adminstrative entities
2. only those point to at least one wikipedia page, 
using descriptive text as entity features
baseline: use one primary description per entity
使用规则进行挑选：entity e, ne(l):number of mentions of l, n(l):mentions of l of all entities
ne(l) 大的，n(l)大的，选择出现这个entity mention多的语言的description

'''
'''
{
  "id": "Q60",
  "type": "item", item for data items, properties for properites
  "labels": {}, in different languages language&value
  "descriptions": {}, in different languages
  "aliases": {}, in different languages
  "claims": {}, any number of statements
  "sitelinks": {},
  "lastrevid": 195301613,
  "modified": "2020-02-10T12:42:02Z"
}
'''
common_entities = ['Q4167836','Q24046192','Q20010800','Q11266439','Q11753321',
                'Q19842659','Q21528878','Q17362920','Q14204246','Q21025364','Q17442446',
                'Q26267864','Q4663903','Q15184295']
exclude_list = ['P279','P31']

i = 0
# an empty dataframe which will save items information
# you need to modify the columns in this data frame to save your modified data
df_record_all = pd.DataFrame(columns=['id', 'type',
                                      'english_label', 'longitude',
                                      'latitude', 'english_desc'])

def wikidata(filename):
    with bz2.open(filename, mode='rb') as f:
        f.read(2) # skip first two bytes: "{\n"
        for line in f:
            try:
                yield json.loads(line.rstrip(',\n'))
            except:
                continue

if __name__ == '__main__':
    lang_list = ['ja','de','es','ar','sr','tr','fa','ta','en',
                 'fr','it']
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumpfile',default='./data/wikidata/latest-all.json.bz2',type=str)
    args = parser.parse_args()
    entity_list = []
    entity_descriptions = {}
    entity_relations = {}
    for record in wikidata(args.dumpfile):
        # only extract items with geographical coordinates (P625)
        # with open('record.json','w') as f:
        #     json.dump(record,f,ensure_ascii=False,indent=2)
        #     exit()
        flag=True
        for rela in exclude_list:
            if pydash.has(record, 'claims.%s'%rela):
                # with open('record.json','w') as f:
                #     json.dump(record,f,ensure_ascii=False,indent=2)
                #     exit()
                inspect_list = pydash.get(record, 'claims.%s'%rela)
                for instance in inspect_list:
                    # if instance['mainsnak']['datavalue']['value']['id'] in common_entities:
                    if pydash.get(instance,'mainsnak.datavalue.value.id') in common_entities:
                        flag=False
                        break

            if flag==False:
                break

        if len(record['descriptions']) == 0:
            flag=False

        if flag:
            ###为True的时候才对entity进行存储
            ### 第一步，存实体
            # entity_list.append(record['id'])

            ### 第二步，存描述
            # descriptions = record['descriptions']
            # entity_descriptions[record['id']]={}
            # for lang in lang_list:
            #     if lang in descriptions:
            #         entity_descriptions[record['id']][lang] = descriptions[lang]['value']

            ### 第三步，存triples
            relations = record['claims']
            entity_relations[record['id']]=[]
            for proper in relations:
                datavalues = relations[proper]
                for data_value in datavalues:
                    if 'datavalue' not in data_value['mainsnak']:
                        continue
                    datavalue = data_value['mainsnak']['datavalue']['value']

                    if isinstance(datavalue,dict):
                        if 'id' not in datavalue:
                            continue
                        value = datavalue['id']
                    else:
                        value = datavalue

                    entity_relations[record['id']].append((value, proper))


    # print(len(entity_list))
    # with open('data/wikidata/entity_list.txt','w') as f:
    #     for e in entity_list:
    #         f.write(e+'\n')
    # print(len(entity_descriptions))
    # with open('data/wikidata/entity_descriptions.json','w') as f:
    #     json.dump(entity_descriptions,f,ensure_ascii=False,indent=2)
    print(len(entity_relations))
    with open('data/wikidata/entity_relations.json','w') as f:
        json.dump(entity_relations,f,ensure_ascii=False,indent=2)



        # if pydash.has(record, 'claims.P625'):
        #     print('i = '+str(i)+' item '+record['id']+'  started!'+'\n')
        #     latitude = pydash.get(record, 'claims.P625[0].mainsnak.datavalue.value.latitude')
        #     longitude = pydash.get(record, 'claims.P625[0].mainsnak.datavalue.value.longitude')
        #     print(longitude)
        #     print(latitude)
        #     exit()
        #     english_label = pydash.get(record, 'labels.en.value')
        #     item_id = pydash.get(record, 'id')
        #     item_type = pydash.get(record, 'type')
        #     english_desc = pydash.get(record, 'descriptions.en.value')
        #     df_record = pd.DataFrame({'id': item_id,
        #                               'type': item_type,
        #                               'english_label': english_label,
        #                               'longitude': longitude,
        #                               'latitude': latitude,
        #                               'english_desc': english_desc}, index=[i])
        #     df_record_all = df_record_all.append(df_record, ignore_index=True)
        #     i += 1
        #     print(i)
    #         if (i % 5000 == 0):
    #             pd.DataFrame.to_csv(df_record_all, path_or_buf='\\wikidata\\extracted\\till_'+record['id']+'_item.csv')
    #             print('i = '+str(i)+' item '+record['id']+'  Done!')
    #             print('CSV exported')
    #             df_record_all = pd.DataFrame(columns=['id', 'type', 'english_label', 'longitude', 'latitude', 'english_desc'])
    #         else:
    #             continue
    # pd.DataFrame.to_csv(df_record_all, path_or_buf='\\wikidata\\extracted\\final_csv_till_'+record['id']+'_item.csv')
    # print('i = '+str(i)+' item '+record['id']+'  Done!')
    # print('All items finished, final CSV exported!')