import pandas as pd
from ast import literal_eval

keywords = pd.read_csv('keywords.csv')
meta = pd.read_csv('movies_metadata.csv')
ratings = pd.read_csv('ratings_small.csv')
credit = pd.read_csv('credits.csv')

ratings = ratings.rename(columns={'movieId':'id'})
#Convert IDs to numeric from string
meta = meta[meta.status=="Released"]
meta = meta[meta.revenue!=0]
meta = meta[meta.budget!=0]
meta.id = pd.to_numeric(meta.id,errors="coerce")
meta = meta.dropna(subset=["id"])
meta.release_date=pd.to_datetime(meta.release_date, format = '%Y-%m-%d', errors="coerce")
meta = meta.dropna(subset=["release_date"])
meta = meta.drop_duplicates(subset=['id'])
meta = meta.drop(['homepage','poster_path','production_countries','video','spoken_languages','original_title'], axis =1)
keywords = keywords[keywords.keywords!="[]"]
keywords = keywords.drop_duplicates(subset=['id'])
meta = pd.merge(meta,
                 keywords,
                 on='id',how='left')

tab_info=pd.DataFrame(meta.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(meta.isnull().sum()).T.rename(index={0:'null values'}))
tab_info=tab_info.append(pd.DataFrame(meta.isnull().sum()/meta.shape[0]*100).T.
                         rename(index={0:'null values (%)'}))


meta.popularity = pd.to_numeric(meta.popularity, errors='coerce')
tasks = ['genres','production_companies','keywords']

def count_word(df, ref_col, liste):
    keyword_count = dict()
    for s in liste: keyword_count[s] = 0
    for liste_keywords in df[ref_col]:        
        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue        
        for s in [s for s in liste_keywords if s in liste]: 
            if pd.notnull(s): keyword_count[s] += 1
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count

def convert(task):
    temp_dict = dict()
    temp_arr = []
    temp_series = meta[task]
    for item in temp_series:
        if type(item) != type(str()):
            continue
        t = literal_eval(item)
        if type(t)!=type([]):
            continue
        temp =[]
        for i in t:
            temp.append(i['name'])
            if str(i['name']) not in temp_dict.keys():
                temp_dict[str(i['name'])] = 1
            else:
                temp_dict[str(i['name'])]+=1
        temp_arr.append(temp)
    meta[task]=pd.Series(temp_arr)
    return temp_dict

genre_dict = convert(tasks[0])
prodcomp_dict = convert(tasks[1])
convert(tasks[2])
