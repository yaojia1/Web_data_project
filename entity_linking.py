from SPARQLWrapper import SPARQLWrapper, JSON
import wikipedia
import pandas as pd
import requests
import wikipediaapi
import re 
import sys

pd.set_option('display.max_columns', None)


## try with a random entity
entity = 'Rome'


### fetch possible entities from wikidata
def fetch_wikidata(params):
    url = 'https://www.wikidata.org/w/api.php'
    try:
        return requests.get(url, params=params)
    except:
        return 'There was and error'
    

 
params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'search': entity,
        'language': 'en'
    }
 
data = fetch_wikidata(params)
 
data = data.json()


### find corresponding wikipedia page name from wikidata id
def wikipage_name(id):
    endpoint_url = "https://query.wikidata.org/sparql"

    query = """#Namen van Wikipedia-artikelen in meerdere talen
    SELECT DISTINCT ?lang ?name WHERE {{
    ?article schema:about wd:{} . hint:Prior hint:runFirst true.
    ?article schema:inLanguage ?lang ;
        schema:name ?name ;
        schema:isPartOf [ wikibase:wikiGroup "wikipedia" ] .
    FILTER(?lang in ('en')) .
    FILTER (!CONTAINS(?name, ':')) .
    }}""".format(id)

    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


### create dataframe with possible entities and features
candidate_df = pd.DataFrame(columns=['ID', 'label', 'page_id', 'description', 'URL', 'wikipedia_page'])
for candidate in range(len(data['search'])):
    candidate_df.loc[candidate, 'ID'] = data['search'][candidate]['id']
    candidate_df.loc[candidate, 'label'] = data['search'][candidate]['display']['label']['value']
    candidate_df.loc[candidate, 'page_id'] = data['search'][candidate]['pageid']

    try:
        candidate_df.loc[candidate, 'description'] = data['search'][candidate]['display']['description']['value']
    except:
        candidate_df.loc[candidate, 'description'] = 'not found'
    candidate_df.loc[candidate, 'URL'] = data['search'][candidate]['url']

    results = wikipage_name(id=candidate_df.loc[candidate, 'ID'])
    if results['results']['bindings'] == []:
        continue
    candidate_df.loc[candidate, 'wikipedia_page'] = results['results']['bindings'][0]['name']['value']

print("Drop all rows where no wikipedia page is found")
candidate_df.dropna(subset=['wikipedia_page'], inplace=True)
print(candidate_df)



### context independent features
def dice_coefficient(entity, candidate):
    overlap = len(''.join(set(entity).intersection(candidate)))
    return 2 * overlap / (len(set(entity)) + len(set(candidate)))

def hamming_distance(entity, candidate): 
    distance = 0
    L = len(entity)
    for i in range(L):
        if entity[i] != candidate[i]:
            distance += 1
    return distance

def get_wikipedia_link_count(entity):
    # Use Wikipedia API to get information about the entity
    wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')
    page = wiki_wiki.page(entity)
    return len(repr(page.backlinks))



### context dependent features
sentence = "is rome the capital of italy"
vocab = re.split("\W", sentence)

def jaccard_similarity(list1, list2):
    list2 = re.split("\W", str(list2))
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union


def wiki_summary(descr):
    c = candidate_df.loc[candidate_df['description'] == descr, 'wikipedia_page']
    print(wikipedia.summary(c))
    return jaccard_similarity(sentence, wikipedia.summary(c))




candidate_df['dice_coeff'] = candidate_df['wikipedia_page'].apply(lambda x: dice_coefficient(entity, x))
candidate_df['hamming_dist'] = candidate_df['wikipedia_page'].apply(lambda x: hamming_distance(entity, x))
candidate_df['link_count'] = candidate_df['wikipedia_page'].apply(lambda x: get_wikipedia_link_count(x))
candidate_df['popularity_rate'] = candidate_df['link_count'].apply(lambda x: x / candidate_df['link_count'].sum())
candidate_df['jaccard_description'] = candidate_df['description'].apply(lambda x: jaccard_similarity(vocab, x))
candidate_df['jaccard_summary'] = candidate_df['description'].apply(lambda x: wiki_summary(x))


### To do: the summary does not always match with the wikipedia page, instead of looking for name, look for URL

print(candidate_df)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


## TO Do
def calculate_similarity(entity, candidate_entities):
    # Tokenize the mention and candidate entities
    # tokens = word_tokenize(mention.lower())
    # mention_tokens = ' '.join(tokens)

    # entity_tokens = [' '.join(word_tokenize(entity.lower())) for entity in candidate_entities]

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([entity] + candidate_entities)

    # Calculate cosine similarity between the mention and each candidate entity
    similarities = cosine_similarity(vectors[0], vectors[1:]).flatten()

    # Combine mention and entity pairs with their respective similarities
    ranked_entities = list(zip(candidate_entities, similarities))

    # Sort entities based on similarity in descending order
    ranked_entities.sort(key=lambda x: x[1], reverse=True)

    return ranked_entities

