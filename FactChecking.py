from __future__ import unicode_literals
import nltk
"""
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
"""
from nltk.tree import Tree
import nltk
import re
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import wikipedia

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


cache_dict = {}



# functions to remove noise
# remove html tags
def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


# remove brackets
def remove_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


# remove special characters
def remove_char(text):
    pattern = r'[^a-zA-z0–9\s]'
    text = re.sub(pattern, '', text)
    return text


# remove noise(combine above functions)
def remove_noise(text):
    text = text.replace("Question:", "")
    text = text.replace("Question", "")
    text = text.replace("Answer:", "")
    text = text.replace("Answer", "")
    text = clean_html(text)
    text = remove_brackets(text)
    text = remove_char(text)
    return text

def lemm_words(text):
    # sentences = nltk.sent_tokenize(text)
    ps = PorterStemmer()
    stemmer_ss = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()
    # Lemmatization
    token_words = text
    lt = []
    for word in token_words:
        word = str(word)
        if word == word.title():
            # lt.append(lemmatizer.lemmatize(word).capitalize())
            lt.append(word.capitalize())
        elif word.isupper():
            lt.append(lemmatizer.lemmatize(word).upper())
        else:
            lt.append(lemmatizer.lemmatize(word))
    return " ".join(lt)
def stem_words(text):
    ps = PorterStemmer()
    stemmer_ss = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()

    st = []
    token_words = text
    for word in token_words:
        word = str(word)
        if word == word.title():
            # st.append(ps.stem(word).capitalize())
            st.append(word.capitalize())
        elif word.isupper():
            st.append(ps.stem(word).upper())
        else:
            st.append(ps.stem(word))

    pt = []
    for word in st:
        word = str(word)
        if word == word.title():
            # pt.append(stemmer_ss.stem(word).capitalize())
            pt.append(word.capitalize())
        elif word.isupper():
            pt.append(stemmer_ss.stem(word).upper())
        else:
            pt.append(stemmer_ss.stem(word))

    lt = []
    for word in pt:
        word = str(word)
        if word == word.title():
            # lt.append(lemmatizer.lemmatize(word).capitalize())
            lt.append(word.capitalize())
        elif word.isupper():
            lt.append(lemmatizer.lemmatize(word).upper())
        else:
            lt.append(lemmatizer.lemmatize(word))
    return " ".join(lt)


# removing the stopwords from review
def remove_stopwords(text):
    # list to add filtered words from review
    # creating list of english stopwords
    stopword_list = stopwords.words('english')
    filtered_text = []
    # verify & append words from the text to filtered_text list
    for word in text.split():
        if word not in stopword_list:
            filtered_text.append(word)
    # add content from filtered_text list to new variable
    clean_review = filtered_text[:]
    # emptying the filtered_text list for new review
    filtered_text.clear()
    return clean_review


# join back all words as single paragraph
def join_back(text):
    return ' '.join(text)

def pre_only_sentence(sentence):
    # Pre process the sentence
    sentence = remove_noise(sentence)
    sentence = remove_stopwords(sentence)
    sentence = lemm_words(sentence)  # stem_words(sentence)
    print(sentence)
    return sentence

def pre_process_sentence(sentence):
    #
    # Pre process the sentence
    sentence = remove_noise(sentence)
    sentence = remove_stopwords(sentence)
    sentence = lemm_words(sentence)  #stem_words(sentence)
    print(sentence)

    # stop_words = stopwords.words('english')
    sentence = nltk.word_tokenize(sentence)
    final_tokens = []
    # for each in sentence:
    #     if each not in stop_words:
    #         final_tokens.append(each)
    sentence = nltk.pos_tag(sentence)
    # print(sentence)
    return sentence


# Get the Wikipedia page
def get_entity_wikipage_cached(entity):
    page = None
    if entity in cache_dict:
        page = cache_dict[entity]
        # print(entity+": "+page.url)
    else:
        try:
            page = wikipedia.page(entity, auto_suggest=False)
            # print(entity + ": " + page.url)
        except wikipedia.exceptions.DisambiguationError:
            try:
                page = wikipedia.page(entity, auto_suggest=True)
                # print(entity + ": " + page.url)
            except:
                page = None
        except:
            try:
                page = wikipedia.page(entity, auto_suggest=True)
                # print(entity + ": " + page.url)
            except:
                page = None

        cache_dict[entity] = page

    return page


# Get the relation of the collected named entities
def get_continuous_chunks(chunk):
    continuous_chunk = []
    current_chunk = []
    all_chunk = []
    counter = 0

    for i in chunk:
        counter = counter + 1
        # Named entity will be in form of a tree
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            print("token ".join([token+pos for token, pos in i]))
            #all_chunk.append(())

        else:
            # discontiguous, append to known contiguous chunks.
            if current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        # if counter == len(list(chunk)):
        #     # discontiguous, append to known contiguous chunks.
        #     if current_chunk:
        #         named_entity = " ".join(current_chunk)
        #         if named_entity not in continuous_chunk:
        #             continuous_chunk.append(named_entity)
        #             current_chunk = []
        #     else:
        #         continue

    return continuous_chunk


# Extract the named entities
def extract_named_entities(sentence):
    processed_sentence = pre_process_sentence(sentence)
    named_entity_chunk = nltk.ne_chunk(processed_sentence, binary=True)
    print(named_entity_chunk)
    list_of_named_entities = get_continuous_chunks(named_entity_chunk)
    non_empty_values = [value for value in named_entity_chunk if value not in list_of_named_entities]
    print(non_empty_values)
    return list_of_named_entities


# Check the fact and returns either 1 or 0
def check_fact(fact):
    # get named entities
    named_entities = extract_named_entities(fact)

    named_entities_with_pages = {}
    wikipedia_urls = []
    unique_entities = []

    for entity in named_entities:

        if not wikipedia_urls.__contains__(get_entity_wikipage_cached(entity).url):
            wikipedia_urls.append(get_entity_wikipage_cached(entity).url)
            named_entities_with_pages[entity] = get_entity_wikipage_cached(entity)
            unique_entities.append(entity)

    num_of_common_occurences = 0
    total_num_of_occurences = 0
    compared_entities = set()

    for subject_entity in unique_entities:
        for candidate_entity in unique_entities:

            subject_page = named_entities_with_pages[subject_entity]

            if subject_page is None:
                continue

            comma_seperated_entities = subject_entity + ',' + candidate_entity
            comma_seperated_entities_reverse = candidate_entity + ',' + subject_entity

            if subject_entity != candidate_entity and not comma_seperated_entities in compared_entities:

                # check if entities exist together
                if candidate_entity in subject_page.content:
                    num_of_common_occurences += 1

                total_num_of_occurences += 1
                compared_entities.add(comma_seperated_entities)
                compared_entities.add(comma_seperated_entities_reverse)

    if total_num_of_occurences == 0:
        return 0.0

    similarity_percentage = num_of_common_occurences / total_num_of_occurences

    if similarity_percentage > 0.7:
        return 1.0
    else:
        return 0.0


# Append random text as the final value
ran = "ran"
fact = "Amsterdam is the capital of Netherlands" # LLaMa output
user_input = "Is Amsterdam capital of Netherlands?" # Question

nltk.download('nps_chat')
posts = nltk.corpus.nps_chat.xml_posts()

posts_text = [post.text for post in posts]

# divide train and test in 80 20
train_text = posts_text[:int(len(posts_text) * 0.8)]

# Get TFIDF features
vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                             min_df=0.001,
                             max_df=0.7,
                             analyzer='word')

X_train = vectorizer.fit_transform(train_text)

question_types = ["whQuestion", "ynQuestion"]
f = open('ntlk_CheckQuestion.pickle', 'rb')


def is_ques_using_nltk(ques):
    ListQuestion = [ques]
    print(ques)
    question = vectorizer.transform(ListQuestion)
    print(question)
    f = open('./ntlk_CheckQuestion.pickle', 'rb')
    classifier = pickle.load(f)
    question_type = classifier.predict(question)
    print(question_type)
    return question_type in question_types


question_pattern = ["do i", "do you", "what", "who", "is it", "why", "would you", "how", "is there",
                    "are there", "is it so", "is this true", "to know", "is that true", "are we", "am i",
                    "question is", "tell me more", "can i", "can we", "tell me", "can you explain",
                    "question", "answer", "questions", "answers", "ask"]
open_question_partten = ["what", "who", "whose", "why", "how", "which", "when", "where"]

helping_verbs = ["is", "am", "can", "are", "do", "does"]


def wh_question(question):
    is_ques = False
    for pattern in open_question_partten:  # question_pattern:
        is_ques = pattern in question
        if is_ques:
            break
    return is_ques

def wh_answer(question, answer, q_entities):
    #if wh_question(question):
    # named_entities1, nonq = extract_named_entities(question, type="all")
    #sentence_arr = answer.split(".")
    #for sent in sentence_arr:
    # named_entities2, nona = extract_named_entities(answer, type="all")
    sentence = pre_only_sentence(question)
    processed_sentence = sentence.lower().strip()
    #named_entity_chunk = nltk.ne_chunk(pre_process_sentence(processed_sentence), binary=True)
    #non_empty_values = [value for value in a_entities if value not in q_entities]
    #print(named_entity_chunk, non_empty_values)
    statement = ""
    sentence = nltk.sent_tokenize(processed_sentence)
    if question in answer:
        answer = answer[len(question):]
        print("ans:",answer)
    print(sentence)
    if len(q_entities) == 1:
        statement = statement + q_entities[0] + " "
        for words in sentence:
            for word in words.split():
                if word not in question_pattern and word not in q_entities[0].lower().split():
                    statement = statement + word + " "
                    print("w:", word)
    else:
        for words in sentence:
            for word in words.split():
                if word not in question_pattern:
                    statement = statement + word + " "
                    print("w:", word)
    print(statement)
    statement = statement+answer
    print("statement:", statement)
    return statement

def choose_ent(q_entities, a_entities, desp, ascore):
    score = [1 - i / len(a_entities) for i in range(len(a_entities))]
    score = list(score)
    for i in range(len(a_entities)):
        score[i] += ascore[i][3]
        for dsp in desp:
            if a_entities[i] in dsp[2]:
                score[i] += 1
            for k in range(len(a_entities)):
                if q_entities[0] == a_entities[k]:
                    score[i] += (1 - abs(k - i) / len(a_entities))
    max1 = -1
    c = 0
    for i in range(len(a_entities)):
        if score[i] > max1:
            max1 = score[i]
            c = i
    print(score)
    return c
    #sentence_arr = answer.split(".")
    # for sent in sentence_arr:
    # named_entities2, nona = extract_named_entities(answer, type="all")


# check with custom pipeline if still this is a question mark it as a question
def is_question(question):
    question = question.lower().strip()
    if not False:#is_ques_using_nltk(question):
        is_ques = False
        # check if any of pattern exist in sentence
        for pattern in open_question_partten:  # question_pattern:
            is_ques = pattern in question
            if is_ques:
                break

        # there could be multiple sentences so divide the sentence
        sentence_arr = question.split(".")
        for sentence in sentence_arr:
            if len(sentence.strip()):
                # if question ends with ? or start with any helping verb
                # word_tokenize will strip by default
                first_word = nltk.word_tokenize(sentence)[0]
                if sentence.endswith("?") or first_word in helping_verbs:
                    is_ques = True
                    break
        return is_ques
    else:
        return True


def fact_checking(user_input, fact, q_ents, e_ents, dep_text, a_score):
    # Append random text as the final value
    ran = "ran"
    # fact = "Amsterdam is the capital of Netherlands" # LLaMa output
    # user_input = "Where is San Francisco?"  # "Is Amsterdam capital of Netherlands?" # Question
    if user_input in fact[:len(user_input)]:
        fact = fact[len(user_input):].strip()
        print("answer:", fact)
    if wh_question(user_input.lower().strip()):
        # Check the negation of the LLAMA returned text
        fact = wh_answer(user_input, fact, q_ents)


    nltk.download('nps_chat')
    posts = nltk.corpus.nps_chat.xml_posts()

    posts_text = [post.text for post in posts]

    # divide train and test in 80 20
    train_text = posts_text[:int(len(posts_text) * 0.8)]
    # print(posts_text)



    # Check whether the user input a question or statement
    is_q = str(is_question(user_input.lower().strip()))
    print("Is this a question: " + is_q)
    f = open('ntlk_NegationClassifier_Blob.pickle', 'rb')
    classifier = pickle.load(f)

    # Check whether input is a question or answer
    # When classifying text, features are extracted automatically
    q_type = classifier.classify(fact)
    print('True' if q_type == 'pos' else 'False')
    estimated_val2 = float(check_fact(fact + " " + ran))
    print('a Correct' if estimated_val2 == 1 else 'a Incorrect')
    if is_q == 'yes' and not wh_question(user_input):
        # statement yes/no
        estimated_val = float(check_fact(user_input + " " + ran))
        print('q Correct' if estimated_val == 1 else 'q Incorrect')

        if estimated_val2 == 1:  # estimated_val == estimated_val2 or
            return 'yes' if q_type == 'pos' else 'no', True
        else:
            return 'yes' if q_type == 'pos' else 'no', False
    else:
        return choose_ent(q_ents, e_ents, dep_text, a_score), True if estimated_val2 == 1 else False










