import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
from nltk.tree import Tree
import nltk
import re
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('wordnet')


class FactEntityExtraction:
    def __init__(self):
        pass

    def extract_named_entities(self, sentence):
        processed_sentence = self.pre_process_sentence(sentence)

        named_entity_chunk = nltk.ne_chunk(processed_sentence, binary=True)
        list_of_named_entities = self.get_continuous_chunks(named_entity_chunk)
        non_empty_values = [value for value in list_of_named_entities if value]
        print(non_empty_values)

        # print(named_entity_chunk)
        # print(list_of_named_entities)

        return list_of_named_entities

    # functions to remove noise
    # remove html tags
    def clean_html(self, text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    # remove brackets
    def remove_brackets(self, text):
        return re.sub('\[[^]]*\]', '', text)

    # remove special characters
    def remove_char(self, text):
        pattern = r'[^a-zA-z0–9\s]'
        text = re.sub(pattern, '', text)
        return text

    # remove noise(combine above functions)
    def remove_noise(self, text):
        text = self.clean_html(text)
        text = self.remove_brackets(text)
        text = self.remove_char(text)
        return text

    def stem_words(self, text):
        ps = PorterStemmer()
        stemmer_ss = SnowballStemmer("english")
        lemmatizer = WordNetLemmatizer()
        # text = ''.join(ps.stem(word) for word in text)
        # text = ''.join(stemmer_ss.stem(word) for word in text)

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
    def remove_stopwords(self, text):
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
    def join_back(self, text):
        return ' '.join(text)

    def pre_process_sentence(self, sentence):
        #
        # Pre process the sentence
        sentence = self.remove_noise(sentence)
        sentence = self.remove_stopwords(sentence)
        sentence = self.stem_words(sentence)

        # stop_words = stopwords.words('english')
        sentence = nltk.word_tokenize(sentence)
        final_tokens = []
        # for each in sentence:
        #     if each not in stop_words:
        #         final_tokens.append(each)
        sentence = nltk.pos_tag(sentence)
        print(sentence)

        return sentence

    @staticmethod
    def get_continuous_chunks(chunk):
        continuous_chunk = []
        current_chunk = []
        counter = 0

        for i in chunk:
            # Named entity will be in form of a tree
            if type(i) == Tree:
                current_chunk.append(" ".join([token for token, pos in i.leaves()]))

            else:
                # discontiguous, append to known contiguous chunks.
                if current_chunk:
                    named_entity = " ".join(current_chunk)
                    if named_entity not in continuous_chunk:
                        continuous_chunk.append(named_entity)
                        current_chunk = []
                else:
                    continue

            counter = counter + 1
            if counter == len(list(chunk)):
                # discontiguous, append to known contiguous chunks.
                if current_chunk:
                    named_entity = " ".join(current_chunk)
                    if named_entity not in continuous_chunk:
                        continuous_chunk.append(named_entity)
                        current_chunk = []
                else:
                    continue

        return continuous_chunk

    def process_fact(self, fact):
        self.extract_named_entities(fact)
