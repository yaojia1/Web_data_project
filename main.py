from FactChecker import FactChecker
from FactEntityExtraction import FactEntityExtraction
from WikipediaHelper import WikipediaHelper
from TextProcess import *
from entity_linking import *

model_path = "models/llama-2-7b.Q4_K_M.gguf"
llm = Llama(model_path=model_path, verbose=False)
from llama_cpp import Llama
# If you want to use larger models...
# model_path = "models/llama-2-13b.Q4_K_M.gguf"
# All models are available at https://huggingface.co/TheBloke. Make sure you download the ones in the GGUF format

class entity_extraction():
    question = []
    text = []
    entities = []
    answer = []
    completion = []
    def oldinit(self):
        self.fact_entity_extraction = FactEntityExtraction()
        self.wikipedia_helper = WikipediaHelper()
        prompt = input("Type your question or q to quit (for instance: \"The capital of Italy is \") and type ENTER "
                       "to finish:\n")
        while prompt != "q":
            print("Computing the answer (can take some time)...")
            self.question.append(prompt)
            self.completion.append(llm(prompt))
            print("COMPLETION: %s" % self.completion[-1])
            prompt = input(
                "Type your question or q to quit (for instance: \"The capital of Italy is \") and type ENTER to "
                "finish:\n")
    def linking(self, entities, sents):
        entities_link = list()
        for entity in entities:
            ent, lnk = entity_linking(entity, sents)
            entities_link.append([ent, lnk])
            print(entity, ":", entities_link[-1])
        return entities_link
    def run_question(self, sents):
        #nlp
        entities = []
        tags = []
        sentence = nltk.sent_tokenize(sents)
        for sent in sentence:
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent)), binary=False):
                if hasattr(chunk, 'label'):
                    entities.append(' '.join(c[0] for c in chunk))
                    tags.append(chunk.label())
        # print(entities)
        entities_tags = list(set(zip(entities, tags)))
        # print(entities)
        print(entities_tags)
        # linking
        entities_link = self.linking(entities, sents)
        return entities_link

    def text_processing(self, question, answer):
        # question
        # get named entities
        factexa = FactEntityExtraction()
        question_entities = factexa.extract_named_entities(question)
        answer_entities = factexa.extract_named_entities(answer)
        print("question:", question_entities)
        print("answer:",answer_entities)
        # completion
        pass
    def enety_recognition(self):

        pass

    def disambiguation(self):
        pass
    def answer_extraction(self):

        pass
    def fact_checking(self):
        pass

    def main(self):
        pass
question = "What is the capital of Italy? "
# q1 = entity_extraction().run_question()
ans1 = 'What is the capital of Italy? 2019\n obviously, Rome.\n'
ans2 = "What is the capital of Italy? 437 people answered this question\n shouldn't be in my way, I'd help you if it " \
       "didn't matter. "
asn3 = 'What is the capital of Italy? 1067\n nobody knew. The city was now so large it could be divided into five ' \
       'different quarters, each with its own laws and customs. Each quarter was ruled by a warrior- '
ans4 = "Where is San Francisco? 2010-05-03 19:14\n everyone says it's in the Bay Area but i have never heard of it. i " \
       "am driving from Los Angeles to San "

#entity_extraction().text_processing(question, ans1)
#entity_extraction().text_processing(question, ans2)
question = "Where is San Francisco?"
#entity_extraction().text_processing(question, ans4)
question = "What is the capital of Italy? "
question_entites = entity_extraction().run_question(question)
answer_entities = entity_extraction().run_question(ans4)



