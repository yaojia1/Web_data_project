import sys

from FactChecker import FactChecker
#from FactEntityExtraction import FactEntityExtraction
from WikipediaHelper import WikipediaHelper
from TextProcess import *
from entity_linking import *
from FactChecking import *

model_path = "models/llama-2-7b.Q4_K_M.gguf"
#llm = Llama(model_path=model_path, verbose=False)
#from llama_cpp import Llama
# If you want to use larger models...
# model_path = "models/llama-2-13b.Q4_K_M.gguf"
# All models are available at https://huggingface.co/TheBloke. Make sure you download the ones in the GGUF format

class entity_extraction():
    #def sim_score(self, string1, string2):

    def linking(self, entities, sents):
        entities_link = list()
        print(entities, sents)
        for entity in entities:
            print("linking ---", entity, sents)
            ent, lnk, dsp, score = entity_linking(entity, sents)
            entities_link.append([ent, lnk, dsp, score])
            print(entity, ":", entities_link[-1], entities_link[-1][2] )
        return entities_link
    def run_question(self, sents):
        # linking
        entities, tags = self.text_processing(sents)
        entities_link = self.linking(entities, sents)
        return entities_link, entities

    def text_processing(self, sents):
        # nlp
        entities = []
        tags = []
        dsp = []
        sentence = pre_only_sentence(sents)
        sentence = nltk.sent_tokenize(sentence)
        for sent in sentence:
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent)), binary=False):
                if hasattr(chunk, 'label'):
                    entities.append(' '.join(c[0] for c in chunk))
                    tags.append(chunk.label())
        entities_tags = list(set(zip(entities, tags)))
        print(entities)
        print(entities_tags)
        return entities, tags
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


question = "Where is San Francisco?"
#entity_extraction().text_processing(question, ans4)
question1 = "What is the capital of Italy? "

#question_entites, q = entity_extraction().text_processing(question)
#allt = entity_extraction().linking(question_entites, question)
#answer_entities, a = entity_extraction().text_processing(ans4)
#alit2 = entity_extraction().linking(answer_entities, ans4)
#print(question_entites, q)
#print(answer_entities,a)
#print(allt[-1])
asn3 = 'What is the capital of Italy? 1067\n nobody knew. The city was now so large it could be divided into five ' \
       'different quarters, each with its own laws and customs. Each quarter was ruled by a warrior- '

#a, b = fact_checking(question, ans4, question_entites, answer_entities, allt, alit2)
#print(a, b)
if __name__ == '__main__':
    debug = True
    input_file = "example_input.txt"
    output_file = "output.txt"
    print(sys.argv)
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if len(sys.argv) >= 2:
            output_file = sys.argv[2]
    question_file = open(input_file, 'r')
    count = 0

    q = entity_extraction()
    if not debug:
        llm = Llama(model_path=model_path, verbose=False)

    while True:
        count += 1

        # Get next line from file
        line = question_file.readline()
        # if line is empty
        # end of file is reached
        if not line:
            break
        print("Line{}: {}".format(count, line.strip().replace('\n','').replace('\r','')))
        line = line.strip().replace('\n','').replace('\r','')
        q_id = line.split('\t')[0]
        question = line.split('\t')[1]
        print("Asking the question \"%s\" to %s (wait, it can take some time...)" % (question, model_path))

        if debug:
            answer = ans4
        else:
            output = llm(
                question,  # Prompt
                max_tokens=32,  # Generate up to 32 tokens
                stop=["Q:", "\n"],  # Stop generating just before the model would generate a new question
                echo=True  # Echo the prompt back in the output
            )
            print("Here is the output")
            print(output['choices']['text'])
            answer = output['choices']['text']

        q_links, question_entites = q.run_question(question)
        a_links, answer_entities = q.run_question(answer)

        a1, a2 = fact_checking(question, answer, question_entites, answer_entities, q_links, a_links)
        if type(a1) == int:
            result1 = a_links[a1][1]
        else:
            result1 = a1
        result2 = 'correct' if a2 else 'incorrect'
        with open(output_file, 'a', encoding='utf-8') as the_file:
            the_file.write(q_id+"\t"+"R\""+answer+"\"\n")
            the_file.write(q_id+"\t"+"A\"" + str(result1)+"\"\n")
            the_file.write(q_id+"\t"+"C\"" + str(result2) +"\"\n")
            for ent in a_links:
                the_file.write(q_id+"\t"+"E\"" + str(ent[1]) +"\"\n")



