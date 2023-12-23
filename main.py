import sys
from entity_linking import *
from FactChecking import *
from llama_cpp import Llama
model_path = "models/llama-2-7b.Q4_K_M.gguf"
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('nps_chat')
nltk.download('nps_chat')

# llm = Llama(model_path=model_path, verbose=False)

# If you want to use larger models...
# model_path = "models/llama-2-13b.Q4_K_M.gguf"
# All models are available at https://huggingface.co/TheBloke. Make sure you download the ones in the GGUF format

class entity_extraction():
    def linking(self, entities, sents):
        entities_link = list()
        print(entities, sents)
        for entity in entities:
            print("linking ---", entity, sents)
            ent, lnk, dsp, score = entity_linking(entity, sents)
            entities_link.append([ent, lnk, dsp, score])
            print(entity, ":", entities_link[-1], entities_link[-1][2])
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



question = "Where is San Francisco?"

ans3 = 'What is the capital of Italy? 1067\n nobody knew. The city was now so large it could be divided into five ' \
       'different quarters, each with its own laws and customs. Each quarter was ruled by a warrior- '

if __name__ == '__main__':
    debug = False
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
    for line in question_file.readlines():
        count += 1
        # if line is empty
        # end of file is reached
        if not line or line == None or len(line) == 0:
            break
        print("Line{}: {}".format(count, line.strip().replace('\n', '').replace('\r', '')))
        line = line.strip().replace('\n', '').replace('\r', '')
        q_id = line.split('\t')[0]
        question = line.split('\t')[1]
        print("Asking the question \"%s\" to %s (wait, it can take some time...)" % (question, model_path))

        if debug:
            answer = ans3
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
        with open(output_file, 'w+', encoding='utf-8') as the_file:
            the_file.write(q_id + "\t" + "R\"" + answer + "\"\n")
            the_file.write(q_id + "\t" + "A\"" + str(result1) + "\"\n")
            the_file.write(q_id + "\t" + "C\"" + str(result2) + "\"\n")
            for ent in a_links:
                the_file.write(q_id + "\t" + "E\"" + str(ent[1]) + "\"\n")
