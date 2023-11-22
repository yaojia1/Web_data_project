from ctransformers import AutoModelForCausalLM

repository="TheBloke/Llama-2-7B-GGUF"
model_file="llama-2-7b.Q4_K_M.gguf"
llm = AutoModelForCausalLM.from_pretrained(repository, model_file=model_file, model_type="llama")

class entity_extraction():
    question = []
    text = []
    entities = []
    answer = []
    completion = []
    def __init__(self):
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
    def text_processing(self):
        # question
        # completion
        pass
    def enety_recognition(self):
        pass
    def linking(self):
        pass
    def disambiguation(self):
        pass
    def answer_extraction(self):
        pass
    def fact_checking(self):
        pass

q1 = entity_extraction()

