from typing import Dict
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

class NLPManager:

    def __init__(self):
        # initialize the model here
        model_name = 'deepset/roberta-base-squad2'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    def qa(self, context: str) -> Dict[str, str]:
        # perform NLP question-answering
        qa = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer)
        answer = {}
        labels_list=["heading", "tool", "target"]
        for label in labels_list:
            input = f"What is the {label}?"
            answer[label] = qa(question=input, context=context)['answer']

        return {"heading": self.stringToNumbers(answer["heading"]), "tool": answer["tool"], "target": answer["target"]}    

    def stringToNumbers(self, text: str):
        word_to_num_dict = {
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'zero': '0'
        }
        textSplit = text.split()

        output = ""
        
        for word in textSplit:
            digit = ""
            for key in word_to_num_dict:
                if key in word:
                    digit = word_to_num_dict[key]
                    break
            output = output + str(digit)
        return output