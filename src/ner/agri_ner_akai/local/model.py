from transformers import pipeline
from request import ModelRequest
from regex_parse_ner import RegNERModel
from bert_ner import BertNERModel

class Model():
    def __init__(self, context):
        self.context = context
        print("Loading models...")
        self.regex_model = RegNERModel()
        print("Regex model loaded successfully")
        self.bert_model = BertNERModel()
        print("Bert model loaded successfully")

    def combine_entities(self, reg_entities, bert_entities):
        combined_entities = reg_entities

        for entity in bert_entities:
            if entity['entity_group'] not in combined_entities:
                combined_entities[entity['entity_group']] = []

            entity_info = {
                'name': entity['word'],
                'start': entity['start'],
                'end': entity['end'],
                'score': entity['score']
            }

            combined_entities[entity['entity_group']].append(entity_info)

        return combined_entities

    async def inference(self, request: ModelRequest):
        sentence = request.text
        types = request.type

        reg_entities = self.regex_model.inference(sentence)
        bert_entities = self.bert_model.inference(sentence)

        combined_entities = self.combine_entities(reg_entities, bert_entities)

        final_entities = {}

        if types is None:
            return combined_entities

        for entity_group in combined_entities:
            if entity_group in types:
                final_entities[entity_group] = combined_entities[entity_group]

        return final_entities