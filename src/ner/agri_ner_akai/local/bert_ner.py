from transformers import pipeline
from request import ModelRequest

class BertNERModel():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(BertNERModel, cls).__new__(cls)
        cls.nlp_ner = pipeline("ner", model="GautamR/akai_ner", tokenizer="GautamR/akai_ner")
        return cls.instance

    def inference(self, sentence):
        entities = self.nlp_ner(sentence)
        return self.aggregate_entities(sentence, entities)

    @staticmethod
    def aggregate_entities(sentence, entity_outputs):
        aggregated_entities = []
        current_entity = None

        for entity in entity_outputs:
            entity_type = entity["entity"].split("-")[-1]

            # Handle subwords
            if entity["word"].startswith("##"):
                # If we encounter an I-PEST or any other I- entity
                if "I-" in entity["entity"]:
                    if current_entity:  # Add previous entity
                        aggregated_entities.append(current_entity)
                
                    word_start = sentence.rfind(" ", 0, entity["start"]) + 1
                    word_end = sentence.find(" ", entity["end"])
                    if word_end == -1:
                        word_end = len(sentence)

                    current_entity = {
                        "entity_group": entity_type,
                        "score": float(entity["score"]),
                        "word": sentence[word_start:word_end].replace('.','').replace('?',''),
                        "start": float(word_start),
                        "end": float(word_end)
                    }
                    aggregated_entities.append(current_entity)
                    current_entity = None

                else:
                    if current_entity:
                    # If it's a subword but not an I- entity
                        current_entity["word"] += entity["word"][2:]
                        current_entity["end"] = entity["end"]
                        current_entity["score"] = float((current_entity["score"] + entity["score"]) / 2)  # averaging scores

            # Handle full words
            else:
                if current_entity:
                    aggregated_entities.append(current_entity)

                current_entity = {
                    "entity_group": entity_type,
                    "score": float(entity["score"]),
                    "word": entity["word"],
                    "start": float(entity["start"]),
                    "end": float(entity["end"])
                }

        if current_entity:
            aggregated_entities.append(current_entity)

        return aggregated_entities
