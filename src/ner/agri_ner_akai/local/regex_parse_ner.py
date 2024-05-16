import re
import spacy
from datetime import datetime, timedelta

class RegNERModel():
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        print("Model loaded successfully")

    def detect_email(self, sentence):
        email_regex_pattern = '[A-Za-z0-9._%+-]*@[A-Za-z0-9.-]*\.[A-Z|a-z]*'
        emails_matches = []

        for match in re.finditer(email_regex_pattern, sentence):
            emails_matches.append( {"name": match.group(), "start": match.start(), "end": match.end(), "score": 1.0} )

        return emails_matches

    def detect_time(self, sentence):
        time_regex = r'\b(?:1[0-2]|0?[1-9])(?::[0-5][0-9])?(?:\s?[ap]m)?\b'
        times = []

        for match in re.finditer(time_regex, sentence, re.IGNORECASE):
            times.append( {"name": match.group(), "start": match.start(), "end": match.end(), "score": 1.0} )

        return times

    def detect_phone_numbers(self, sentence):
        phone_regex = r'(\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})'

        phone_numbers = []
        for match in re.finditer(phone_regex, sentence):
            phone_numbers.append( {"name": match.group(), "start": match.start(), "end": match.end(), "score": 1.0} )

        return phone_numbers

    def detect_numbers_with_units(self, sentence, phone_numbers):
        number_unit_regex = r'(?<!\d)(\d+(?:\.\d+)?)(?:\s+)(\w+)(?!\d)'

        numbers_with_units = []

        for match in re.finditer(number_unit_regex, sentence):
            number, unit = match.groups()
            if number not in phone_numbers:
                numbers_with_units.append( {"name": f"{number} {unit}", "start": match.start(), "end": match.end(), "score": 1.0} )

        return numbers_with_units

    def detect_dates(self, sentence):
        # Current date
        today = datetime.now()

        # Define regex patterns for relative date expressions
        patterns = [
            r"(next|agle)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|somvar|mangalwar|budhwar|guruwar|shukrawar|shaniwar|raviwar)",
            r"(kal)",
            r"(next|agle)\s+(week|month|year|hafte|mahine|saal)"
        ]

        # Initialize empty list to store detected dates
        detected_dates = []

        # Iterate through patterns and search for matches in text
        for pattern in patterns:
            for matchdates in re.finditer(pattern, sentence.lower()):
                match = matchdates.groups()
                if match[0] in ['next', 'agle']:
                    if match[1] in ['monday', 'somvar']:
                        # Find next Monday
                        days_until_weekday = (today.weekday() - 1) % 7
                        next_date = today + timedelta(days=days_until_weekday)
                        detected_dates.append({"name": next_date.strftime("%d-%m-%Y"), "start": matchdates.start(), "end": matchdates.end(), "score": 1.0})
                    elif match[1] in ['tuesday', 'mangalwar']:
                        # Find next Tuesday
                        days_until_weekday = (today.weekday() - 0) % 7
                        next_date = today + timedelta(days=days_until_weekday )
                        detected_dates.append({"name": next_date.strftime("%d-%m-%Y"), "start": matchdates.start(), "end": matchdates.end(), "score": 1.0})
                    elif match[1] in ['wednesday', 'budhwar']:
                        # Find next Wednesday
                        days_until_weekday = (today.weekday() +1) % 7
                        next_date = today + timedelta(days=days_until_weekday )
                        detected_dates.append({"name": next_date.strftime("%d-%m-%Y"), "start": matchdates.start(), "end": matchdates.end(), "score": 1.0})
                    elif match[1] in ['thursday', 'guruwar']:
                        # Find next Thursday
                        days_until_weekday = (today.weekday() +2) % 7
                        next_date = today + timedelta(days=days_until_weekday )
                        detected_dates.append({"name": next_date.strftime("%d-%m-%Y"), "start": matchdates.start(), "end": matchdates.end(), "score": 1.0})
                    elif match[1] in ['friday', 'shukrawar']:
                        # Find next Friday
                        days_until_weekday = (today.weekday() +3) % 7
                        next_date = today + timedelta(days=days_until_weekday )
                        detected_dates.append({"name": next_date.strftime("%d-%m-%Y"), "start": matchdates.start(), "end": matchdates.end(), "score": 1.0})
                    elif match[1] in ['saturday', 'shaniwar']:
                        # Find next Saturday
                        days_until_weekday = (today.weekday() +4) % 7
                        next_date = today + timedelta(days=days_until_weekday )
                        detected_dates.append({"name": next_date.strftime("%d-%m-%Y"), "start": matchdates.start(), "end": matchdates.end(), "score": 1.0})
                    elif match[1] in ['sunday', 'raviwar']:
                        # Find next Sunday
                        days_until_weekday = (today.weekday() +5) % 7
                        next_date = today + timedelta(days=days_until_weekday )
                        detected_dates.append({"name": next_date.strftime("%d-%m-%Y"), "start": matchdates.start(), "end": matchdates.end(), "score": 1.0})
                    elif match[1] in ['week', 'hafte']:
                        # Find next week
                        next_date = today + timedelta(days=(7 - today.weekday())+6)
                        detected_dates.append({"name": next_date.strftime("%d-%m-%Y"), "start": matchdates.start(), "end": matchdates.end(), "score": 1.0})
                    elif match[1] in ['month', 'mahine']:
                        # Find next month
                        next_date = today.replace(day=1, month=today.month+1)
                        detected_dates.append({"name": next_date.strftime("%d-%m-%Y"), "start": matchdates.start(), "end": matchdates.end(), "score": 1.0})
                    elif match[1] in ['year', 'saal']:
                        # Find next year
                        next_date = today.replace(day=1, month=1, year=today.year+1)
                        detected_dates.append({"name": next_date.strftime("%d-%m-%Y"), "start": matchdates.start(), "end": matchdates.end(), "score": 1.0})
                elif match[0] == 'kal':
                    # Find tomorrow's date
                    next_date = today + timedelta(1)
                    detected_dates.append({"name": next_date.strftime("%d-%m-%Y"), "start": matchdates.start(), "end": matchdates.end(), "score": 1.0})

        return detected_dates

    def inference(self, sentence):
        detected_emails = self.detect_email(sentence)
        detected_time = self.detect_time(sentence)
        detected_phone_numbers = self.detect_phone_numbers(sentence)
        detected_number_units = self.detect_numbers_with_units(sentence, detected_phone_numbers)
        detected_dates = self.detect_dates(sentence)

        aggregated_entities = {}
        
        if detected_emails:
            aggregated_entities["email"] = detected_emails
        if detected_time:
            aggregated_entities["time"] = detected_time
        if detected_phone_numbers:
            aggregated_entities["phone_number"] = detected_phone_numbers
        if detected_number_units:
            aggregated_entities["number_with_unit"] = detected_number_units
        if detected_dates:
            aggregated_entities["date"] = detected_dates

        return aggregated_entities
