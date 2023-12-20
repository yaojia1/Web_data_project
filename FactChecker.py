import csv
from solver.FactEntityExtraction import FactEntityExtraction
from solver.WikipediaHelper import WikipediaHelper
import nltk


class FactChecker:
    def __init__(self):
        self.fact_entity_extraction = FactEntityExtraction()
        self.wikipedia_helper = WikipediaHelper()
        pass

    def check_fact(self, fact):

        # get named entities
        named_entities = self.fact_entity_extraction.extract_named_entities(fact)

        named_entities_with_pages = {}
        wikipedia_urls = []
        unique_entities = []

        for entity in named_entities:

            if not wikipedia_urls.__contains__(self.wikipedia_helper.get_entity_wikipage_cached(entity).url):
                wikipedia_urls.append(self.wikipedia_helper.get_entity_wikipage_cached(entity).url)
                named_entities_with_pages[entity] = self.wikipedia_helper.get_entity_wikipage_cached(entity)
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


def main():
    fact_checker = FactChecker()
    fact = "Amsterdam is capital of Netherlands"

    estimated_val = float(fact_checker.check_fact(fact))
    print('Yes' if estimated_val == 1 else 'No')
    # outputFactLine = estimated_val
    # writer.write(outputFactLine + "\n")

    # writer.close()


if __name__ == "__main__":
    main()
