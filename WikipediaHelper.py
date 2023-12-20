import wikipedia

class WikipediaHelper:

    def __init__(self):
        self.cache_dict = {}
        pass

    def get_entity_wikipage_cached(self, entity):
        page = None
        if entity in self.cache_dict:
            page = self.cache_dict[entity]
            # print(entity+": "+page.url)
        else:
            try:
                page = wikipedia.page(entity, auto_suggest=False)
                # print(entity + ": " + page.url)
            except wikipedia.exceptions.DisambiguationError:
                try:
                    page = wikipedia.page(entity, auto_suggest=True)
                    # print(entity + ": " + page.url)
                except:
                    page = None
            except:
                try:
                    page = wikipedia.page(entity, auto_suggest=True)
                    # print(entity + ": " + page.url)
                except:
                    page = None

            self.cache_dict[entity] = page

        return page

