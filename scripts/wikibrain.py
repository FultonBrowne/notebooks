#Hook up wikipedia to a digital brain

import wikipediaapi, sys
class WikiBrain:
    def __init__(self, lang):
        self.wiki = wikipediaapi.Wikipedia(lang)

    def pageexist(self, term):
        return self.wiki.page(term)
    
    def summary(self, page):
        return self.wiki.page(page).summary
    def full_page(self, page):
        return self.wiki.page(page).text

def test(t):
    wikibrain = WikiBrain("en")
    print(wikibrain.pageexist(t))
    print(wikibrain.summary(t))
    print(wikibrain.full_page(t))
