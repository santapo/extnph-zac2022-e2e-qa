from flashtext import KeywordProcessor
from memoization import cached
class FlashSearch:
    def __init__(self,keywords,is_file=False,with_replace=False):
        self.keyword_extractor = KeywordProcessor()
        if is_file:
            self.keyword_extractor.add_keyword_from_file(keywords)
        elif isinstance(keywords,list):
            self.keyword_extractor.add_keywords_from_list(keywords)
        elif isinstance(keywords,dict):
            self.keyword_extractor.add_keywords_from_dict(keywords)
    @cached(ttl=3600)
    def find(self,text):
        kws = self.keyword_extractor.extract_keywords(text)
        return [x for x in kws if len(x.split()) > 1]


