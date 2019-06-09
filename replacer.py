import csv


class WordReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map
    def replace(self, word):
        return [self.word_map.get(w, w) for w in word]
    
    
class CsvWordReplacer(WordReplacer):
    def __init__(self, fname):
        word_map = {}
        for line in csv.reader(open(fname)):
            word, syn = line
            if word.startswith("#"):
                continue
            word_map[word] = syn
        super(CsvWordReplacer, self).__init__(word_map)
