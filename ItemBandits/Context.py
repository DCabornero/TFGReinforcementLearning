import pandas as pd
import numpy as np

from Datos import Datos

class Context:
    def __init__(self,itemName='movieId',tagName='tag'):
        self.names = {'item':itemName,
                      'tag': tagName}


    def read_csv(self,tags):
        self.tags = Datos(tags)
        self.listItems = np.unique(self.tags.extraeCols([self.names['item']]))
        listTags,count = np.unique(self.tags.extraeCols([self.names['tag']]),return_counts=True)

        self.tag_counts = dict(zip(listTags,count))
        # self.tag_counts['no-tag'] = 0
        self.numTags = len(self.tag_counts.keys())

        self.occurrences = pd.DataFrame(0,index=self.listItems,columns=listTags)
        tag_hist = self.tags.extraeCols([self.names['item'],self.names['tag']])
        for row in tag_hist:
            self.occurrences.at[row[0],row[1]] += 1
        # self.occurrences['no-tag'] = 0

    def tf(self,tag,item):
        f = self.occurrences.at[item,tag]
        if f:
            return 1 + np.log2(f)
        else:
            return 0

    def idf(self,tag):
        total = np.sum(list(self.tag_counts.values()))
        count = self.tag_counts[tag]
        if count:
            return np.log2(total/count)
        else:
            return 0

    def context(self,item):
        cont = np.zeros(self.numTags)
        if item not in self.listItems:
            self.listItems = np.append(self.listItems,item)
            vect = np.zeros(self.numTags)
            vect[-1] = 1
            self.occurrences.loc[item] = vect
            self.tag_counts['no-tag'] += 1
        for i,tag in enumerate(self.tag_counts.keys()):
            cont[i] = self.tf(tag,item)*self.idf(tag)
        return cont
