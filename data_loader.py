import nltk
import pandas as pd
import sys
import os
import numpy as np

class DataLoader():

    data_path1 = "./data/Kindle_Store_5.json"
    data_path2 = "./data/Digital_Music_5.json"

    def __init__(self, data_path = data_path1) -> None:
        
        self.table = pd.read_json(data_path, 'records', lines = True);


    def load_table(self, samples = 200):

        asins = set()
        while len(asins) != 200:
            ind = np.random.randint(len(self.table))
            asins.add(self.table.iloc[ind]['asin'])
        inds = []
        for row in self.table.iterrows():
            if row[1]['asin'] in asins:
                inds.append(row[0])


        return self.table.iloc[inds]


    def load_review_text(self, samples = 200):

        reviews = self.table.sample(samples)['reviewText']
        return reviews.to_numpy()

    

