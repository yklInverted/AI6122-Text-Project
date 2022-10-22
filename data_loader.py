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

        inds = np.random.choice(range(len(self.table)), samples, replace=False)
        print(len(inds))
        asins = [row[1]['asin'] for row in self.table.iterrows() if row[0] in inds]
        print(len(asins))

        inds = []
        for row in self.table.iterrows():
            if row[1]['asin'] in asins:
                inds.append(row[0])


        return self.table.iloc[inds]


    def load_review_text(self, samples = 200):

        reviews = self.table.sample(samples)['reviewText']
        return reviews.to_numpy()

    

