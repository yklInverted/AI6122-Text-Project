import nltk
import pandas as pd
import sys
import os

class DataLoader():

    data_path1 = "./data/Kindle_Store_5.json"
    data_path2 = "./data/Digital_Music_5.json"

    def __init__(self, data_path = data_path1) -> None:
        
        self.table = pd.read_json(data_path, 'records', lines = True);


    def load_table(self, samples = 200):

        samples = self.table.sample(samples, random_state = 0)
        return samples.reset_index(drop = True, inplace = False)

    def load_review_text(self, samples = 200):

        reviews = self.table.sample(samples)['reviewText']
        return reviews.to_numpy()

    

