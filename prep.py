import json
import pandas as pd
import numpy as np

class Reader:
    def __init__(self, link_to_dataset):
        self.link_to_dataset = link_to_dataset

    def read(self):
        polls = []
        with open(self.link_to_dataset) as f:
            for line in f.readlines():
                polls.append(json.loads(line))

        df = pd.DataFrame.from_records(polls)
        return df
    
class Preprocessor:
    def __init__(self, df):
        self.df = df

    def preprocess_df(self, only_english = True, only_retweeted = False, positional = False):
        # df.drop(df[df['withheld_in_countries'] == True].index, inplace=True)
        for col in self.df.columns:
            try:
                if self.df[col].nunique() == 0 or self.df[col].nunique() == 1:
                    print(f"dropping {col}")
                    self.df.drop(columns = [col], inplace = True)
                    continue
            except:
                pass

            try:
                if self.df[col].notnull().sum() == 0:
                    print(f"dropping {col}")
                    self.df.drop(columns = [col], inplace = True)
                    continue
            except:
                pass

        self.df['RT'] = self.df['text'].str.startswith('RT', na = False)
        if only_retweeted:
            self.df = self.df[self.df['RT'] == True]

        if only_english:
            self.df = self.df[self.df['lang'] == 'en']

        if positional:
            self.df['positional'] = self.df['text'].astype(str) + self.df['timestamp_ms'].astype(str)


    def preprocess_text(self):
        # TODO 
        self.df = self.df.drop_duplicates(subset='text', keep="first")
        return

    def display_info(self):
        print()
        print(self.df.head(5))
        print(f"\nShape: {self.df.shape}")


    def get_dataframe(self):
        self.df.to_csv('./processed_polls.csv')
        self.display_info()
        return self.df  
    
class SentenceTransformerEmbedding:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def embed(self, texts):
        texts = np.array(texts)
        self.embeddings = self.embedder.encode(texts)

        return self.embeddings
