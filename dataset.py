import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import config
from tqdm import tqdm


class ToxicCommentsDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
        self.data = pd.read_csv(csv_path)
        #self.data = self.data.head(100)

        self.tokenizer = tokenizer
        self.max_len = max_len

        self.comments_df = self.data[['id', 'comment_text']]
        self.labels_df = self.data.drop(['id', 'comment_text'], axis=1)
        #self.comments_df = self.data["comment_text"].fillna("no comment")
        #self.labels_df = self.data[config.LABEL_COLUMNS].values.astype(float)


        self.tokenized_data = []
        for comment in tqdm(self.comments_df["comment_text"], desc="Tokenizing test comments", total=len(self.comments_df)):
            encoding = self.tokenizer.encode_plus(
                comment,
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            self.tokenized_data.append(encoding)

    def __len__(self):
        return len(self.comments_df)

    def __getitem__(self, item):
        encoding = self.tokenized_data[item]
        labels = self.labels_df.iloc[item].values.astype(float)  # Ensure labels are in the correct format

        # Return the processed inputs and labels
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(labels, dtype=torch.float)
        }
 



class ToxicCommentsWithLabelsDataset(Dataset):
    def __init__(self, comments_csv, labels_csv, tokenizer, max_len):
        # Load the CSV files
        self.comments_df = pd.read_csv(comments_csv)
        self.labels_df = pd.read_csv(labels_csv)

        # Preprocess labels: Convert -1 to 0
        self.labels_df.replace(-1, 0, inplace=True)  # Convert all -1s to 0s

        # Merge comments and labels on 'id' field
        merged_df = pd.merge(self.comments_df, self.labels_df, on='id')

        # Take the first 100 data points
        #merged_df = merged_df.head(500)

        # Re-assign to the original DataFrames
        self.comments_df = merged_df[['id', 'comment_text']]
        self.labels_df = merged_df.drop(['id', 'comment_text'], axis=1)

        # Initialize tokenizer and max_len
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Tokenize the dataset and store the results
        self.tokenized_data = []
        
        # Add tqdm progress bar for tokenizing
        for comment in tqdm(self.comments_df["comment_text"], desc="Tokenizing test comments", total=len(self.comments_df)):
            encoding = self.tokenizer.encode_plus(
                comment,
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            self.tokenized_data.append(encoding)

    def __len__(self):
        return len(self.comments_df)

    def __getitem__(self, item):
        # Get the tokenized data and labels
        encoding = self.tokenized_data[item]
        labels = self.labels_df.iloc[item].values.astype(float)  # Ensure labels are in the correct format

        # Return the processed inputs and labels
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(labels, dtype=torch.float)
        }