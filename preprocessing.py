import re
import unicodedata
import string

import re

def remove_special_characters(text):
    text = re.sub(r'<.*?>', ' ', text)  
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text) 
    text = re.sub(r'[^\w\s.!?@]', ' ', text)  
    return text


def lowercase(text):
    return text.lower()

def remove_extra_whitespaces(text):
    text = text.strip()  
    text = re.sub(r'\s+', ' ', text)  
    return text


def preprocess_text(text):
    text = lowercase(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespaces(text)
    return text

def remove_duplicate_rows(df, column_name):
    df.drop_duplicates(subset=column_name, keep='first', inplace=True)
    return df
