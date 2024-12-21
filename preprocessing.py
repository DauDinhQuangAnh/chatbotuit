# preprocessing.py
import re
import pandas as pd
import unicodedata

def remove_special_characters(text):
    """Loại bỏ các ký tự đặc biệt, HTML tags, và URLs."""
    text = re.sub(r'[!@#$%^&*(),.?":{}|<>]', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    return text

def standardize_unicode(text):
    """Chuẩn hoá unicode về chuẩn nfc"""
    text = unicodedata.normalize('NFC', text)
    return text

def lowercase(text):
    """Chuyển đổi tất cả các ký tự thành chữ thường."""
    return text.lower()

def remove_extra_whitespaces(text):
    """Loại bỏ các khoảng trắng thừa."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_duplicate_rows(df, column_name):
    """Loại bỏ các dòng trùng lặp trong DataFrame dựa trên cột được chỉ định."""
    df.drop_duplicates(subset=column_name, keep='first', inplace=True)
    return df

def preprocess_text(text):
    """Hàm tiền xử lý tổng hợp cho một chuỗi văn bản."""
    text = standardize_unicode(text)
    text = lowercase(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespaces(text)
    return text