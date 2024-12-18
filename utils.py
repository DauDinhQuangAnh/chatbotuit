import math
import uuid
import tiktoken
import streamlit as st
import re

def process_batch(batch_df, model, collection):
    try:
        embeddings = model.encode(batch_df['chunk'].tolist())
        metadatas = [row.to_dict() for _, row in batch_df.iterrows()]
        batch_ids = [str(uuid.uuid4()) for _ in range(len(batch_df))]

        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
    except Exception as e:
        if str(e) == "'NoneType' object has no attribute 'encode'":
            raise RuntimeError("model error.")
        raise RuntimeError(f"Err save in Chroma in batch: {str(e)}")

def divide_dataframe(df, batch_size):
    """Chia DataFrame thành các phần nhỏ dựa trên kích thước batch."""
    num_batches = math.ceil(len(df) / batch_size)  # Tính số lượng batch
    return [df.iloc[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

def clean_collection_name(name):
    # Chỉ cho phép các ký tự chữ và số, gạch dưới, dấu gạch ngang, và một dấu chấm giữa
    cleaned_name = re.sub(r'[^a-zA-Z0-9_.-]', '', name)   # Loại bỏ các ký tự không hợp lệ
    cleaned_name = re.sub(r'\.{2,}', '.', cleaned_name)    # Loại bỏ các dấu chấm liên tiếp
    cleaned_name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', cleaned_name)  # Loại bỏ ký tự không hợp lệ ở đầu/cuối
    return cleaned_name[:63] if 3 <= len(cleaned_name) <= 63 else None