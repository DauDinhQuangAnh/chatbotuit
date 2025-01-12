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
