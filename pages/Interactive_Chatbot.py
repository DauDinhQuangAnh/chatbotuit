import streamlit as st
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from IPython.display import Markdown
from chunking import SemanticChunker
from utils import process_batch, divide_dataframe, clean_collection_name
from search import vector_search, hyde_search
from llms.onlinellms import OnlineLLMs
import time
from constant import VI, USER, ASSISTANT, VIETNAMESE, ONLINE_LLM, GEMINI, DB
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document as langchainDocument
from collection_management import list_collection
from preprocessing import preprocess_text, remove_duplicate_rows  # Import các hàm cần thiết
# --- Streamlit Configuration ---
st.set_page_config(page_title="UIT Admissions Chatbot", layout="wide" , page_icon="https://tuyensinh.uit.edu.vn/sites/default/files/uploads/images/uit_footer.png")
st.markdown("""
    <style>
        .reportview-container { margin-top: -2em; }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def clear_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]
st.sidebar.header("Đánh giá Chatbot")
review = st.sidebar.text_area("Viết đánh giá của bạn tại đây:")

if st.sidebar.button("Lưu đánh giá"):
    if review.strip():
        with open("user_reviews.txt", "a", encoding="utf-8") as review_file:
            review_file.write(f" Rview of user: {review}\n")
        st.sidebar.success("Đánh giá của bạn đã được lưu!")
    else:
        st.sidebar.error("Vui lòng nhập nội dung đánh giá trước khi lưu!")
# --- UI Setup ---
st.sidebar.title("Doc Retrievaled")
st.markdown(
    """
    <h1 style='display: flex; align-items: center;'>
        <img src="https://tuyensinh.uit.edu.vn/sites/default/files/uploads/images/uit_footer.png" width="50" style='margin-right: 10px'>
        UIT Admissions Chatbot 
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("Welcome to the UIT Admissions Chatbot!❓❓❓ Discover all the information you need about admissions, 📚programs, 💸scholarships, 🌟Student Life at UIT and more with us.")

if "language" not in st.session_state:
    st.session_state.language = VIETNAMESE

if "embedding_model" not in st.session_state:
    if st.session_state.language == VIETNAMESE:
        st.session_state.embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
        st.session_state.embedding_model_name = 'keepitreal/vietnamese-sbert'
        st.success("Using Vietnamese embedding model: keepitreal/vietnamese-sbert")
    else:
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        st.session_state.embedding_model_name = 'all-MiniLM-L6-v2'
        st.success("Using Vietnamese embedding model: all-MiniLM-L6-v2")

# Initialize llm_model
if "llm_model" not in st.session_state:
    api_key = "AIzaSyBwvb8_AXyn_IZwn92WnqWQpKKM9sMfszQ"  # Replace with your actual API key
    st.session_state.llm_model = OnlineLLMs(name=GEMINI, api_key=api_key, model_version="learnlm-1.5-pro-experimental")
    st.session_state.api_key_saved = True
    print(" API Key saved successfully!")

# Initialize other session state variables
if "client" not in st.session_state:
    st.session_state.client = chromadb.PersistentClient("db")
if "search_option" not in st.session_state:
    st.session_state.search_option = "Vector Search"
if "open_dialog" not in st.session_state:
    st.session_state.open_dialog = None
if "source_data" not in st.session_state:
    st.session_state.source_data = "UPLOAD"
if "chunks_df" not in st.session_state:
    st.session_state.chunks_df = pd.DataFrame()
if "random_collection_name" not in st.session_state:
    st.session_state.random_collection_name = None

st.session_state.chunkOption = "SemanticChunker"
st.session_state.number_docs_retrieval = 15
st.session_state.llm_type = ONLINE_LLM
if "random_collection_name" in st.session_state and st.session_state.random_collection_name and not st.session_state.chunks_df.empty:
    st.session_state.columns_to_answer = [col for col in st.session_state.chunks_df.columns if col != "chunk"]
# Tự động tải collection khi trang được load
if "collection" not in st.session_state:  # Kiểm tra nếu collection chưa được load
    def load_func(collection_name):
        st.session_state.collection = st.session_state.client.get_collection(name=collection_name)
        st.session_state.random_collection_name = collection_name
        st.session_state.data_saved_success = True
        st.session_state.source_data = DB
        data = st.session_state.collection.get(include=["documents", "metadatas"])
        metadatas = data["metadatas"]
        column_names = []
        if metadatas and metadatas[0].keys():
            column_names.extend(metadatas[0].keys())
            column_names = list(set(column_names))
        st.session_state.chunks_df = pd.DataFrame(metadatas, columns=column_names)

    # Gọi load_func khi trang được load
    load_func("rag_collection_data-2")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I assist you today?"):
    # Append the prompt to the chat history
    st.session_state.chat_history.append({"role": USER, "content": prompt})
    with st.chat_message(USER):
        st.markdown(prompt)

    # Save the user's question to a text file
    with open("user_questions.txt", "a", encoding="utf-8") as file:
        file.write(f"Question user: {prompt}\n")

    with st.chat_message(ASSISTANT):
        if st.session_state.collection:
            metadatas, retrieved_data = [], ""
            if st.session_state.columns_to_answer:
                search_func = hyde_search if st.session_state.search_option == "Hyde Search" else vector_search
                model = st.session_state.llm_model if st.session_state.llm_type == ONLINE_LLM else None

                if st.session_state.search_option == "Vector Search":
                    metadatas, retrieved_data = vector_search(
                        st.session_state.embedding_model,
                        prompt,
                        st.session_state.collection,
                        st.session_state.columns_to_answer,
                        st.session_state.number_docs_retrieval
                    )

                    enhanced_prompt = """
                    Nêu câu trả lời sau không liên quan đến tuyển Sinh trường Đại Học Công Nghệ Thông Tin Đại Học Quốc GIa Thành Phố HCM (UIT) thì lich sự từ chối trả lời, 
                    Câu hỏi của người dùng là: "{}".Trả lời nó dựa trên dữ liệu được truy xuất sau đây: \n{} """.format(prompt, retrieved_data)

                elif st.session_state.search_option == "Hyde Search":
                    if st.session_state.llm_type == ONLINE_LLM:
                        metadatas, retrieved_data = search_func(
                            model,
                            st.session_state.embedding_model,
                            prompt,
                            st.session_state.collection,
                            st.session_state.columns_to_answer,
                            st.session_state.number_docs_retrieval,
                            num_samples=1 if st.session_state.search_option == "Hyde Search" else None
                        )

                    enhanced_prompt = """
                    Nêu câu trả lời sau không liên quan đến tuyển Sinh trường Đại Học Công Nghệ Thông Tin Đại Học Quốc GIa Thành Phố HCM (UIT) thì lich sự từ chối trả lời, 
                    Câu hỏi của người dùng là: "{}".Trả lời nó dựa trên dữ liệu được truy xuất sau đây: \n{} """.format(prompt, retrieved_data)

                if metadatas:
                    flattened_metadatas = [item for sublist in metadatas for item in sublist]
                    metadata_df = pd.DataFrame(flattened_metadatas)
                    st.sidebar.subheader("Retrieval data")
                    st.sidebar.dataframe(metadata_df)
                    st.sidebar.subheader("Full prompt for LLM")
                    st.sidebar.markdown(enhanced_prompt)
                else:
                    st.sidebar.write("No metadata to display.")

                if st.session_state.llm_model:
                    response = st.session_state.llm_model.generate_content(enhanced_prompt)
                    with open("user_questions.txt", "a", encoding="utf-8") as file:
                        file.write(f"Answer of Chatbot: {response}\n")
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": ASSISTANT, "content": response})
            else:
                st.warning("Please select a model to run.")
        else:
            st.warning("Please select columns for the chatbot to answer from.")


