import streamlit as st
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from IPython.display import Markdown
from chunking import SemanticChunker
from utils import process_batch, divide_dataframe
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
        header {visibility: hidden;}
        footer {visibility: hidden;}
        /* Giảm khoảng trắng phía trên */
        .block-container {
            padding-top: 1rem;
        }
        [data-testid="stSidebarNav"] {display: none;}

    </style>
    """, unsafe_allow_html=True)
# Nút đăng xuất
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.success("✅ Đã đăng xuất thành công!")
    st.switch_page("app.py")  # Quay lại trang đăng nhập
st.sidebar.title("Doc Retrievaled")

if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.switch_page("app.py")  # Chuyển hướng về trang đăng nhập

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

# --- Session State Initialization ---
if "language" not in st.session_state:
    st.session_state.language = VIETNAMESE

language_choice = st.radio(
    "Select Model:", [
        "vietnamese-sbert",
        "all-MiniLM-L6-v2", 
        
    ],
    index=0
    )

# Switch embedding model based on language choice
if language_choice == "all-MiniLM-L6-v2":
    st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.session_state.embedding_model_name = 'all-MiniLM-L6-v2'
    st.success("Using Vietnamese embedding model: all-MiniLM-L6-v2")
elif language_choice == "vietnamese-sbert":
    st.session_state.embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
    st.session_state.embedding_model_name = 'keepitreal/vietnamese-sbert'
    st.success("Using Vietnamese embedding model: keepitreal/vietnamese-sbert")

# Initialize llm_model
if "llm_model" not in st.session_state:
    api_key = "AIzaSyBwvb8_AXyn_IZwn92WnqWQpKKM9sMfszQ"  # Replace with your actual API key
    st.session_state.llm_model = OnlineLLMs(name=GEMINI, api_key=api_key, model_version="learnlm-1.5-pro-experimental")#gemini-1.5-pro
    st.session_state.api_key_saved = True
    print(" API Key saved successfully!")

# Initialize other session state variables
if "client" not in st.session_state:
    st.session_state.client = chromadb.PersistentClient("db")
if "collection" not in st.session_state:
    st.session_state.collection = None
if "search_option" not in st.session_state:
    st.session_state.search_option = None
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

# --- Data Upload and Processing ---
st.header("1. Setup data source")
st.subheader("1.1. Upload data (Upload CSV files)", divider=True)
uploaded_files = st.file_uploader("", accept_multiple_files=True)
all_data = []
if uploaded_files:
    for file in uploaded_files:
        if file.name.endswith(".csv"):
            try:
                df = pd.read_csv(file)
                all_data.append(df)
            except pd.errors.ParserError:
                st.error(f"Error: The file {file.name} is not a valid .csv file.")
    df = pd.concat(all_data, ignore_index=True)
    # --- Preprocessing ---
    if not df.empty:
        df["Câu trả lời"] = df["Câu trả lời"].apply(preprocess_text)
        df = remove_duplicate_rows(df, "Câu hỏi")
        # Loại bỏ các dòng trùng lặp dựa trên cột "Câu trả lời"
        df = remove_duplicate_rows(df, "Câu trả lời")
        st.session_state.df = df
        st.dataframe(df)

    st.subheader("Chunking")
    if not df.empty:
        index_column = "Câu trả lời"
        st.write(f"Selected column for indexing: {index_column}")

        chunk_records = []
        for _, row in df.iterrows():
            selected_column_value = row[index_column]
            if isinstance(selected_column_value, str) and selected_column_value:
                chunker = SemanticChunker(embedding_type="tfidf")
                chunks = chunker.split_text(selected_column_value)
                for chunk in chunks:
                    chunk_records.append({**row.to_dict(), 'chunk': chunk})

        st.session_state.chunks_df = pd.DataFrame(chunk_records)

if "chunks_df" in st.session_state and not st.session_state.chunks_df.empty:
    st.write("Number of chunks:", len(st.session_state.chunks_df))
    st.dataframe(st.session_state.chunks_df)

# --- Data Saving ---
if st.button("Save Data"):
    if st.session_state.chunks_df.empty:
        st.warning("No data available to process.")
    else:
        try:
            if st.session_state.collection is None:
                collection_name = "rag_collection-DataOfUIT"   
                st.session_state.random_collection_name = collection_name
                st.session_state.collection = st.session_state.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"Chunk ": "", "Question": "", "Answer": ""}
                )
            batch_size = 256
            df_batches = divide_dataframe(st.session_state.chunks_df, batch_size)
            num_batches = len(df_batches)

            progress_text = "Saving data to Chroma. Please wait..."
            my_bar = st.progress(0, text=progress_text)

            for i, batch_df in enumerate(df_batches):
                if not batch_df.empty:
                    process_batch(batch_df, st.session_state.embedding_model, st.session_state.collection)
                    progress_percentage = int(((i + 1) / num_batches) * 100)
                    my_bar.progress(progress_percentage, text=f"Processing batch {i + 1}/{num_batches}")
                    time.sleep(0.1)
            my_bar.empty()
            st.success("Data saved to Chroma vector store successfully!")
            st.markdown(f"Collection name: {st.session_state.random_collection_name}")
            st.session_state.data_saved_success = True
        except Exception as e:
            st.error(f"Error saving data to Chroma: {e}")

# --- Load from Saved Collection ---
st.subheader("1.2. Or load from saved collection", divider=True)
if st.button("Load from saved collection"):
    st.session_state.open_dialog = "LIST_COLLECTION"

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

    def delete_func(collection_name):
        st.session_state.client.delete_collection(name=collection_name)

    list_collection(st.session_state, load_func, delete_func)

if "random_collection_name" in st.session_state and st.session_state.random_collection_name and not st.session_state.chunks_df.empty:
    st.session_state.columns_to_answer = [col for col in st.session_state.chunks_df.columns if col != "chunk"]

# --- Search Algorithm Setup ---
st.header("2. Set up search algorithms")
selected_option = st.radio(
    "Please select one of the options below.",
    ["Vector Search", "Hyde Search"],
)
st.session_state.search_option = selected_option
# --- Chat Interface ---
st.header("Interactive Chatbot")

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
                    Bạn đang đóng vai một chuyên gia tư vấn tuyển sinh của Trường Đại học Công nghệ Thông tin (UIT) thuộc Đại học Quốc gia TP.HCM. 
                    Khi nhận được câu hỏi, hãy bắt đầu bằng lời chào thân thiện và giới thiệu ngắn gọn về vai trò của bạn. 
                    Nếu câu hỏi không liên quan đến tuyển sinh của UIT, hãy lịch sự từ chối và giải thích rằng bạn chỉ hỗ trợ các câu hỏi liên quan đến tuyển sinh UIT. 
                    Dựa trên dữ liệu đã truy xuất bên dưới, hãy trả lời theo dạng liệt kê một cách chính xác, ngắn gọn và thân thiện. 
                    Câu hỏi của người dùng là: "{}"
                    Dữ liệu được cung cấp để trả lời là: \n{} """.format(prompt, retrieved_data)
                    
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
                    Bạn đang đóng vai một chuyên gia tư vấn tuyển sinh của Trường Đại học Công nghệ Thông tin (UIT) thuộc Đại học Quốc gia TP.HCM. 
                    Khi nhận được câu hỏi, hãy bắt đầu bằng lời chào thân thiện và giới thiệu ngắn gọn về vai trò của bạn. 
                    Nếu câu hỏi không liên quan đến tuyển sinh của UIT, hãy lịch sự từ chối và giải thích rằng bạn chỉ hỗ trợ các câu hỏi liên quan đến tuyển sinh UIT. 
                    Dựa trên dữ liệu đã truy xuất bên dưới, hãy trả lời theo dạng liệt kê một cách chính xác, ngắn gọn và thân thiện. 
                    Câu hỏi của người dùng là: "{}"
                    Dữ liệu được cung cấp để trả lời là: \n{} """.format(prompt, retrieved_data)

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


