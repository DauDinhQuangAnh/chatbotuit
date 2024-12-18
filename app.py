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
from constant import  VI,  USER, ASSISTANT, VIETNAMESE, ONLINE_LLM, GEMINI,  DB
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document as langchainDocument
from collection_management import list_collection

st.set_page_config(page_title="Page Title", layout="wide")
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)
def clear_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]
# Tiêu đề chính

st.markdown(
    """
    <h1 style='display: flex; align-items: center;'>
        <img src="https://tuyensinh.uit.edu.vn/sites/default/files/uploads/images/uit_footer.png" width="50" style='margin-right: 10px'>
        UIT Admissions Chatbot 🎓
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("Welcome to the UIT Admissions Chatbot❓❓❓ Discover all the information you need about admissions, 📚programs, 💸scholarships, 🌟Student Life at UIT and more with us.")

if "language" not in st.session_state:
    st.session_state.language = VIETNAMESE  
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "llm_model" not in st.session_state:
    st.session_state.llm_model = None
if "client" not in st.session_state:
    st.session_state.client = chromadb.PersistentClient("db")
if "collection" not in st.session_state:
    st.session_state.collection = None
if "search_option" not in st.session_state:
    st.session_state.search_option = "Hyde Search"
if "open_dialog" not in st.session_state:
    st.session_state.open_dialog = None
if "source_data" not in st.session_state:
    st.session_state.source_data = "UPLOAD"
if "chunks_df" not in st.session_state:
    st.session_state.chunks_df = pd.DataFrame()
if "random_collection_name" not in st.session_state:
    st.session_state.random_collection_name = None

st.session_state.chunkOption = "SemanticChunker" 

st.session_state.number_docs_retrieval = 10

if st.session_state.language != VI and st.session_state.embedding_model is None:
    st.session_state.embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
    st.session_state.embedding_model_name = 'keepitreal/vietnamese-sbert'
st.session_state.llm_type = ONLINE_LLM

# Thiết lập LLM
if st.session_state.llm_model is None:
    api_key = "AIzaSyAgOBMLyULtQE6PBI6u6v-bawhlF3UkhNI"
    st.session_state.llm_model = OnlineLLMs(
        name=GEMINI, api_key=api_key, model_version="gemini-1.5-pro")
    st.session_state.api_key_saved = True
    print("✅ API Key saved successfully!")
 
header_i = 1
st.header(f"{header_i}. Setup data source")
st.subheader(f"{header_i}.1. Upload data (Upload CSV files)", divider=True)
uploaded_files = st.file_uploader(
    "", 
    accept_multiple_files=True
)

st.session_state.data_saved_success = False

if uploaded_files is not None:
        all_data = []
        for uploaded_file in uploaded_files: 
            print(uploaded_file.type)
            if uploaded_file.name.endswith(".csv"):
                try:
                    df = pd.read_csv(uploaded_file)                                                                                                       
                    all_data.append(df)
                except pd.errors.ParserError:
                    raise ValueError(f"Error: The file {uploaded_file.name} is not in the correct format of a .csv file.")
            
if all_data:
    df = pd.concat(all_data, ignore_index=True) #noi dataframe
    value_df = pd.DataFrame([[df.iloc[1, 1]]], columns=[df.columns[1]])
    st.dataframe(value_df)
    st.subheader("Chunking")

    if not df.empty:
        index_column = "Câu trả lời"
        st.write(f"Selected column for indexing: {index_column}")
               
    chunkOption = st.session_state.get("chunkOption") 
    
    chunk_records = []

    for index, row in df.iterrows():
        chunker = None
        selected_column_value = row[index_column]
        chunks = []
        if not (type(selected_column_value) == str and len(selected_column_value) > 0):
            continue
        
        if chunkOption == "SemanticChunker":
            chunker = SemanticChunker(
                embedding_type="tfidf",
            )
        chunks = chunker.split_text(selected_column_value)
        
        # For each chunk, add a dictionary with the chunk and to the list
        for chunk in chunks:
            chunk_record = {**row.to_dict(), 'chunk': chunk}
            chunk_records.append(chunk_record)

    st.session_state.chunks_df = pd.DataFrame(chunk_records)

if "chunks_df" in st.session_state and len(st.session_state.chunks_df) > 0:
    st.write("Number of chunks:", len(st.session_state.chunks_df))
    st.dataframe(st.session_state.chunks_df)

if st.button("Save Data"):
    try:
        if st.session_state.collection is None:
            if uploaded_files:
                first_file_name = os.path.splitext(uploaded_files[0].name)[0]  
                collection_name = f"rag_collection_{clean_collection_name(first_file_name)}"
            else:
                collection_name = "rag_collection"

            st.session_state.random_collection_name = collection_name
            st.session_state.collection = st.session_state.client.get_or_create_collection(
                name=st.session_state.random_collection_name,
                metadata={"Chunk ": "",
                          "Question": "",
                          "Answer": ""},
            )

        batch_size = 256
        df_batches = divide_dataframe(st.session_state.chunks_df, batch_size)

        if not df_batches:
            st.warning("No data available to process.")
        else:
            num_batches = len(df_batches)

            progress_text = "Saving data to Chroma. Please wait..."
            my_bar = st.progress(0, text=progress_text)

            for i, batch_df in enumerate(df_batches):
                if batch_df.empty:
                    continue  
                
                process_batch(batch_df, st.session_state.embedding_model, st.session_state.collection)

                progress_percentage = int(((i + 1) / num_batches) * 100)
                my_bar.progress(progress_percentage, text=f"Processing batch {i + 1}/{num_batches}")

                time.sleep(0.1)  

            my_bar.empty()

            st.success("Data saved to Chroma vector store successfully!")
            st.markdown("Collection name: `{}`".format(st.session_state.random_collection_name))
            st.session_state.data_saved_success = True

    except Exception as e:
        st.error(f"Error saving data to Chroma: {str(e)}")

# Set up the interface
st.subheader(f"{header_i}.2. Or load from saved collection", divider=True)
if st.button("Load from saved collection"):
    st.session_state.open_dialog = "LIST_COLLECTION"
    def load_func(collection_name):
        st.session_state.collection = st.session_state.client.get_collection(
            name=collection_name
        )
        st.session_state.random_collection_name = collection_name
        st.session_state.data_saved_success = True
        st.session_state.source_data = DB
        data = st.session_state.collection.get(
            include=[
                "documents", 
                "metadatas"
            ],
        )
        metadatas = data["metadatas"]
        column_names = []
        if len(metadatas) > 0 and len(metadatas[0].keys()) > 0:
            column_names.extend(metadatas[0].keys())
            column_names = list(set(column_names))

        st.session_state.chunks_df = pd.DataFrame(metadatas, columns=column_names)

    def delete_func(collection_name):
        st.session_state.client.delete_collection(name=collection_name)
    
    list_collection(st.session_state, load_func, delete_func)
        
if "random_collection_name" in st.session_state and st.session_state.random_collection_name is not None and st.session_state.chunks_df is not None:
    # delete "chunk"
    columns_to_select = [col for col in st.session_state.chunks_df.columns if col != "chunk" ]
    st.session_state.columns_to_answer = columns_to_select

header_i += 1
header_text_llm = "{}. Set up search algorithms".format(header_i)
st.header(header_text_llm)

st.radio(
    "Please select one of the options below.",
    [
        # "Keywords Search", 
        "Hyde Search",
        "Vector Search"],
    captions = [
        # "Search using traditional keyword matching",
        "Search using the HYDE algorithm",
        "Search using vector similarity"
    ],
    key="search_option",
    index=0,
)

st.header("Interactive Chatbot")

# Initialize chat history in session state
if "chat_history" not in st.session_state:   #tao trang thai 'chat_history'
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#xu ly dau vao nguoi dung
if prompt := st.chat_input("How can I assist you today?"):  #neu co input ng dung
    # them vao chat history
    st.session_state.chat_history.append({"role": USER, "content": prompt})
    # Hien thi tin nhan cua ng dung
    with st.chat_message(USER):
        st.markdown(prompt)
 
    with st.chat_message(ASSISTANT):   #xu ly ben admin
        if st.session_state.collection is not None:
            # Combine retrieved data to enhance the prompt based on selected columns
            metadatas, retrieved_data = [], ""
            if st.session_state.columns_to_answer:
                if st.session_state.search_option == "Vector Search":
                    metadatas, retrieved_data = vector_search(
                        st.session_state.embedding_model, 
                        prompt, 
                        st.session_state.collection, 
                        st.session_state.columns_to_answer,
                        st.session_state.number_docs_retrieval
                    )
                    #retrieved_data,
                    enhanced_prompt = """
                    Câu hỏi của người dùng là: "{}". 
                    Bạn là một chatbot được thiết kế để trả lời các câu hỏi liên quan đến tuyển sinh tại UIT (Đại học Công nghệ Thông tin). 
                    Nếu người dùng chào hỏi, chỉ cần trả lời bằng một lời chào thân thiện và giới thiệu bạn là Chatbot của UIT. 
                    Nếu không, sử dụng dữ liệu đã được truy xuất dưới đây để trả lời câu hỏi của người dùng một cách thân thiện và hữu ích. 
                    Các câu trả lời của bạn phải chính xác, chi tiết và dựa trên dữ liệu đã được truy xuất: \n{}""".format(prompt, retrieved_data)

                elif st.session_state.search_option == "Hyde Search":
              
                    if st.session_state.llm_type == ONLINE_LLM:
                        model = st.session_state.llm_model
                    metadatas, retrieved_data = hyde_search(
                        model,
                        st.session_state.embedding_model,
                        prompt,
                        st.session_state.collection,
                        st.session_state.columns_to_answer,
                        st.session_state.number_docs_retrieval,
                        num_samples=1
                    )

                    enhanced_prompt = """
                    Câu hỏi của người dùng là: "{}". 
                    Bạn là một chatbot được thiết kế để trả lời các câu hỏi liên quan đến tuyển sinh tại UIT (Đại học Công nghệ Thông tin). 
                    Nếu người dùng chào hỏi, chỉ cần trả lời bằng một lời chào thân thiện và giới thiệu bạn là Chatbot của UIT. 
                    Nếu không, sử dụng dữ liệu đã được truy xuất dưới đây để trả lời câu hỏi của người dùng một cách thân thiện và hữu ích. 
                    Các câu trả lời của bạn phải chính xác, chi tiết và dựa trên dữ liệu đã được truy xuất: \n{}""".format(prompt, retrieved_data)
                
                if metadatas:
                    flattened_metadatas = [item for sublist in metadatas for item in sublist]  # Lam Phang du lieu vidu: [[1,2],[3,4]] se thanh [1,2,3,4]
                    
                    # Convert the flattened list of dictionaries to a DataFrame
                    metadata_df = pd.DataFrame(flattened_metadatas)
                    
                    st.sidebar.subheader("Retrieval data")
                    st.sidebar.dataframe(metadata_df) # hien thi ben thanh ung dung
                    st.sidebar.subheader("Full prompt for LLM")
                    st.sidebar.markdown(enhanced_prompt) # hien thi theo dang markdown
                else:
                    st.sidebar.write("No metadata to display.")

                if st.session_state.llm_type == ONLINE_LLM:
                    if "llm_model" in st.session_state and st.session_state.llm_model is not None:
                        response = st.session_state.llm_model.generate_content(enhanced_prompt)

                    st.markdown(response)

                    # Update chat history
                st.session_state.chat_history.append({"role": ASSISTANT, "content": response})
            else:
                st.warning("Please select a model to run.")
        else:
            st.warning("Please select columns for the chatbot to answer from.")
