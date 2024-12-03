import streamlit as st
from sentence_transformers import SentenceTransformer
import pandas as pd
from search import vector_search, hyde_search
from constant import USER, ASSISTANT, ONLINE_LLM
import chromadb
import json

# Hiển thị tiêu đề và giới thiệu của chatbot UIT
st.markdown(
    """
    <h1 style='display: flex; align-items: center;'>
        <img src="https://tuyensinh.uit.edu.vn/sites/default/files/uploads/images/uit_footer.png" width="50" style='margin-right: 10px'>
        UIT Admissions Chatbot 🎓
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("Welcome to the UIT Admissions Chatbot!❓❓❓ Discover all the information you need about admissions, 📚programs, 💸scholarships, 🌟Student Life at UIT and more with us.")

# Hàm tải trạng thái phiên làm việc từ một tệp JSON
def load_session_state(file_path="session_state.json"):
    try:
        with open(file_path, "r") as file:
            session_data = json.load(file)  # Đọc dữ liệu từ tệp JSON
        for key, value in session_data.items():
            st.session_state[key] = value  # Cập nhật trạng thái phiên làm việc với dữ liệu từ tệp
        # st.success("Session state loaded successfully!")
    except FileNotFoundError:
        st.error("Session state file not found.")  # Hiển thị lỗi nếu không tìm thấy tệp
    except json.JSONDecodeError:
        st.error("Error decoding session state file.")  # Hiển thị lỗi nếu có vấn đề khi đọc tệp JSON

# Thiết lập tùy chọn tìm kiếm mặc định nếu chưa được đặt trong session
if "search_option" not in st.session_state:
    st.session_state.search_option = "Vector Search"

# Hiển thị tên của tập dữ liệu (collection) nếu đã có sẵn
if "collection" not in st.session_state:
    load_session_state(file_path="pages/session_state.json")  # Tải trạng thái phiên làm việc từ tệp JSON

# Khởi tạo khách hàng Chroma nếu chưa có
if "client" not in st.session_state:
    st.session_state.client = chromadb.PersistentClient("db")  # Tạo kết nối với cơ sở dữ liệu Chroma
    if "random_collection_name" in st.session_state:
        st.session_state.collection = st.session_state.client.get_collection(
            st.session_state.random_collection_name  # Lấy collection dựa trên tên được lưu trong session
        )

# Khởi tạo mô hình embedding nếu chưa có
if "embedding_model_name" in st.session_state and "embedding_model" not in st.session_state:
    st.session_state.embedding_model = SentenceTransformer(
        st.session_state.embedding_model_name  # Tải mô hình SentenceTransformer dựa trên tên mô hình
    )

# Khởi tạo cột mặc định cho câu trả lời
if "columns_to_answer" not in st.session_state:
    st.session_state.columns_to_answer = ["chunk", "question", "answer"]

# Khởi tạo lịch sử hội thoại trong session nếu chưa có
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Hiển thị lịch sử hội thoại bằng giao diện trò chuyện
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])  # Hiển thị nội dung từng tin nhắn trong lịch sử hội thoại

# Xử lý và hiển thị câu hỏi của người dùng
if prompt := st.chat_input("What is up?"):
    # Thêm tin nhắn của người dùng vào lịch sử hội thoại
    st.session_state.chat_history.append({"role": USER, "content": prompt})

    # Hiển thị tin nhắn của người dùng trong giao diện trò chuyện
    with st.chat_message(USER):
        st.markdown(prompt)

    # Xử lý và cải thiện prompt của người dùng
    if st.session_state.collection is not None:
        metadatas, retrieved_data = [], ""
        if "columns_to_answer" in st.session_state:
            columns_to_answer = st.session_state.columns_to_answer
            search_option = st.session_state.search_option

            # Thực hiện tìm kiếm dựa trên tùy chọn tìm kiếm hiện tại
            if search_option == "Vector Search":
                metadatas, retrieved_data = vector_search(
                    st.session_state.embedding_model,
                    prompt,
                    st.session_state.collection,
                    columns_to_answer,
                    st.session_state.number_docs_retrieval
                )

            elif search_option == "Hyde Search":
                model = st.session_state.llm_model
                metadatas, retrieved_data = hyde_search(
                    model,
                    st.session_state.embedding_model,
                    prompt,
                    st.session_state.collection,
                    columns_to_answer,
                    st.session_state.number_docs_retrieval,
                    num_samples=1
                )

            # Tạo prompt nâng cao và sinh câu trả lời từ mô hình ngôn ngữ
            if metadatas:
                # Thêm ngữ cảnh chi tiết vào prompt nâng cao
                enhanced_prompt = (
                f'The user prompt is: "{prompt}". '
                "You are a chatbot designed to answer questions related to admissions at UIT (University of Information Technology). "
                "Please respond in a friendly and helpful manner, providing accurate and detailed information about admissions, "
                "scholarships, programs, and student life at UIT. Use the following retrieved data to craft your response:\n"
                f"{retrieved_data}")
                response = st.session_state.llm_model.generate_content(enhanced_prompt)

                # Cập nhật lịch sử hội thoại
                st.markdown(response)
                st.session_state.chat_history.append({"role": ASSISTANT, "content": response})
            else:
                st.warning("No data to enhance the prompt.")  # Cảnh báo nếu không có dữ liệu nào để hỗ trợ tạo câu trả lời
        else:
            st.warning("Select columns to answer from.")  # Cảnh báo nếu chưa chọn cột để trả lời
    else:
        st.error("No collection found. Upload data and save it first.")  # Hiển thị lỗi nếu chưa tải tập dữ liệu
