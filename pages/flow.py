import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import streamlit as st

# Header setup
header_i = 1
st.header(f"{header_i}. Semantic chunk")
st.subheader(f"{header_i}.1. Sử dụng Natural Language Toolkit để chia data thành các câu theo dấu ., ?, !, ...", divider=True)

# Text input
text_input = st.text_area("Nhập văn bản của bạn vào đây:", height=100)
nltk.download('punkt')  # Tải gói dữ liệu punkt (nếu chưa tải)
if text_input:
    # Tách văn bản thành các câu
    sentences = sent_tokenize(text_input)
    # Hiển thị các câu đã tách
    st.subheader("Các câu đã tách:")
    for sentence in sentences:
        st.write(sentence)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Chuyển ma trận TF-IDF thành DataFrame để dễ dàng hiển thị
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Hiển thị ma trận TF-IDF dưới dạng bảng
    st.subheader(f"{header_i}.2. Vector hóa cho từng câu sử dụng TF-IDF ", divider=True)

    st.subheader("Ma trận TF-IDF:")
    st.write(tfidf_df)

    # Tính độ tương đồng cosine giữa các câu
    st.subheader(f"{header_i}.3. Tính toán độ tương đồng giữa các câu", divider=True)
    similarities = cosine_similarity(tfidf_matrix)

    # Hiển thị ma trận độ tương đồng
    similarity_df = pd.DataFrame(similarities, index=[f"Câu {i+1}" for i in range(len(sentences))],
                                 columns=[f"Câu {i+1}" for i in range(len(sentences))])
    st.write("Ma trận độ tương đồng (cosine similarity):")
    st.write(similarity_df)

    # Ghép các câu thành đoạn dựa trên ngưỡng similarity
    st.subheader(f"{header_i}.4. Ghép các câu thành đoạn dựa trên độ tương đồng", divider=True)
    st.write(f"Độ tương đồng tối thiểu = 0.3", divider=True)
    threshold = 0.3

    # Nhóm các câu thành đoạn
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim_score = similarities[i - 1, i]  # Similarity giữa câu hiện tại và câu trước đó
        if sim_score >= threshold:
            # Nếu similarity vượt ngưỡng, thêm câu vào đoạn hiện tại
            chunks[-1].append(sentences[i])
        else:
            # Nếu không, tạo đoạn mới
            chunks.append([sentences[i]])

    # Hiển thị các đoạn văn đã ghép
    grouped_paragraphs = [' '.join(chunk) for chunk in chunks]
    for idx, paragraph in enumerate(grouped_paragraphs):
        st.write(f"**Đoạn {idx + 1}:** {paragraph}")

    # st.session_state.embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')

    header_i += 1
    st.header(f"{header_i}. Embedding model sử dụng Sentence-BERT")
    st.subheader("Ma trận embeddings của các đoạn văn:", divider=True)
    # Bây giờ vector hóa các đoạn văn (semantic chunks)
    embeddings = st.session_state.embedding_model.encode(grouped_paragraphs)

    # Hiển thị các vector (embeddings) của các đoạn văn
    embeddings_df = pd.DataFrame(embeddings)

    # Chuyển các embeddings thành DataFrame để dễ dàng hiển thị
    st.write(embeddings_df)
else:
    st.info("Vui lòng nhập văn bản để tách thành các câu.")
