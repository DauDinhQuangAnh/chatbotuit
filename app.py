import streamlit as st
from db_auth.chroma_auth import register_user, authenticate_user

# Cấu hình Streamlit
st.set_page_config(page_title="UIT Admissions Chatbot", layout="wide" , page_icon="https://tuyensinh.uit.edu.vn/sites/default/files/uploads/images/uit_footer.png")

# CSS để ẩn header và footer + Tùy chỉnh nút trong sidebar
st.markdown("""
    <style>
        header {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stSidebarNav"] {display: none;}
        
        /* Căn chỉnh và tùy chỉnh nút trong sidebar */
        .sidebar-button {
            padding: 10px;
            width: 100%;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            color: white;
            cursor: pointer;
            text-align: center;
            margin: 10px 0;
        }

        .admin-btn { background-color: #3498db; }
        .admin-btn:hover { background-color: #2980b9; }

        .register-btn { background-color: #2ecc71; }
        .register-btn:hover { background-color: #27ae60; }

        .chatbot-btn { background-color: #f39c12; }
        .chatbot-btn:hover { background-color: #e67e22; }

    </style>
    """, unsafe_allow_html=True)
# Khởi tạo trạng thái xác thực và giao diện
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "page" not in st.session_state:
    st.session_state.page = "login"

def switch_to_admin():
    st.switch_page("pages/admin.py")

def switch_to_user():
    st.switch_page("pages/user_chatbot.py")

# Giao diện đăng nhập
def login():
    st.title("🔑 Đăng Nhập Admin")
    username = st.text_input("Tên đăng nhập", key="login_username")
    password = st.text_input("Mật khẩu", type="password", key="login_password")

    if st.button("Đăng nhập"):
        if authenticate_user(username, password):
            st.session_state.authenticated = True
            st.success("✅ Đăng nhập thành công!")
            switch_to_admin()
        else:
            st.error("❌ Tài khoản hoặc mật khẩu không chính xác!")

# Giao diện đăng ký
def register():
    st.title("📝 Đăng Ký Tài Khoản")
    new_username = st.text_input("Tên đăng nhập mới", key="register_username")
    new_password = st.text_input("Mật khẩu mới", type="password", key="register_password")

    if st.button("Đăng ký"):
        if register_user(new_username, new_password):
            st.success("🎉 Tài khoản đã được tạo thành công!")
        else:
            st.error("Tên đăng nhập đã tồn tại!")

# Giao diện Chatbot
def Chatbot():
    switch_to_user()

# --- Tạo Giao Diện Chính với Các Nút Trong Sidebar ---
st.sidebar.title("🚀 UIT Admissions Chatbot")

# Điều hướng bằng session_state
if st.sidebar.button("📊 Admin Đăng Nhập"):
    st.session_state.page = "login"

if st.sidebar.button("📝 Đăng Ký"):
    st.session_state.page = "register"

if st.sidebar.button("💬 ChatBot UIT (Sinh Vien)"):
    st.session_state.page = "chatbot"

# Xử lý hiển thị giao diện dựa trên trạng thái
if st.session_state.page == "login":
    login()
elif st.session_state.page == "register":
    register()
elif st.session_state.page == "chatbot":
    Chatbot()
elif st.session_state.page == "admin" and st.session_state.authenticated:
    switch_to_admin()
