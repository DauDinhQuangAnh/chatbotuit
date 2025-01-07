import chromadb
import bcrypt

# Kết nối tới ChromaDB và tạo collection
def get_chroma_client():
    client = chromadb.PersistentClient(path="db_auth/auth_db")
    return client

# Tạo hoặc lấy collection người dùng (sửa lại cho đồng nhất)
def get_user_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(name="user_authen")

# Đăng ký người dùng mới
def register_user(username, password):
    collection = get_user_collection()
    
    # Kiểm tra xem user đã tồn tại hay chưa
    results = collection.query(query_texts=[username], n_results=1)
    if results['ids'] and len(results['ids'][0]) > 0:
        return False  # Tài khoản đã tồn tại

    # Mã hóa mật khẩu với bcrypt
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Thêm người dùng vào collection ChromaDB
    collection.add(
        ids=[username],
        documents=[username],  # Thêm documents để tránh lỗi thiếu thông tin
        metadatas=[{"username": username, "password": hashed_pw.decode('utf-8')}]
    )
    return True

# Xác thực đăng nhập
def authenticate_user(username, password):
    collection = get_user_collection()
    results = collection.query(query_texts=[username], n_results=1)

    # Kiểm tra nếu kết quả trả về rỗng
    if not results['ids'] or len(results['ids'][0]) == 0:
        return False  # Không tìm thấy tài khoản

    # Lấy mật khẩu đã lưu trữ và kiểm tra với bcrypt
    stored_pw = results['metadatas'][0][0]['password']
    return bcrypt.checkpw(password.encode('utf-8'), stored_pw.encode('utf-8'))
