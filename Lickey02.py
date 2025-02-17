import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai


GOOGLE_API_KEY = 'AIzaSyDPEAfdPByyoDpaB1s5oRfb388qIQUQmKY'
genai.configure(api_key=GOOGLE_API_KEY)

model_gemini = genai.GenerativeModel('gemini-pro')

# Load dữ liệu CSV
df = pd.read_csv('m02.csv')

# Chọn các cột chứa thông tin sinh viên quan trọng
student_info_columns = ['họ tên', 'quê quán', 'tài khoản facebook', 'mã sinh viên']
df['combined_info'] = df[student_info_columns].apply(lambda row: ' '.join(row.astype(str)), axis=1)

# Khởi tạo Sentence Transformer
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)
if torch.cuda.is_available():
    model = model.to('cuda')

# Tạo embeddings cho thông tin sinh viên
student_info_embeddings = model.encode(df['combined_info'].tolist(), convert_to_tensor=True)

# Hàm xác định xem câu hỏi có liên quan đến sinh viên hay không
def is_student_related(query):
    # Các từ khóa liên quan đến sinh viên
    student_keywords = ['sinh viên', 'mã số', 'họ tên', 'quê quán', 'facebook', 'lớp', 'khóa', 'tài khoản']  # Thêm 'tài khoản'
    query = query.lower() # Chuyển query về chữ thường để so sánh không phân biệt hoa thường

    # Kiểm tra xem câu hỏi có chứa bất kỳ từ khóa nào không
    for keyword in student_keywords:
        if keyword in query:
            return True

    # Nếu không có từ khóa nào, sử dụng Sentence Transformer để đánh giá ngữ nghĩa
    # Tạo embedding cho câu hỏi
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Tính độ tương đồng với tất cả các thông tin sinh viên
    similarity_scores = util.cos_sim(query_embedding, student_info_embeddings)[0]

    # Nếu độ tương đồng cao nhất vượt quá ngưỡng, coi như là liên quan đến sinh viên
    threshold = 0.5  # Giảm ngưỡng xuống để dễ nhận diện hơn
    if torch.max(similarity_scores) > threshold:
        return True

    return False

# Hàm trả lời câu hỏi
def get_response(query, similarity_threshold=0.4):
    if is_student_related(query):
        # Tìm kiếm thông tin sinh viên trong CSV
        query = query.lower()  # Chuyển query về chữ thường
        query_embedding = model.encode(query, convert_to_tensor=True)
        similarity_scores = util.cos_sim(query_embedding, student_info_embeddings)[0]
        best_match_index = torch.argmax(similarity_scores).item()
        best_match_score = torch.max(similarity_scores).item()

        print(f"Độ tương đồng cao nhất: {best_match_score}")  # In ra độ tương đồng

        if best_match_score > similarity_threshold:
            # Trả lời dựa trên thông tin từ CSV
            student_data = df.iloc[best_match_index]

            # Xác định giá trị cần tìm
            if "mã số" in query or "mã sinh viên" in query:
                return f"Mã số sinh viên: {student_data['mã sinh viên']}"
            elif "quê quán" in query or "xuất thân" in query:
                return f"Quê quán: {student_data['quê quán']}"
            elif "facebook" in query or "tài khoản facebook" in query:
                return f"Tài khoản Facebook: {student_data['tài khoản facebook']}"
            elif "thông tin" in query:
                return f"Thông tin của sinh viên{student_data['họ tên']}:\nMã sinh viên là: {student_data['mã sinh viên']}\nQuê quán: {student_data['quê quán']}"
            else:
                return f"Họ tên: {student_data['họ tên']}" # Mặc định trả về họ tên
        else:
            return "Không tìm thấy thông tin phù hợp trong dữ liệu sinh viên."
    else:
        try:
            response = model_gemini.generate_content(query)
            return response.text        
        except Exception as e:
            return f"Lỗi khi sử dụng Gemini: {e}"

# Triển khai LICKEY

print("Chào mừng bạn đã đến chương trình AI LICKEY do LIC phát triển (nhấn thoat để dừng chương trình)")
while True:
    user_input = input("Bạn: ")
    if user_input.lower() == 'thoat' or user_input.lower() == "thoát":
        print("Tạm biệt")
        exit()
        
    response = get_response(user_input)
    print(f"Lickey: {response}")