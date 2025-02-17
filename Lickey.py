import google.generativeai as genai
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy

# Cấu hình gemini ai
GOOGLE_API_KEY = 'AIzaSyDPEAfdPByyoDpaB1s5oRfb388qIQUQmKY'
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel(model_name='gemini-pro')
QUESTION_COLUM= 'question'
ANSWER_COLUMN= 'answer'




def load_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        questions = df[QUESTION_COLUM].tolist()
        answers = df[ANSWER_COLUMN].tolist()

        return questions, answers
    except FileNotFoundError:
        print(f"Không thể tìm thấy file csv: {csv_file}")
        return None
    
def classify_question(question, csv_keywords):
    for keyword in csv_keywords:
        if keyword in question.lower():
            return 'csv'
    
    return "outside"

request, response = load_data('m01.csv')
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
transformer_model = SentenceTransformer(model_name)
embeddings = transformer_model.encode(request)

def get_response_csv(user_input):
    user_embedding = transformer_model.encode(user_input)
    smilarities= cosine_similarity([user_embedding], embeddings)[0]
    best_index = numpy.argmax(smilarities)
    return response[best_index]

def get_response_gemini(prompt):
    gemini_response = gemini_model.generate_content(prompt)
    return gemini_response.text

print("Chào mừng bạn đã đến với chatbot Lickey do LIC sản xuất (nhấn thoat để dừng chương trình)")
csv_keywords = ['sinh viên', 'học sinh', 'LIC', 'UDPM', 'lịch sử', 'tên bạn', 'tên', 'bạn']

while True:
    userinput = input("Bạn: ")
    if userinput.lower() == 'thoat':
        print('Lickey: Tạm biệt')
        exit()
    question_type = classify_question(userinput, csv_keywords)
    if question_type == 'csv':
        answer = get_response_csv(userinput)
        print(f"Lickey: {answer}")
    else:
        prompt = f"Trả lời câu hỏi sau: {userinput}"
        answer = get_response_gemini(prompt)
        print(f"Lickey: {answer}")