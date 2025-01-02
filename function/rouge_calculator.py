import pandas as pd
from rouge import Rouge

def calculate_rouge(file_path):
    print(f"Đang xử lý file: {file_path}")
    # Đọc file CSV
    df = pd.read_csv(file_path)

    # Kiểm tra cột "Câu trả lời" và "Answer" có tồn tại
    if "Câu trả lời" not in df.columns or "Answer" not in df.columns:
        raise ValueError("File CSV phải chứa cột 'Câu trả lời' và 'Answer'.")

    # Loại bỏ các dòng chứa giá trị NaN
    df = df.dropna(subset=["Câu trả lời", "Answer"])

    # Lấy dữ liệu từ cột
    reference_answers = df["Câu trả lời"].astype(str).tolist()
    chatbot_answers = df["Answer"].astype(str).tolist()

    # Tạo một đối tượng Rouge
    rouge = Rouge()

    # Danh sách kết quả ROUGE
    rouge_scores = {
        "rouge-1": {"f": [], "p": [], "r": []},
        "rouge-2": {"f": [], "p": [], "r": []},
        "rouge-l": {"f": [], "p": [], "r": []}
    }

    # Tính ROUGE cho từng cặp câu trả lời
    for ref, hyp in zip(reference_answers, chatbot_answers):
        try:
            scores = rouge.get_scores(hyp, ref)[0]  # Lấy điểm số cho từng cặp
            for metric in rouge_scores:
                for sub_metric in rouge_scores[metric]:
                    rouge_scores[metric][sub_metric].append(scores[metric][sub_metric])
        except Exception as e:
            print(f"Lỗi khi tính ROUGE cho: {hyp} vs {ref}. Lỗi: {e}")
            continue

    # Tạo DataFrame kết quả
    results = {
        "metric": [],
        "sub_metric": [],
        "average": []
    }

    for metric in rouge_scores:
        for sub_metric in rouge_scores[metric]:
            # Kiểm tra danh sách không rỗng trước khi tính trung bình
            if len(rouge_scores[metric][sub_metric]) > 0:
                avg_score = sum(rouge_scores[metric][sub_metric]) / len(rouge_scores[metric][sub_metric])
            else:
                avg_score = 0.0  # Hoặc giá trị mặc định khác nếu phù hợp
            results["metric"].append(metric)
            results["sub_metric"].append(sub_metric)
            results["average"].append(avg_score)

    results_df = pd.DataFrame(results)

    # Hiển thị trung bình cộng
    print("Trung bình cộng điểm ROUGE:")
    print(results_df)

    # Ghi kết quả ra file CSV
    output_file = "rouge_scores_output.csv"
    results_df.to_csv(output_file, index=False)

    print(f"Kết quả điểm ROUGE đã được lưu vào file: {output_file}")

# Đường dẫn tới file CSV
file_path = input("Nhập đường dẫn tới file CSV: ")
calculate_rouge(file_path)


#/home/quang-anh/Bản tải về/all-MiniLM-L6-v2 + Vector - Sheet1.csv
#/home/quang-anh/Bản tải về/all-MiniLM-L6-v2 + HYDE - Sheet1.csv
#/home/quang-anh/Bản tải về/SBert_Vector - Sheet1.csv
#/home/quang-anh/Bản tải về/SBert_HYDE - Sheet1.csv