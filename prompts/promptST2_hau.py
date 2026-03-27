def get_paraphrasing_prompt(original_texts, labels, n_samples=15):
    label_names = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]
    active_labels = [label_names[i] for i, val in enumerate(labels) if val == 1]
    inactive_labels = [label_names[i] for i, val in enumerate(labels) if val == 0]
    
    texts_block = "\n".join([f"- {text}" for text in original_texts])

    prompt = f"""
### Vai trò
Bạn là một chuyên gia ngôn ngữ học về tiếng Hausa, có am hiểu sâu sắc về các sắc thái phân cực xã hội tại Nigeria và Niger. 
DEFINITION:
** Phân Cực (polarization) đề cập đến quá trình hoặc hiện tượng mà trong đó các ý kiến, niềm tin hoặc hành vi trở nên cực đoan hơn hoặc chia rẽ hơn, dẫn đến khoảng cách hoặc xung đột lớn hơn giữa các nhóm khác nhau. Phân hóa thái độ (Attitude polarization) là thái độ tiêu cực mà các cá nhân hoặc nhóm thể hiện đối với các cá nhân và nhóm nằm ngoài nhóm của họ, đồng thời thể hiện sự ủng hộ và đoàn kết mù quáng đối với những người trong nhóm của mình.
** Phân Cực biểu thị sự rập khuôn, phỉ báng, phi nhân hóa, phi cá nhân hóa hoặc không khoan dung đối với quan điểm, niềm tin và danh tính của người khác.
Nhiệm vụ: Dựa trên DNA của dữ liệu NGUỒN, hãy tạo ra {n_samples} mẫu văn bản mới có tính phân cực cao.
### 1. KHÁI NIỆM VỀ CÁC LOẠI PHÂN CỰC (TAXONOMY)
Hãy coi đây là các "vùng không gian khái niệm". Bạn cần hiểu bản chất để có thể tự do sáng tạo từ ngữ tương đương:
* **Political:** Xung đột giữa các đảng phái, sự thù ghét phe phái hoặc các nhân vật chính trị. (Ví dụ: Châm chọc lãnh đạo, cáo buộc tham ô, phản bội đảng...)
* **Racial/Ethnic:** Sự chia rẽ sắc tộc, bộ lạc, vùng miền. (Ví dụ: Miệt thị nguồn gốc, thù ghét nhóm thiểu số/đa số, chia rẽ Nam-Bắc...)
* **Religious:** Xung đột hệ phái, niềm tin, hoặc bài xích người ngoại đạo/người có đức tin khác. (Ví dụ: Phỉ báng cách thực hành tôn giáo, dùng thuật ngữ ám chỉ kẻ vô thần, kẻ phản đạo...)
* **Gender/Sexual:** Định kiến giới tính, xúc phạm nhân phẩm dựa trên giới hoặc lối sống.
* **Other:** Các nhóm yếu thế hoặc đặc thù xã hội (Trẻ ăn xin, dân bến xe, dân nghèo thành thị...).
Lưu ý: Các chủ đề trên chỉ là khung tham chiếu. Bạn cần sử dụng sự sáng tạo ngôn ngữ để mở rộng sang các vấn đề tương tự, đảm bảo văn bản luôn thể hiện polarization.
### 2. DỮ LIỆU NGUỒN (NGUỒN CẢM HỨNG CHÍNH)
Hãy phân tích sâu và mô phỏng phong cách, văn phong, nhịp điệu cùng các hệ thống từ lóng ẩn ý từ các mẫu dữ liệu thực tế sau đây:
{texts_block}

### 3. CHỈ THỊ THIẾT LẬP NHÃN (LOGIC ĐIỀU KHIỂN)
**MỤC TIÊU (STATUS: ON): {", ".join(active_labels) if active_labels else "None"}**
- Bạn PHẢI sử dụng vốn từ vựng phong phú nhất của mình (bao gồm cả từ lóng, thuật ngữ địa phương, và các cách miệt thị ẩn dụ) để thể hiện sự phân cực cho nhãn này. 
- Đừng giới hạn ở vài từ phổ biến; hãy mô phỏng cách người dùng mạng xã hội thực thụ sáng tạo ra các cách chửi bới mới.

**VÙNG CẤM (STATUS: OFF): {", ".join(inactive_labels) if inactive_labels else "None"}**
- Tuyệt đối giữ cho nội dung "sạch" khỏi bất kỳ ẩn ý nào liên quan đến các danh mục này.

### 4. YÊU CẦU VỀ NGÔN NGỮ VÀ VĂN PHONG
Dựa vào phong cách, ngôn ngữ, cấu trúc, thông tin,... từ dữ liệu Nguồn đã cho hãy áp dụng các quy tắc sau nếu phù hợp:
- **Đa dạng hóa ngôn ngữ:** Sử dụng các từ đồng nghĩa, cách diễn đạt khác nhau trong tiếng Hausa để tránh lặp từ.
- **Sáng tạo từ vựng:** Ngoài các từ trong dữ liệu nguồn, hãy sử dụng các từ đồng nghĩa hoặc các cách nói châm biếm khác trong tiếng Hausa để tăng độ đa dạng cho tập dữ liệu.
- **Văn phong:** Ưu tiên phong cách "Hausar baka" (tiếng Hausa khẩu ngữ), sử dụng linh hoạt Code-switching (Hausa-English-Pidgin) nếu phù hợp với ngữ cảnh và dữ liệu NGUỒN được gửi vào.
Tính thực tế: Tạo ra các câu có độ dài ngắn khác nhau. Việc sử dụng Emoji và các ký tự đặc biệt cần dựa vào tỷ lệ xuất hiện của chúng trong dữ liệu NGUỒN được cung cấp và tái lập đúng tỷ lệ đó để mô phỏng sự hỗn loạn.

### 5. Kiểm tra phân cực: 
Văn bản phải thể hiện rõ ràng sự phân cực về thái độ (attitude polarization).
### 6. ĐỊNH DẠNG ĐẦU RA (JSON)
[
  {{
    "text": "Nội dung tiếng Hausa đa dạng...",
    "political": {labels[0]},
    "racial/ethnic": {labels[1]},
    "religious": {labels[2]},
    "gender/sexual": {labels[3]},
    "other": {labels[4]}
  }}
]
"""
    return prompt

def get_balancing_plan():
    # Hau
    return [
        # --- NHÓM ĐƠN LẺ (SINGLE LABELS) ---
        {"combo": "10000", "target_count": 1400, "desc": "Political (Baseline)"},
        {"combo": "01000", "target_count": 350, "desc": "Racial/Ethnic"},
        {"combo": "00100", "target_count": 350, "desc": "Religious"},
        # {"combo": "00010", "target_count": 50, "desc": "Gender/Sexual"},
        # {"combo": "00001", "target_count": 50, "desc": "Other"},

        # # --- NHÓM COMBO HIỆN CÓ (EXISTING COMBOS) ---
        {"combo": "10100", "target_count": 50,  "desc": "Political + Religious (Hiện có: 13)"},
        # {"combo": "11000", "target_count": 50,  "desc": "Political + Racial/Ethnic (Hiện có: 11)"},
        # {"combo": "01100", "target_count": 30,  "desc": "Racial/Ethnic + Religious (Hiện có: 5)"},
        # {"combo": "01010", "target_count": 30,  "desc": "Racial/Ethnic + Gender/Sexual (Hiện có: 3)"},
        # {"combo": "10010", "target_count": 30,  "desc": "Political + Gender/Sexual (Hiện có: 3)"},
        # {"combo": "00110", "target_count": 30,  "desc": "Religious + Gender/Sexual (Hiện có: 2)"}
    ]














