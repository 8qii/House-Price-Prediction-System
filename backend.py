import os
import json
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai
from flask import Flask, request, jsonify
import time

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DATA_CSV = "data/HousePrice_processed.csv"
MODEL_PKL = "models/random_forest_model.pkl"
MAPPINGS_PKL = "models/mappings.pkl"
ENCODERS_PKL = "models/label_encoders.pkl"

# RAG weights
WEIGHT_STRUCTURED = 0.6
WEIGHT_TEXT = 0.4

# Minimal numeric features required to make a prediction
MINIMAL_NUMERIC_REQUIRED = ["Super Area","BHK"]  # chỉnh nếu cần

# Top sizes
DEFAULT_TOP_K = 100
DEFAULT_RERANK_K = 10

GOOGLE_API_KEY = "AIzaSyA7KMSr2hrq4Px5U544NXOI2YVJVrmBL9c"  # <-- Thay bằng API key của bạn
genai.configure(api_key=GOOGLE_API_KEY)

extract_model = genai.GenerativeModel("gemini-2.0-flash")
gen_model = genai.GenerativeModel("gemini-2.5-flash-lite")


if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(f"Missing data file: {DATA_CSV}")
if not os.path.exists(MODEL_PKL):
    raise FileNotFoundError(f"Missing model file: {MODEL_PKL}")
if not os.path.exists(MAPPINGS_PKL):
    raise FileNotFoundError(f"Missing mappings file: {MAPPINGS_PKL}")
if not os.path.exists(ENCODERS_PKL):
    raise FileNotFoundError(f"Missing encoders file: {ENCODERS_PKL}")

df_raw = pd.read_csv(DATA_CSV)
rf_model = joblib.load(MODEL_PKL)

with open(MAPPINGS_PKL, "rb") as f:
    mappings_obj = pickle.load(f)

label_encoders = joblib.load(ENCODERS_PKL)  # dict: {col: LabelEncoder()}

if isinstance(mappings_obj, dict) and "feature_order" in mappings_obj:
    model_features = list(mappings_obj["feature_order"])
    categorical_cols = list(mappings_obj.get("categorical_cols", []))
    # if mappings for categorical value->int also present, use them
    categorical_value_maps = mappings_obj.get("value_maps", {})  # optional
else:
    # If the file is a mapping col->value2int, then categorical_cols are keys of dict
    # and model_features we must infer: take label_encoders keys + numeric columns from df
    if all(isinstance(v, dict) for v in mappings_obj.values()):
        categorical_value_maps = mappings_obj
        categorical_cols = list(categorical_value_maps.keys())
        # numeric cols: choose df columns not in categorical and not in textual rag-only
        rag_text_cols = ["Title", "Description", "Society", "overlooking"]
        numeric_and_others = [c for c in df_raw.columns if c not in categorical_cols and c not in rag_text_cols]
        model_features = categorical_cols + [c for c in numeric_and_others if c != mappings_obj.get("target_col", None)]
    else:
        # fallback: try to use label_encoders keys and df columns
        categorical_cols = list(label_encoders.keys())
        rag_text_cols = ["Title", "Description", "Society", "overlooking"]
        numeric_and_others = [c for c in df_raw.columns if c not in categorical_cols and c not in rag_text_cols]
        model_features = categorical_cols + [c for c in numeric_and_others if c != "Price" and c != "Amount"]

# Keep RAG text columns (we won't use them as structured features for RF)
RAG_TEXT_COLS = [c for c in ["Title", "Description", "Society", "overlooking"] if c in df_raw.columns]

def row_to_structured_vector(row, features, label_encoders_local):
    vec = []
    for col in features:
        if col in label_encoders_local:
            # categorical mapping via encoder
            val = row.get(col, None)
            if pd.isna(val):
                val = "Unknown"
            try:
                le = label_encoders_local[col]
                # If value unseen, use transform on "Unknown" if encoder fit had it; else use 0
                if str(val) not in le.classes_.tolist():
                    if "Unknown" in le.classes_:
                        mapped = int(le.transform(["Unknown"])[0])
                    else:
                        # try to coerce to string and if still fails, 0
                        try:
                            mapped = int(le.transform([str(val)])[0])
                        except Exception:
                            mapped = 0
                else:
                    mapped = int(le.transform([str(val)])[0])
                vec.append(float(mapped))
            except Exception:
                # fallback: 0
                vec.append(0.0)
        else:
            # numeric
            v = row.get(col, None)
            try:
                vec.append(float(v) if not pd.isna(v) else 0.0)
            except:
                vec.append(0.0)
    return np.array(vec, dtype=np.float32)

print("Building dataset structured vectors...")
# Build DataFrame copy for RAG uses (we keep original df_raw for returning rows)
df = df_raw.copy().fillna(0)
dataset_struct_vectors = np.stack([row_to_structured_vector(row, model_features, label_encoders) for _, row in df.iterrows()])

# Min-max scaling per column for cosine stability
col_min = dataset_struct_vectors.min(axis=0)
col_max = dataset_struct_vectors.max(axis=0)
col_range = col_max - col_min
col_range[col_range == 0] = 1.0
dataset_struct_scaled = (dataset_struct_vectors - col_min) / col_range

text_corpus = (
    df_raw["Title"].fillna("") if "Title" in df_raw.columns else pd.Series([""] * len(df_raw))
) + " || " + (
    df_raw["Description"].fillna("") if "Description" in df_raw.columns else pd.Series([""] * len(df_raw))
) + " || " + (
    df_raw["Society"].fillna("") if "Society" in df_raw.columns else pd.Series([""] * len(df_raw))
) + " || " + (
    df_raw["overlooking"].fillna("") if "overlooking" in df_raw.columns else pd.Series([""] * len(df_raw))
)

tfidf = TfidfVectorizer(max_features=20000, stop_words='english')
print("Fitting TF-IDF...")
text_matrix = tfidf.fit_transform(text_corpus.values)

# --- Hàm gọi Gemini để extract đặc trưng ---
def extract_entities_with_gemini(user_query: str) -> dict:
    """
    Dùng Gemini để trích xuất các đặc trưng từ câu mô tả BĐS.
    Trả về dict với các trường cần thiết, nếu không có thì giá trị None.
    """
    prompt = f"""
Bạn là hệ thống trích xuất dữ liệu BĐS.
Hãy phân tích mô tả sau và trả về duy nhất 1 đối tượng JSON với các trường:
- BHK (int|null)
- Bathroom (int|null)
- Balcony (int|null)
- Car Parking (string|null)
- Furnishing (string|null)
- Facing (string|null)
- Current Floor (int|null)
- Total Floors (int|null)
- Carpet Area (float|null)
- Super Area (float|null)
- Location (string|null)

Ví dụ:
Mô tả: "2 BHK nhà gần trung tâm, 2 phòng tắm, 1 chỗ đỗ xe mở, bán phần nội thất, hướng đông, diện tích thảm 1200 sqft"
JSON: {{"BHK":2,"Bathroom":2,"Balcony":null,"Car Parking":"1 Open","Furnishing":"Semi-Furnished","Facing":"East","Current Floor":null,"Total Floors":null,"Carpet Area":1200,"Super Area":null,"Location":null}}

Chỉ trả về JSON hợp lệ, không thêm giải thích.

Mô tả: "{user_query}"
    """

    try:
        response = extract_model.generate_content(prompt)
        text = response.text.strip()

        # Nếu model trả về dạng code fence ```json ... ```
        if text.startswith("```"):
            text = text.strip("`").replace("json", "").strip()

        data = json.loads(text)
        return data

    except Exception as e:
        print(f"Lỗi khi gọi Gemini: {e}")
        return {}
    
def json_to_model_vector(entity_json: dict):
    """
    Return:
      model_vec_raw: array shape (n_features,)
      model_vec_scaled: scaled version (min-max using training dataset)
      missing_features: list of features from model_features considered missing (only numeric ones we treat required)
    """
    vec_raw = []
    missing = []
    for idx, col in enumerate(model_features):
        if col in label_encoders:
            # categorical
            val = entity_json.get(col, None)
            # try some aliases if None
            if val is None:
                # common alias mapping
                alias_map = {
                    "Location": ["Location","location","city","area"],
                    "Furnishing": ["Furnishing","furnishing"],
                    "Facing": ["Facing","facing"],
                    "Transaction": ["Transaction","transaction"]
                }
                for alias in alias_map.get(col, []):
                    if alias in entity_json and entity_json.get(alias) is not None:
                        val = entity_json.get(alias)
                        break
            if val is None:
                # map to 'Unknown' if encoder saw it, else 0
                le = label_encoders[col]
                if "Unknown" in le.classes_:
                    mapped = int(le.transform(["Unknown"])[0])
                else:
                    mapped = 0
            else:
                le = label_encoders[col]
                try:
                    # if unseen value, fallback to Unknown if exists, else 0
                    if str(val) not in le.classes_.tolist():
                        if "Unknown" in le.classes_:
                            mapped = int(le.transform(["Unknown"])[0])
                        else:
                            mapped = 0
                    else:
                        mapped = int(le.transform([str(val)])[0])
                except Exception:
                    mapped = 0
            vec_raw.append(float(mapped))
        else:
            # numeric
            # try direct keys and aliases
            found_val = None
            if col in entity_json and entity_json.get(col) is not None:
                found_val = entity_json.get(col)
            else:
                alias_map = {
                    "Carpet Area": ["Carpet Area","carpet_area","area","CarpetArea","carpet"],
                    "Super Area": ["Super Area","super_area","superarea"],
                    "BHK": ["BHK","bhk","Bedrooms","Bedroom"],
                    "Bathroom": ["Bathroom","bathrooms","bath"]
                }
                for a in alias_map.get(col, []):
                    if a in entity_json and entity_json.get(a) is not None:
                        found_val = entity_json.get(a)
                        break
            if found_val is None:
                vec_raw.append(0.0)
                missing.append(col)
            else:
                try:
                    vec_raw.append(float(found_val))
                except:
                    vec_raw.append(0.0)
                    missing.append(col)
    model_vec_raw = np.array(vec_raw, dtype=np.float32)
    model_vec_scaled = (model_vec_raw - col_min) / col_range
    return model_vec_raw, model_vec_scaled, missing


# We'll consider as categorical: object dtype and not in RAG_TEXT_COLS
candidate_categorical = [c for c in df.columns if df[c].dtype == "object" and c not in RAG_TEXT_COLS]
# Numeric candidates
candidate_numeric = [c for c in df.columns if c not in candidate_categorical and c not in RAG_TEXT_COLS]

def rag_and_rerank(entity_json, top_k=DEFAULT_TOP_K, rerank_k=DEFAULT_RERANK_K):
    q_raw, q_scaled, missing = json_to_model_vector(entity_json)
    # structured similarity
    struct_sims = cosine_similarity([q_scaled], dataset_struct_scaled)[0]
    # text similarity: build query text
    text_parts = []
    if entity_json.get("Location"):
        text_parts.append(str(entity_json.get("Location")))
    if entity_json.get("Society"):
        text_parts.append(str(entity_json.get("Society")))
    # also consider user may have provided location or other text in entity_json
    qtext = " || ".join(text_parts) if text_parts else ""
    if qtext.strip() == "":
        text_sims = np.zeros(len(df))
    else:
        qtext_vec = tfidf.transform([qtext])
        text_sims = cosine_similarity(qtext_vec, text_matrix).flatten()
    combined = WEIGHT_STRUCTURED * struct_sims + WEIGHT_TEXT * text_sims
    top_idx = np.argsort(combined)[::-1][:top_k]
    top_scores = combined[top_idx]
    # re-ranking by feature-match heuristic
    top_vectors_raw = dataset_struct_vectors[top_idx]
    q_unscaled = q_raw
    max_per_feature = np.maximum(top_vectors_raw.max(axis=0), 1.0)
    feature_diffs = np.abs(top_vectors_raw - q_unscaled)
    feature_match_scores = 1 - feature_diffs.mean(axis=1) / (np.mean(max_per_feature) + 1e-6)
    total_scores = 0.4 * top_scores + 0.6 * feature_match_scores
    order = np.argsort(total_scores)[::-1]
    reranked_idx = top_idx[order][:rerank_k]
    reranked_scores = total_scores[order][:rerank_k]
    results = []
    for idx, score in zip(reranked_idx, reranked_scores):
        row = df_raw.iloc[idx].to_dict()  # return original row with text fields
        row["_combined_score"] = float(score)
        row["_structured_sim"] = float(struct_sims[idx])
        row["_text_sim"] = float(text_sims[idx])
        results.append((int(idx), row))
    return results, missing

def predict_for_entities(entity_json):
    model_vec_raw, model_vec_scaled, missing = json_to_model_vector(entity_json)
    missing_minimal = [m for m in MINIMAL_NUMERIC_REQUIRED if m in missing]
    if len(missing_minimal) > 0:
        return {"ok": False, "missing_fields": missing_minimal}
    # RF expects ordering model_features
    vec2d = model_vec_raw.reshape(1, -1)
    pred = float(rf_model.predict(vec2d)[0])
    return {"ok": True, "predicted_amount": pred}

def handle_user_query(query_str, top_k=DEFAULT_TOP_K, rerank_k=DEFAULT_RERANK_K):
    entities = extract_entities_with_gemini(query_str)
    if entities is None:
        return {"error": "Gemini parse failed. Please rephrase or provide structured data."}
    # include query text optionally
    entities["_user_query"] = query_str
    rag_results, missing_rag = rag_and_rerank(entities, top_k=top_k, rerank_k=rerank_k)
    pred_info = predict_for_entities(entities)
    response = {
        "entities": entities,
        "rag_candidates": [{"index": idx, "row": row} for idx, row in rag_results],
        "prediction": None,
        "missing_required_fields": None,
        "message": None
    }
    if not pred_info["ok"]:
        response["missing_required_fields"] = pred_info["missing_fields"]
        response["message"] = f"Thiếu thông tin cần thiết để dự đoán: {pred_info['missing_fields']}. Vui lòng cung cấp thêm (ví dụ: Carpet Area, Super Area, BHK)."
    else:
        response["prediction"] = pred_info["predicted_amount"]
        response["message"] = "Dự đoán thành công."
    return response
PROMPT_TEMPLATE_MISSING = """
Bạn là một chuyên gia bất động sản tại Ấn Độ, đồng thời là một trợ lý thân thiện.

Trước tiên, hãy xác định ý định của người dùng:
- Nếu câu hỏi/nhắn tin của họ không liên quan đến bất động sản hoặc dự đoán giá nhà (ví dụ: lời chào, hỏi thăm, câu nói chung chung), hãy trả lời ngắn gọn, tự nhiên và thân thiện, không ép buộc họ cung cấp thông tin về nhà đất.
- Nếu liên quan đến bất động sản nhưng thiếu thông tin để dự đoán giá, hãy làm các bước sau:

1. Xác nhận rằng hiện tại chưa đủ thông tin để dự đoán giá căn nhà họ cần.
2. Liệt kê rõ các thuộc tính còn thiếu: {missing_fields}
3. Đưa ra danh sách những căn nhà tương đồng nhất với yêu cầu của họ, mỗi căn gồm: địa điểm, diện tích, số phòng (BHK), giá (nếu có), và một câu mô tả ngắn.
4. Kết thúc bằng lời nhắc:
   "Nếu bạn muốn có dự đoán chính xác, vui lòng cung cấp thêm đầy đủ các thông tin còn thiếu."

Dữ liệu căn nhà tương đồng:
{rag_candidates}

Câu hỏi gốc của người dùng:
{user_query}

Hãy viết câu trả lời bằng tiếng Việt, súc tích, rõ ràng, không thêm thông tin thừa.
"""


PROMPT_TEMPLATE_PREDICT = """
Bạn là một chuyên gia bất động sản tại Ấn Độ, đồng thời là một trợ lý thân thiện.

Trước tiên, hãy xác định ý định của người dùng:
- Nếu câu hỏi/nhắn tin của họ không liên quan đến bất động sản hoặc dự đoán giá nhà (ví dụ: lời chào, hỏi thăm, câu nói chung chung), hãy trả lời ngắn gọn, tự nhiên và thân thiện, không ép buộc họ cung cấp thông tin về nhà đất.
- Nếu liên quan đến bất động sản và đã đủ thông tin để dự đoán, hãy làm các bước sau:

Người dùng hỏi: {user_query}

Dự đoán giá: {predicted_price}

Dữ liệu few-shot (15 bản ghi tương đồng):
{few_shot_records}

Hãy viết câu trả lời:
- Thông báo kết quả dự đoán giá nhà của người dùng.
- Giải thích ngắn gọn tại sao có kết quả này (dựa vào few-shot records).
- Trình bày dữ liệu ví dụ từ few-shot records để tăng độ tin cậy.

Chỉ trả lời tập trung, không thêm thông tin thừa.
"""


def generate_gemini_response(user_query, handler_output):

    if handler_output["missing_required_fields"]:
        # Case 1: Thiếu thông tin
        prompt = PROMPT_TEMPLATE_MISSING.format(
            user_query=user_query,
            rag_candidates="\n".join(
                [str(c["row"]) for c in handler_output["rag_candidates"]]
            ),
            missing_fields=", ".join(handler_output["missing_required_fields"])
        )
    else:
        # Case 2: Đủ thông tin
        prompt = PROMPT_TEMPLATE_PREDICT.format(
            user_query=user_query,
            predicted_price=handler_output["prediction"],
            few_shot_records="\n".join(
                [str(c["row"]) for c in handler_output["rag_candidates"]]
            )
        )

    response = gen_model.generate_content(prompt)
    return response.text

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request"}), 400
    query = data['query']
    handler_output = handle_user_query(query)
    if "error" in handler_output:
        return jsonify({"answer": handler_output["error"]}), 400
    answer = generate_gemini_response(query, handler_output)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False)
    time.sleep(2)
