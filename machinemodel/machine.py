from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
CORS(app)

# SECURITY WARNING: You should use environment variables for this in production!
client = OpenAI( 
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# 2. Data Cleaning Utility
def clean_cols(name):
    return re.sub(r'[^a-zA-Z0-9\s]', '', name).strip()

# 3. Load and Process Dataset
try:
    df_raw = pd.read_csv("uptac2.csv", encoding='latin1', engine='python')
    df_raw1 = pd.read_csv("Book3.csv", encoding='latin1', engine='python')
    df_raw2 = pd.read_csv("Book4.csv", encoding='latin1', engine='python')
    df_raw3 = pd.read_csv("JEE_Rank_2016_2024.csv", encoding='latin1', engine='python')
    df_raw4 = pd.read_csv("Book2.csv", encoding='latin1', engine='python')
    
    df_raw.columns = [clean_cols(col) for col in df_raw.columns]
    df_raw1.columns = [clean_cols(col) for col in df_raw1.columns]
    df_raw2.columns = [clean_cols(col) for col in df_raw2.columns]
    df_raw3.columns = [clean_cols(col) for col in df_raw3.columns]
    df_raw4.columns = [clean_cols(col) for col in df_raw4.columns]
    
    # Assuming these CSVs have the same structure and you want to stack them:
    # Changed axis=1 to axis=0 to stack the rows. Using ignore_index=True is safer.
    combined_df = pd.concat([df_raw, df_raw1, df_raw2, df_raw3, df_raw4], axis=0, ignore_index=True)
    
    mapping = {
        'Institute': 'college',
        'Program': 'branch',
        'Opening Rank': 'opening_rank',
        'Closing Rank': 'closing_rank'
    }
    
    # USE the combined data, not just df_raw
    df = combined_df.rename(columns=mapping)
    
    # FIX: Don't drop 'city', 'fees_lakhs', etc. Keep all relevant columns!
    cols_to_keep = ['college', 'branch', 'opening_rank', 'closing_rank', 'city', 'counselling_board', 'fees_lakhs', 'avg_package']
    existing_cols = [col for col in cols_to_keep if col in df.columns]
    df = df[existing_cols]

    # Rank cleaning
    df['opening_rank'] = pd.to_numeric(df['opening_rank'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
    df['closing_rank'] = pd.to_numeric(df['closing_rank'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')

    # Drop rows where critical rank data is missing
    df = df.dropna(subset=['college', 'branch', 'opening_rank', 'closing_rank'])

    # Remove duplicates
    df = df.drop_duplicates(subset=['college', 'branch'], keep='last')

    # Branch Encoding
    df['branch_code'] = df['branch'].astype('category').cat.codes
    reverse_branch_mapping = dict(enumerate(df['branch'].astype('category').cat.categories))
    branch_mapping = {v: k for k, v in reverse_branch_mapping.items()}

    print(f"✅ Dataset Loaded: {len(df)} Unique College-Branch Pairs Found.")

except Exception as e:
    print(f" Error loading CSV: {e}")
    df = pd.DataFrame()

# 4. Model Training
# NOTE: You train this model here but you never actually use it in the /recommend endpoint!
if not df.empty:
    X = df[['opening_rank', 'branch_code']]
    y = df['closing_rank']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("✅ ML Model Trained and Ready")
else:
    model = None

# --- API ROUTES ---

@app.route("/")
def home():
    return "WiseWays Machine Engine is Running 🚀"

@app.route('/ask', methods=['POST'])
def ask_ai():
    data = request.json
    query = data.get("query", "")
    try:
        completion = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": query}],
            temperature=0.5
        )
        return jsonify({"response": completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    try:
        rank = int(data.get("rank", 0))
        category_rank_input = data.get("categoryRank", "")
        category_rank = int(category_rank_input) if category_rank_input else 0
        
        effective_rank = category_rank if category_rank > 0 else rank
        
        # FIX: Protect against Division by Zero if user passes 0
        if effective_rank <= 0:
            return jsonify({"error": "Please provide a valid rank greater than 0"}), 400
            
        branch_input = data.get("branch", "").strip()
        area = data.get("area", "").strip().lower()
        budget = data.get("budget", "any")
        counselling = data.get("counselling", "any").lower()
        
        temp_df = df.copy()

        # --- Apply Filters ---
        if area and 'city' in temp_df.columns:
            temp_df = temp_df[temp_df['city'].str.lower().str.contains(area, na=False)]
            
        if counselling != "any" and 'counselling_board' in temp_df.columns:
            temp_df = temp_df[temp_df['counselling_board'].str.lower() == counselling]
            
        if budget != "any" and 'fees_lakhs' in temp_df.columns:
            # Convert fees_lakhs to numeric just in case it's strings
            temp_df['fees_lakhs'] = pd.to_numeric(temp_df['fees_lakhs'], errors='coerce')
            if budget == "5-10": temp_df = temp_df[(temp_df['fees_lakhs'] >= 5) & (temp_df['fees_lakhs'] <= 10)]
            elif budget == "10-20": temp_df = temp_df[(temp_df['fees_lakhs'] >= 10) & (temp_df['fees_lakhs'] <= 20)]
            elif budget == "20-40": temp_df = temp_df[(temp_df['fees_lakhs'] >= 20) & (temp_df['fees_lakhs'] <= 40)]
            elif budget == "40+": temp_df = temp_df[temp_df['fees_lakhs'] > 40]

        # --- Recommendation Logic ---
        buffer_rank = effective_rank * 0.95  
        
        eligible_df = temp_df[temp_df['closing_rank'] >= buffer_rank].copy()

        if eligible_df.empty:
            temp_df['diff'] = abs(temp_df['closing_rank'] - effective_rank)
            eligible_df = temp_df.sort_values('diff').head(10)
        else:
            eligible_df['diff'] = eligible_df['closing_rank'] - effective_rank
            
            if branch_input and branch_input in branch_mapping:
                b_code = branch_mapping[branch_input]
                eligible_df['branch_bonus'] = eligible_df['branch_code'].apply(lambda x: 0 if x == b_code else 100000)
                eligible_df['total_sort_score'] = eligible_df['diff'] + eligible_df['branch_bonus']
                eligible_df = eligible_df.sort_values('total_sort_score')
            else:
                eligible_df = eligible_df.sort_values('diff')

        # --- Smart Reality Score Calculation ---
        results = []
        for index, row in eligible_df.head(5).iterrows():
            diff = abs(row['closing_rank'] - effective_rank)
            if row['closing_rank'] >= effective_rank:
                percent_diff = (diff / effective_rank) * 100
                score = max(50, 98 - int(percent_diff * 0.8)) 
            else:
                score = max(30, 85 - int((diff / effective_rank) * 100)) 

            score = min(99, score) 

            results.append({
                "college": row['college'],
                "branch": row['branch'],
                "closing_rank": row['closing_rank'],
                "match_score": score,
                "city": row.get('city', 'N/A'), 
                "avg_package": f"{row.get('avg_package', 'N/A')} LPA" if pd.notna(row.get('avg_package')) else "N/A",
                "fees": f"{row.get('fees_lakhs', 'N/A')} Lakhs" if pd.notna(row.get('fees_lakhs')) else "N/A"
            })
            
        return jsonify({"colleges": results})
        
    except Exception as e:
        print(f"Recommend Error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)