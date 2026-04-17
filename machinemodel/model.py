import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("uptac2.csv", encoding='latin1', engine='python')

df = df[['Institute', 'Academic Program Name', 'Opening Rank', 'Closing Rank']]
df.columns = ['college', 'branch', 'opening_rank', 'closing_rank']

# clean ranks
df['opening_rank'] = df['opening_rank'].astype(str).str.replace(r'\D', '', regex=True)
df['closing_rank'] = df['closing_rank'].astype(str).str.replace(r'\D', '', regex=True)

df['opening_rank'] = pd.to_numeric(df['opening_rank'], errors='coerce')
df['closing_rank'] = pd.to_numeric(df['closing_rank'], errors='coerce')

df = df.dropna()

df['opening_rank'] = df['opening_rank'].astype(int)
df['closing_rank'] = df['closing_rank'].astype(int)


df['branch_code'] = df['branch'].astype('category').cat.codes

# Save mapping (important for prediction)
branch_mapping = dict(enumerate(df['branch'].astype('category').cat.categories))
reverse_branch_mapping = {v: k for k, v in branch_mapping.items()}

X = df[['opening_rank', 'branch_code']]
y = df['closing_rank']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

def recommend_colleges(rank, branch=""):
    data = df.copy()

    # Step 1: Encode branch
    if branch:
        branch_code = reverse_branch_mapping.get(branch, None)

        if branch_code is not None:
            # Predict expected closing rank
            predicted_rank = model.predict([[rank, branch_code]])[0]

            # Filter same branch
            data = data[data['branch'] == branch]

            # Find closest matches using predicted rank
            data['diff'] = abs(data['closing_rank'] - predicted_rank)

        else:
            # fallback if branch not found
            data['diff'] = abs(data['closing_rank'] - rank)

    else:
        # no branch → normal filtering
        data['diff'] = abs(data['closing_rank'] - rank)

    # Step 2: Sort & return top 5
    result = data.sort_values('diff').head(5)

    return result[['college', 'branch', 'opening_rank', 'closing_rank']].to_dict(orient="records")