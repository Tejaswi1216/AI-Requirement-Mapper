import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

requirements = pd.read_csv("requirements.csv")
capabilities = pd.read_csv("capabilities.csv")

requirement_embeddings = model.encode(
    requirements["requirement"].tolist()
)

capability_embeddings = model.encode(
    capabilities["capability"].tolist()
)

results = []

for i, requirement_embedding in enumerate(requirement_embeddings):

    scores = cosine_similarity(
        [requirement_embedding],
        capability_embeddings
    )[0]

    best_index = scores.argmax()
    best_score = scores[best_index]

    if best_score >= 0.60:
        status = "MATCH"
    else:
        status = "REVIEW"

    results.append({
        "Requirement ID": requirements.iloc[i]["id"],
        "Requirement": requirements.iloc[i]["requirement"],
        "Best Match": capabilities.iloc[best_index]["capability"],
        "Similarity": round(float(best_score), 3),
        "Status": status
    })

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

results_df.to_csv("results.csv", index=False)