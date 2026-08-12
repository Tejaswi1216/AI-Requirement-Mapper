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

    top_indices = scores.argsort()[::-1][:3]

    best_index = top_indices[0]
    best_score = scores[best_index]

    if best_score >= 0.75:
        confidence = "HIGH"
        status = "MATCH"
    elif best_score >= 0.60:
        confidence = "MEDIUM"
        status = "MATCH"
    else:
        confidence = "LOW"
        status = "REVIEW"

    result = {
        "Requirement ID": requirements.iloc[i]["id"],
        "Requirement": requirements.iloc[i]["requirement"],
        "Best Match": capabilities.iloc[best_index]["capability"],
        "Best Score": round(float(best_score), 3),
        "Confidence": confidence,
        "Status": status
    }

    for rank, index in enumerate(top_indices, start=1):
        result[f"Match {rank}"] = capabilities.iloc[index]["capability"]
        result[f"Score {rank}"] = round(float(scores[index]), 3)

    results.append(result)

results_df = pd.DataFrame(results)

print("\nRequirement Mapping Results\n")
print(results_df.to_string(index=False))

results_df.to_csv("results.csv", index=False)

print("\nResults saved to results.csv")