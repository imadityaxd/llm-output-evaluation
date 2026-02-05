import pandas as pd

INPUT_PATH = "data/llm_outputs.csv"
OUTPUT_PATH = "data/evaluated_outputs.csv"

def evaluate_response(prompt: str, response: str) -> dict:
    """
    Evaluate LLM response quality using rule-based checks.
    Scores range from 1 (poor) to 5 (excellent).
    """

    relevance = 5 if len(response) > 20 else 3
    clarity = 5 if response.endswith(".") else 3

    known_facts = {
        "capital of india": "new delhi",
        "2 + 2": "4"
    }

    correctness = 5
    hallucination = "No"

    for key, value in known_facts.items():
        if key in prompt.lower() and value not in response.lower():
            correctness = 1
            hallucination = "Yes"

    return {
        "relevance_score": relevance,
        "clarity_score": clarity,
        "correctness_score": correctness,
        "hallucination_flag": hallucination
    }

def main():
    df = pd.read_csv(INPUT_PATH)

    evaluation_results = df.apply(
        lambda row: evaluate_response(row["prompt"], row["llm_response"]),
        axis=1,
        result_type="expand"
    )

    final_df = pd.concat([df, evaluation_results], axis=1)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print("✅ LLM output evaluation completed successfully.")

if __name__ == "__main__":
    main()
