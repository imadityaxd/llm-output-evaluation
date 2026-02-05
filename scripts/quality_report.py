import pandas as pd

INPUT_PATH = "data/evaluated_outputs.csv"
REPORT_PATH = "reports/evaluation_report.txt"

def main():
    df = pd.read_csv(INPUT_PATH)

    total_responses = len(df)
    hallucinated = df[df["hallucination_flag"] == "Yes"].shape[0]

    avg_scores = df[
        ["relevance_score", "clarity_score", "correctness_score"]
    ].mean()

    with open(REPORT_PATH, "w") as report:
        report.write("LLM Output Quality Evaluation Report\n")
        report.write("==================================\n\n")
        report.write(f"Total Responses Evaluated: {total_responses}\n")
        report.write(f"Hallucinated Responses: {hallucinated}\n\n")
        report.write("Average Quality Scores:\n")
        report.write(avg_scores.to_string())
        report.write("\n\nEvaluation completed successfully.")

    print("✅ Evaluation quality report generated.")

if __name__ == "__main__":
    main()
