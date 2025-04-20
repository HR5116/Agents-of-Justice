Automated Courtroom Simulation & Verdict Extraction
This project uses Large Language Models (LLMs) via the Groq API to simulate courtroom trials and automatically extract verdicts (GRANTED or DENIED) from the judge's final ruling.
Each courtroom participant (Judge, Lawyers, Plaintiff, Defendant) is represented by an AI agent with a unique persona and prompt. The system processes a batch of legal cases, simulates the trial for each, and outputs the predicted verdicts in a CSV file.

Project Description
This tool is designed for:

Legal AI research: Automate the simulation of court cases and verdict prediction.

Education: Demonstrate multi-agent AI role-play in a legal context.

Dataset labeling: Generate synthetic verdict labels for large sets of legal scenarios.

Workflow:

Load a batch of legal cases from a CSV file.

For each case:

Simulate a full courtroom trial (opening statements, arguments, closing, judge's ruling).

Extract the verdict (GRANTED or DENIED) from the judge's statement.

Output a CSV file with case IDs and predicted labels.

Architecture
High-Level Pipeline:

text
+---------------------+
|   data.csv (cases)  |
+----------+----------+
           |
           v
+---------------------+
|  CourtroomSimulation|
|  - LLMChatAgent     |
+----------+----------+
           |
           v
+---------------------+
|   Groq API (LLMs)   |
+----------+----------+
           |
           v
+---------------------+
|  Judge's Ruling     |
+----------+----------+
           |
           v
+---------------------+
| Verdict Extraction  |
+----------+----------+
           |
           v
+---------------------+
| submission.csv      |
+---------------------+
