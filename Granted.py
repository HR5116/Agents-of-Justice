import pandas as pd
from dotenv import load_dotenv
import os
import re
from groq import Groq

# Load API key from token.env
load_dotenv('token.env')
api_key = os.getenv("GROQ_API_KEY")

# --- Agent and Simulation Classes ---
AGENT_ROLES = [
    "Defendant",
    "Defense Lawyer",
    "Plaintiff",
    "Prosecution Lawyer",
    "Judge"
]

ROLE_SYSTEM_PROMPTS = {
    "Defendant": "You are the defendant in a court case. Respond truthfully, defend yourself, and answer questions from your lawyer and the prosecution.",
    "Defense Lawyer": "You are the defense lawyer. Your job is to defend the defendant, question witnesses, and counter the prosecution's arguments.",
    "Plaintiff": "You are the plaintiff in a court case. State your claims clearly and respond to questions from your lawyer and the defense.",
    "Prosecution Lawyer": "You are the prosecution lawyer. Present the case against the defendant, question witnesses, and counter the defense's arguments.",
    "Judge": "You are the judge. Oversee the trial, ensure fairness, ask clarifying questions, and deliver a final verdict (GRANTED or DENIED) based on the arguments, evidence, and witness testimonies. In your final statement, clearly state: 'VERDICT: GRANTED' or 'VERDICT: DENIED'."
}

class LLMChatAgent:
    def __init__(self, role, model="llama-3.1-8b-instant"):
        self.role = role
        self.system_prompt = ROLE_SYSTEM_PROMPTS[role]
        self.model = model
        self.client = Groq(api_key=api_key)

    def chat(self, messages, temperature=0.5, max_tokens=512):
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages
        response = self.client.chat.completions.create(
            messages=full_messages,
            model=self.model,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=False,
        )
        return response.choices[0].message.content.strip()

class CourtroomSimulation:
    def __init__(self, case_description):
        self.case_description = case_description
        self.agents = {role: LLMChatAgent(role) for role in AGENT_ROLES}
        self.transcript = []

    def run_phase(self, phase_name, instructions):
        phase_messages = []
        for role, instruction in instructions:
            agent = self.agents[role]
            messages = [
                {"role": "user", "content": f"Case Description: {self.case_description[:2000]}\n\n{instruction}"}
            ]
            reply = agent.chat(messages)
            self.transcript.append((role, phase_name, reply))
            phase_messages.append((role, reply))
        return phase_messages

    def simulate(self):
        self.run_phase(
            "Opening Statements",
            [
                ("Prosecution Lawyer", "Present your opening statement to introduce your side of the case."),
                ("Defense Lawyer", "Present your opening statement to introduce your side of the case."),
            ]
        )
        self.run_phase(
            "Witness Interrogation & Argumentation",
            [
                ("Prosecution Lawyer", "Question the plaintiff and present your main arguments."),
                ("Defense Lawyer", "Question the defendant and present your main arguments."),
                ("Plaintiff", "Respond to the prosecution lawyer's questions."),
                ("Defendant", "Respond to the defense lawyer's questions."),
            ]
        )
        self.run_phase(
            "Closing Statements",
            [
                ("Prosecution Lawyer", "Summarize your side's case in a closing statement."),
                ("Defense Lawyer", "Summarize your side's case in a closing statement."),
            ]
        )
        self.run_phase(
            "Judge's Ruling",
            [
                ("Judge", "Review the arguments and evidence presented and deliver your verdict with reasoning. In your final statement, clearly state: 'VERDICT: GRANTED' or 'VERDICT: DENIED'."),
            ]
        )
        return self.transcript

def extract_verdict(judge_text):
    """Extracts 1 for GRANTED, 0 for DENIED from judge's verdict statement."""
    match = re.search(r"VERDICT:\s*(GRANTED|DENIED)", judge_text, re.IGNORECASE)
    if match:
        verdict = match.group(1).upper()
        return 1 if verdict == "GRANTED" else 0
    # Fallback: try to guess from keywords
    if "grant" in judge_text.lower():
        return 1
    if "deny" in judge_text.lower():
        return 0
    return 0

# --- Main prediction loop ---
if __name__ == "__main__":
    # Load cases starting from index 32 (skip first 32 rows)
    start_index = 40
    df = pd.read_csv("data.csv", header=None, names=["case_text"])
    cases = df.iloc[start_index:start_index + 50, :]  # Next 50 cases from index 32

    results = []
    for i, row in enumerate(cases.itertuples(index=False), 1):
        case_id = f"case_{i + start_index}"  # Preserve original index in ID
        case_text = row.case_text
        print(f"Processing {case_id} ({i}/50)...")
        sim = CourtroomSimulation(case_text)
        transcript = sim.simulate()
        judge_reply = [r for r in transcript if r[0] == "Judge" and r[1] == "Judge's Ruling"]
        if judge_reply:
            judge_text = judge_reply[-1][2]
            label = extract_verdict(judge_text)
        else:
            label = 0
        results.append((case_id, label))
        print(f"{case_id}: {label}")

    # Write to submission.csv
    sub_df = pd.DataFrame(results, columns=["id", "label"])
    sub_df.to_csv("submission.csv", index=False)
    print("Submission file 'submission.csv' created successfully.")
