import os
import pandas as pd
from dotenv import load_dotenv

# === 1. Load Environment Variables ===
load_dotenv('token.env')
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found! Check your token.env file.")

from groq import Groq

# === 2. Constants and Prompts ===
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
    "Judge": "You are the judge. Oversee the trial, ensure fairness, ask clarifying questions, and deliver a final verdict based on the arguments and evidence."
}

DEFAULT_MODEL = "llama-3.1-8b-instant"  # Use a model you have access to on Groq

# === 3. LLM Agent Class ===
class LLMChatAgent:
    def __init__(self, role, model=DEFAULT_MODEL, api_key=GROQ_API_KEY):
        self.role = role
        self.system_prompt = ROLE_SYSTEM_PROMPTS[role]
        self.model = model
        self.client = Groq(api_key=api_key)

    def chat(self, messages, temperature=0.5, max_tokens=256):
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

# === 4. Simulation Class ===
class CourtroomSimulation:
    def __init__(self, case_description, model=DEFAULT_MODEL, api_key=GROQ_API_KEY):
        self.case_description = case_description
        self.agents = {role: LLMChatAgent(role, model=model, api_key=api_key) for role in AGENT_ROLES}
        self.transcript = []

    def run_phase(self, phase_name, instructions):
        phase_messages = []
        for role, instruction in instructions:
            agent = self.agents[role]
            messages = [
                {"role": "user", "content": f"Case Description: {self.case_description}\n\n{instruction}"}
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
                ("Judge", "Review the arguments and evidence presented and deliver your verdict with reasoning."),
            ]
        )
        return self.transcript

# === 5. Main Functionality ===
def main():
    # Load your data.csv file
    df = pd.read_csv("data.csv", header=None, names=["case_description"])

    # Show a preview of available cases (first 100 chars)
    print("\nAvailable Cases:\n")
    for idx, row in df.iterrows():
        preview = str(row['case_description'])[:100].replace('\n', ' ')
        print(f"{idx}: {preview}...")

    # Ask user to pick a case by row number
    selected_idx = input("\nEnter the row number of the case you want to simulate: ")
    try:
        selected_idx = int(selected_idx)
        if selected_idx < 0 or selected_idx >= len(df):
            raise ValueError
    except ValueError:
        print("Invalid selection. Exiting.")
        exit(1)

    selected_case = df.iloc[selected_idx]
    print(f"\nRunning simulation for selected case:\n{selected_case['case_description']}\n")

    # Run the simulation
    MAX_DESC_CHARS = 2000
    short_case_description = selected_case['case_description'][:MAX_DESC_CHARS]
    sim = CourtroomSimulation(short_case_description)

    transcript = sim.simulate()
    for role, phase, reply in transcript:
        print(f"\n[{phase}] {role}: {reply}")

# === 6. Script Entry Point ===
if __name__ == "__main__":
    main()
