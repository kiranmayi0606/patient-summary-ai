import os
import pathway as pw
import pandas as pd
import pathway.persistence as pwp
from pathway.xpacks.llm.embedders import OpenAIEmbedder 
# --- LINE 7 (CORRECTED) ---
from pathway.xpacks.llm.vector_store import build_vector_store_index

print("Starting the data indexing process...")

# --- 1. Define Our "Unified Patient Fact" Schema ---
class PatientFact(pw.Schema):
    fact_id: str
    patient_id: str
    event_date: str
    event_type: str
    description: str
    source_file: str

# --- 2. Data Loading & Cleaning Functions ---
# (These functions are unchanged and correct)
def load_conditions(data_path):
    facts = []
    try:
        df = pd.read_csv(f"{data_path}/conditions.csv", usecols=['START', 'PATIENT', 'DESCRIPTION'])
        for index, row in df.iterrows():
            facts.append(PatientFact(
                fact_id=f"conditions_{index}",
                patient_id=row['PATIENT'],
                event_date=row['START'],
                event_type="Condition",
                description=row['DESCRIPTION'],
                source_file="conditions.csv"
            ))
        print(f"Loaded {len(facts)} facts from conditions.csv")
    except FileNotFoundError:
        print("conditions.csv not found.")
    return facts

def load_medications(data_path):
    facts = []
    try:
        df = pd.read_csv(f"{data_path}/medications.csv", usecols=['START', 'PATIENT', 'DESCRIPTION'])
        for index, row in df.iterrows():
            facts.append(PatientFact(
                fact_id=f"medications_{index}",
                patient_id=row['PATIENT'],
                event_date=row['START'],
                event_type="Medication",
                description=row['DESCRIPTION'],
                source_file="medications.csv"
            ))
        print(f"Loaded {len(facts)} facts from medications.csv")
    except FileNotFoundError:
        print("medications.csv not found.")
    return facts

def load_allergies(data_path):
    facts = []
    try:
        df = pd.read_csv(f"{data_path}/allergies.csv", usecols=['START', 'PATIENT', 'DESCRIPTION'])
        for index, row in df.iterrows():
            facts.append(PatientFact(
                fact_id=f"allergies_{index}",
                patient_id=row['PATIENT'],
                event_date=row['START'],
                event_type="Allergy",
                description=row['DESCRIPTION'],
                source_file="allergies.csv"
            ))
        print(f"Loaded {len(facts)} facts from allergies.csv")
    except FileNotFoundError:
        print("allergies.csv not found.")
    return facts

def load_observations(data_path):
    facts = []
    try:
        df = pd.read_csv(f"{data_path}/observations.csv", usecols=['DATE', 'PATIENT', 'DESCRIPTION', 'VALUE', 'UNITS'])
        df = df.dropna(subset=['VALUE'])
        for index, row in df.iterrows():
            desc = f"{row['DESCRIPTION']}: {row['VALUE']} {row['UNITS'] or ''}".strip()
            facts.append(PatientFact(
                fact_id=f"observations_{index}",
                patient_id=row['PATIENT'],
                event_date=row['DATE'],
                event_type="Observation",
                description=desc,
                source_file="observations.csv"
            ))
        print(f"Loaded {len(facts)} facts from observations.csv")
    except FileNotFoundError:
        print("observations.csv not found.")
    return facts

def load_procedures(data_path):
    facts = []
    try:
        df = pd.read_csv(f"{data_path}/procedures.csv", usecols=['DATE', 'PATIENT', 'DESCRIPTION'])
        for index, row in df.iterrows():
            facts.append(PatientFact(
                fact_id=f"procedures_{index}",
                patient_id=row['PATIENT'],
                event_date=row['DATE'],
                event_type="Procedure",
                description=row['DESCRIPTION'],
                source_file="procedures.csv"
            ))
        print(f"Loaded {len(facts)} facts from procedures.csv")
    except FileNotFoundError:
        print("procedures.csv not found.")
    return facts

def load_all_data(data_path):
    all_facts = []
    all_facts.extend(load_conditions(data_path))
    all_facts.extend(load_medications(data_path))
    all_facts.extend(load_allergies(data_path))
    all_facts.extend(load_observations(data_path))
    all_facts.extend(load_procedures(data_path))
    print(f"Total facts loaded: {len(all_facts)}")
    return all_facts

# --- 3. The Main Pathway Pipeline ---
def run_indexing_pipeline(api_key):
    all_patient_facts = load_all_data("./patient")
    facts_table = pw.table.from_list(all_patient_facts)
    
    embedder = OpenAIEmbedder(api_key=api_key)

    embedded_table = facts_table.select(
        embedding=embedder(facts_table.description),
        data=facts_table
    )

    # --- FUNCTION CALL (CORRECTED) ---
    index = build_vector_store_index(
        embedded_table,
        "embedding"
    )

    pwp.write(index, "patient_index")
    
    print("\n✅ Success! Patient index has been built and saved to 'patient_index' folder.")


if __name__ == "__main__":
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set before.")

    run_indexing_pipeline(api_key=openai_api_key)