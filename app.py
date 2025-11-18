import os
import streamlit as st
import pandas as pd
import openai
import numpy as np
from scipy.spatial.distance import cdist
from dataclasses import dataclass

# --- 1. Configurations ---
try:
    # Try getting from environment, otherwise look for Streamlit secrets or manual entry
    API_KEY = os.environ.get("OPENAI_API_KEY")
    if not API_KEY:
        # Fallback for local testing if env var isn't set
        # You can replace this string with your key for testing, but don't commit it!
        API_KEY = "sk-..." 
        
    if not API_KEY or "sk-" not in API_KEY:
        st.error("OPENAI_API_KEY is missing. Please run: `export OPENAI_API_KEY='sk-...'` in your terminal.", icon="🚨")
        st.stop()
        
    client = openai.OpenAI(api_key=API_KEY)
except Exception as e:
    st.error(f"Configuration Error: {e}", icon="🚨")
    st.stop()

SPECIALIST_PROFILES = {
    'Cardiologist': [
        "heart conditions", "chest pain", "palpitations", "shortness of breath",
        "high blood pressure", "hypertension", "cholesterol levels", "lipid panel",
        "ECG", "Echocardiogram", "stress test", "heart medications",
        "statin", "beta-blocker", "family history of heart disease"
    ],
    'Dermatologist': [
        "skin conditions", "rash", "lesions", "moles", "itchy skin",
        "acne", "eczema", "psoriasis", "dermatitis",
        "skin allergies", "autoimmune skin disorders", "biopsy results",
        "topical medications", "creams", "ointments"
    ],
    'Neurologist': [
        "headaches", "migraines", "seizures", "epilepsy", "dizziness",
        "numbness", "tingling", "weakness", "memory loss", "confusion",
        "stroke", "multiple sclerosis", "Parkinson's disease",
        "brain MRI", "CT scan of head", "EEG results", "neurological medications"
    ],
    'Immunologist': [
        "allergies", "allergic rhinitis", "asthma", "hives", "anaphylaxis",
        "recurrent infections", "autoimmune disorders", "lupus",
        "rheumatoid arthritis", "immunodeficiency", "vaccination history",
        "allergy testing results", "immunotherapy"
    ],
    'Ophthalmologist': [
        "vision problems", "blurry vision", "double vision", "eye pain",
        "glaucoma", "cataracts", "macular degeneration", "diabetic retinopathy",
        "eye exam results", "visual acuity", "eye medications", "glasses prescription"
    ]
}

PROMPT_TEMPLATE = """
You are an expert clinical summarizer. Generate a concise, narrative summary of the patient's relevant medical history, specifically tailored for a **{specialist_type}**. Do not use bullet points or fixed sections that return 'No relevant records found.' Only create sections where data is present.

**Instructions:**
1.  Use **only** the 'Patient Facts' provided below. Do not make up information.
2.  Organize the summary into the following narrative sections, but ONLY include a section if there are facts to support it:
    * **Current Health Overview:** A brief opening statement.
    * **Relevant Diagnoses & History:** Focus on chronic conditions, hospitalizations, and findings most relevant to a **{specialist_type}**.
    * **Active Medications & Allergies:** List ALL medications and ALL known allergies clearly.
    * **Relevant Lab & Procedure Highlights:** Summarize key tests.
3.  **CRITICAL:** For every single statement, you **MUST** provide a citation to its source and date. Example: `(Source: conditions.csv, 2020-05-10)`.
4.  Keep the entire summary to a maximum of 300 words.

---
**Patient Facts:**
{context}
---

**Begin Summary:**
"""
# --- 2. Data Loading & Index Building ---

@dataclass
class PatientFact:
    fact_id: str
    patient_id: str
    event_date: str
    event_type: str
    description: str
    source_file: str

def load_all_data(data_path):
    all_facts = []
    # Increased limit slightly to ensure we get enough data for a demo
    DATA_LIMIT = 1000 
    
    # Helper to handle safe string conversion
    def safe_str(val):
        return str(val) if pd.notna(val) else "Unknown"

    # --- Load Conditions ---
    try:
        df = pd.read_csv(f"{data_path}/conditions.csv", usecols=['START', 'PATIENT', 'DESCRIPTION'], nrows=DATA_LIMIT)
        for i, row in df.iterrows():
            all_facts.append(PatientFact(
                fact_id=f"cond_{i}", 
                patient_id=safe_str(row['PATIENT']), 
                event_date=safe_str(row['START']), 
                event_type="Condition", 
                description=safe_str(row['DESCRIPTION']), 
                source_file="conditions.csv"
            ))
    except FileNotFoundError: pass

    # --- Load Medications ---
    try:
        df = pd.read_csv(f"{data_path}/medications.csv", usecols=['START', 'PATIENT', 'DESCRIPTION'], nrows=DATA_LIMIT)
        for i, row in df.iterrows():
            all_facts.append(PatientFact(
                fact_id=f"med_{i}", 
                patient_id=safe_str(row['PATIENT']), 
                event_date=safe_str(row['START']), 
                event_type="Medication", 
                description=safe_str(row['DESCRIPTION']), 
                source_file="medications.csv"
            ))
    except FileNotFoundError: pass

    # --- Load Allergies ---
    try:
        df = pd.read_csv(f"{data_path}/allergies.csv", usecols=['START', 'PATIENT', 'DESCRIPTION'], nrows=DATA_LIMIT)
        for i, row in df.iterrows():
            all_facts.append(PatientFact(
                fact_id=f"alg_{i}", 
                patient_id=safe_str(row['PATIENT']), 
                event_date=safe_str(row['START']), 
                event_type="Allergy", 
                description=safe_str(row['DESCRIPTION']), 
                source_file="allergies.csv"
            ))
    except FileNotFoundError: pass

    # --- Load Observations ---
    try:
        df = pd.read_csv(f"{data_path}/observations.csv", usecols=['DATE', 'PATIENT', 'DESCRIPTION', 'VALUE', 'UNITS'], nrows=DATA_LIMIT)
        df = df.dropna(subset=['VALUE'])
        for i, row in df.iterrows():
            desc = f"{row['DESCRIPTION']}: {row['VALUE']} {row['UNITS'] or ''}".strip()
            all_facts.append(PatientFact(
                fact_id=f"obs_{i}", 
                patient_id=safe_str(row['PATIENT']), 
                event_date=safe_str(row['DATE']), 
                event_type="Observation", 
                description=desc, 
                source_file="observations.csv"
            ))
    except FileNotFoundError: pass

    # --- Load Procedures ---
    try:
        df = pd.read_csv(f"{data_path}/procedures.csv", usecols=['DATE', 'PATIENT', 'DESCRIPTION'], nrows=DATA_LIMIT)
        for i, row in df.iterrows():
            all_facts.append(PatientFact(
                fact_id=f"proc_{i}", 
                patient_id=safe_str(row['PATIENT']), 
                event_date=safe_str(row['DATE']), 
                event_type="Procedure", 
                description=safe_str(row['DESCRIPTION']), 
                source_file="procedures.csv"
            ))
    except FileNotFoundError: pass
    
    return all_facts

@st.cache_resource
def build_live_index():
    status_container = st.empty()
    status_container.info("Loading patient data... (First run may take 10s)")
    
    all_facts = load_all_data("./patient")
    
    if not all_facts:
        st.error("No data loaded. Check your 'patient' folder.", icon="🚨")
        st.stop()

    descriptions = [fact.description for fact in all_facts]
    
    status_container.info(f"Embedding {len(descriptions)} records via OpenAI...")
    progress_bar = st.progress(0)
    
    try:
        batch_size = 200
        all_embeddings = []
        
        for i in range(0, len(descriptions), batch_size):
            progress_bar.progress(min(i / len(descriptions), 1.0))
            batch = descriptions[i:i+batch_size]
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        progress_bar.empty()
        status_container.success("✅ System Ready")
        
        all_embeddings_array = np.array(all_embeddings)
        return all_facts, all_embeddings_array
    
    except Exception as e:
        st.error(f"Error calling OpenAI Embeddings: {e}", icon="🚨")
        st.stop()

# --- 3. The Core AI Logic (FIXED) ---

def generate_summary(patient_id, specialist_type, all_facts, all_embeddings):
    
    # 1. Filter Data FIRST (The Fix)
    # We isolate only the facts and embeddings for THIS patient
    patient_indices = [i for i, fact in enumerate(all_facts) if fact.patient_id == patient_id]
    
    if not patient_indices:
        return "Error: No records found for this Patient ID in the loaded dataset."
    
    patient_facts = [all_facts[i] for i in patient_indices]
    patient_embeddings = all_embeddings[patient_indices]

    # 2. Embed Specialist Queries
    with st.spinner(f"Analyzing {len(patient_facts)} records for {specialist_type} relevance..."):
        query_texts = SPECIALIST_PROFILES.get(specialist_type, [])
        try:
            response = client.embeddings.create(
                input=query_texts,
                model="text-embedding-ada-002"
            )
            query_embeddings = np.array([item.embedding for item in response.data])
        except Exception as e:
            return f"OpenAI API Error: {e}"

    # 3. Semantic Search (Within Patient's Data)
    # Calculate distance between queries and patient's specific records
    distances = cdist(query_embeddings, patient_embeddings, 'cosine')
    
    # Find the best score for each fact against ANY of the specialist queries
    # (axis=0 gives the minimum distance for each fact column)
    min_distances = np.min(distances, axis=0)
    
    # Get top 30 most relevant facts for this specialist
    # (Or take everything if there are fewer than 30 facts total)
    k = min(30, len(patient_facts))
    nearest_indices = np.argsort(min_distances)[:k]
    
    relevant_facts = [patient_facts[i] for i in nearest_indices]

    # 4. Prepare Context
    relevant_facts.sort(key=lambda x: x.event_date, reverse=True)
    context_lines = []
    for fact in relevant_facts:
        line = f"Source: {fact.source_file} | Date: {fact.event_date} | Type: {fact.event_type} | Text: {fact.description}"
        context_lines.append(line)
    context = "\n".join(context_lines)

    # 5. LLM Generation
    with st.spinner("Generating summary..."):
        try:
            chat_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful clinical assistant."},
                    {"role": "user", "content": PROMPT_TEMPLATE.format(
                        specialist_type=specialist_type,
                        context=context
                    )}
                ]
            )
            return chat_response.choices[0].message.content
        except Exception as e:
            return f"Error from LLM: {e}"

# --- 4. UI ---

def main():
    st.set_page_config(page_title="Patient Summary Assistant", layout="wide")
    st.title("🧑‍⚕️ Patient Summary Assistant")
    st.markdown("### AI-Powered Specialist Handover Tool")

    # Load Data
    all_facts, all_embeddings = build_live_index()
    
    # Get Valid IDs
    unique_patient_ids = sorted(list(set(fact.patient_id for fact in all_facts)))
    
    # Sidebar
    st.sidebar.header("Consultation Details")
    
    # UI CHANGE: Dropdown of Valid IDs
    selected_patient_id = st.sidebar.selectbox(
        "Select Patient ID",
        options=unique_patient_ids,
        help="Only patients with records in the loaded sample are shown."
    )
    
    selected_specialist = st.sidebar.selectbox(
        "Select Target Specialist",
        options=list(SPECIALIST_PROFILES.keys())
    )

    st.sidebar.divider()
    
    # Generate Button
    if st.sidebar.button("Generate Summary", type="primary"):
        st.session_state.summary = generate_summary(
            selected_patient_id,
            selected_specialist,
            all_facts,
            all_embeddings
        )

    # Display Results
    if 'summary' in st.session_state:
        st.divider()
        st.subheader(f"Medical Summary: {selected_specialist}")
        st.caption(f"Patient ID: {selected_patient_id}")
        st.markdown(st.session_state.summary)
        
        # Optional: Show raw data for debugging/demo purposes
        with st.expander("View Source Records Used"):
            st.write("The AI used the most relevant records from the index to generate this summary.")

if __name__ == "__main__":
    main()