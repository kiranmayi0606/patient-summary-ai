import os
import streamlit as st
import pandas as pd
import openai
import numpy as np
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from dotenv import load_dotenv

# --- 1. Configurations ---
load_dotenv()

try:
    # Try getting from environment, otherwise look for Streamlit secrets or manual entry
    API_KEY = os.environ.get("OPENAI_API_KEY")
    if not API_KEY:
        # Fallback for local testing if env var isn't set
        API_KEY = "" 
        
    if not API_KEY or "sk-" not in API_KEY:
        # Check environment first, then secrets/manual fallback
        if "OPENAI_API_KEY" in st.secrets:
            API_KEY = st.secrets["OPENAI_API_KEY"]
        elif not API_KEY:
             st.error("OPENAI_API_KEY is missing. Please set it in your environment or .env file.", icon="🚨")
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
1. Use **only** the 'Patient Facts' provided below. Do not make up information.
2. Organize the summary into the following narrative sections, but ONLY include a section if there are facts to support it:
    * **Current Health Overview:** A brief opening statement.
    * **Relevant Diagnoses & History:** Focus on chronic conditions, hospitalizations, and findings most relevant to a **{specialist_type}**.
    * **Active Medications & Allergies:** List ALL medications and ALL known allergies clearly.
    * **Relevant Lab & Procedure Highlights:** Summarize key tests.
3. **CRITICAL:** For every single statement, you **MUST** provide a citation to its source and date. Example: `(Source: conditions.csv, 2020-05-10)`.
4. Keep the entire summary to a maximum of 300 words.

---
**Patient Facts:**
{context}
---

**Begin Summary:**
"""

QA_PROMPT_TEMPLATE = """
You are a helpful clinical assistant. Your goal is to answer the user's question by synthesizing the patient's specific medical history with general medical knowledge.

**Instructions:**
1. **Grounding:** Base your answer primarily on the 'Patient Context' provided below.
2. **Reasoning:** If the user asks for recommendations, risks, or assessments (e.g., "what labs?", "stroke risk?"), use the patient's diagnoses and medications from the context to inform your medical reasoning.
3. **Differentiation:** Clearly distinguish between **facts found in the record** (cite them) and **general medical guidelines** (state "Based on standard guidelines...").
4. **Safety:** If you lack sufficient patient history to make a safe assessment, state that clearly.

---
**User Question:** {question}

**Patient Context:**
{context}
---

**Answer:**
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
    DATA_LIMIT = 1000 
    
    def safe_str(val):
        return str(val) if pd.notna(val) else "Unknown"

    def append_facts_from_csv(filename, type_label, date_col, other_cols):
        try:
            cols_to_use = [date_col, 'PATIENT', 'DESCRIPTION'] + other_cols
            df = pd.read_csv(f"{data_path}/{filename}", usecols=cols_to_use, nrows=DATA_LIMIT)
            
            if 'VALUE' in other_cols:
                df = df.dropna(subset=['VALUE'])
            
            for i, row in df.iterrows():
                desc = safe_str(row['DESCRIPTION'])
                if 'VALUE' in other_cols:
                    desc = f"{desc}: {row['VALUE']} {row.get('UNITS', '') or ''}".strip()
                    
                all_facts.append(PatientFact(
                    fact_id=f"{filename[:3]}_{i}", 
                    patient_id=safe_str(row['PATIENT']), 
                    event_date=safe_str(row[date_col]), 
                    event_type=type_label, 
                    description=desc, 
                    source_file=filename
                ))
        except FileNotFoundError: pass

    # Ensure your data folder is named 'patient' and is in the same directory
    append_facts_from_csv("conditions.csv", "Condition", 'START', [])
    append_facts_from_csv("medications.csv", "Medication", 'START', [])
    append_facts_from_csv("allergies.csv", "Allergy", 'START', [])
    append_facts_from_csv("observations.csv", "Observation", 'DATE', ['VALUE', 'UNITS'])
    append_facts_from_csv("procedures.csv", "Procedure", 'DATE', [])
    
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

# --- 3. Reusable Retrieval Logic ---

def get_relevant_context(patient_id, query_texts, all_facts, all_embeddings, top_k=30):
    """
    Retrieves the most semantically relevant facts for a given patient and query.
    """
    # 1. Filter Data FIRST
    patient_indices = [i for i, fact in enumerate(all_facts) if fact.patient_id == patient_id]
    
    if not patient_indices:
        return "Error: No records found for this Patient ID.", None
    
    patient_facts = [all_facts[i] for i in patient_indices]
    patient_embeddings = all_embeddings[patient_indices]

    # 2. Embed Query Texts
    if isinstance(query_texts, str):
        query_texts = [query_texts]
        
    try:
        response = client.embeddings.create(
            input=query_texts,
            model="text-embedding-ada-002"
        )
        query_embeddings = np.array([item.embedding for item in response.data])
    except Exception as e:
        return f"OpenAI API Error during embedding: {e}", None

    # 3. Semantic Search
    distances = cdist(query_embeddings, patient_embeddings, 'cosine')
    min_distances = np.min(distances, axis=0)
    
    # Get top K most relevant facts
    k = min(top_k, len(patient_facts))
    nearest_indices = np.argsort(min_distances)[:k]
    
    relevant_facts = [patient_facts[i] for i in nearest_indices]

    # 4. Prepare Context String
    relevant_facts.sort(key=lambda x: x.event_date, reverse=True)
    context_lines = []
    for fact in relevant_facts:
        line = f"Source: {fact.source_file} | Date: {fact.event_date} | Type: {fact.event_type} | Text: {fact.description}"
        context_lines.append(line)
    context = "\n".join(context_lines)
    
    return context, relevant_facts

# --- 4. Core AI Logic (Summary Generation) ---

def generate_summary(patient_id, specialist_type, all_facts, all_embeddings):
    
    with st.spinner(f"Analyzing records for {specialist_type} relevance and generating summary..."):
        
        # Retrieve Context using specialist profile keywords
        query_texts = SPECIALIST_PROFILES.get(specialist_type, [])
        context, relevant_facts = get_relevant_context(patient_id, query_texts, all_facts, all_embeddings, top_k=30)
        
        if "Error" in context:
             return context, []
             
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
            return chat_response.choices[0].message.content, relevant_facts
        except Exception as e:
            return f"Error from LLM: {e}", []

# --- 5. Core AI Logic (Q&A Generation - Reasoning Enabled) ---

def rag_qa_response(patient_id, question, all_facts, all_embeddings):
    """Handles the RAG pipeline for a single user question with hybrid retrieval."""
    
    with st.spinner("Analyzing patient history and formulating answer..."):
        
        # A. Get Core Context (Conditions/Meds) - Critical for reasoning
        # We force the retrieval of diagnoses and meds regardless of the user's specific words
        core_context_str, _ = get_relevant_context(
            patient_id, 
            ["Active medical conditions diagnoses", "Current medications list"], 
            all_facts, 
            all_embeddings, 
            top_k=15
        )
        
        # B. Get Question-Specific Context
        # We search for the specific topic (e.g., "labs", "stroke risk")
        specific_context_str, specific_facts = get_relevant_context(
            patient_id, 
            question, 
            all_facts, 
            all_embeddings, 
            top_k=10
        )
        
        # Fallback: If we found absolutely nothing, stop here.
        if not specific_facts and not core_context_str:
             return "I cannot find any records relevant to that question in the patient's available history."
        
        # Combine them into one context block
        full_context = f"""
        --- CORE PATIENT HISTORY (Diagnoses & Meds) ---
        {core_context_str}
        
        --- RELEVANT RECORDS FOR QUESTION ---
        {specific_context_str}
        """

        # 2. LLM Generation
        try:
            chat_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful clinical Q&A assistant."},
                    {"role": "user", "content": QA_PROMPT_TEMPLATE.format(
                        question=question,
                        context=full_context
                    )}
                ]
            )
            return chat_response.choices[0].message.content
        except Exception as e:
            return f"Error from LLM: {e}"


# --- 6. UI ---

def main():
    st.set_page_config(page_title="Patient Summary Assistant", layout="wide")
    st.title("🧑‍⚕️ Patient Summary & Q&A Assistant")
    st.markdown("### AI-Powered Specialist Handover Tool")

    # Initialize Session State
    if 'messages' not in st.session_state: st.session_state.messages = []
    if 'current_patient_id' not in st.session_state: st.session_state.current_patient_id = None
    
    # Load Data
    all_facts, all_embeddings = build_live_index()
    unique_patient_ids = sorted(list(set(fact.patient_id for fact in all_facts)))
    
    # --- Sidebar ---
    st.sidebar.header("Consultation Details")
    
    selected_patient_id = st.sidebar.selectbox(
        "Select Patient ID",
        options=unique_patient_ids,
        help="Only patients with records in the loaded sample are shown."
    )
    
    selected_specialist = st.sidebar.selectbox(
        "Select Target Specialist",
        options=list(SPECIALIST_PROFILES.keys())
    )

    # Reset context if patient changes
    if st.session_state.current_patient_id != selected_patient_id:
        st.session_state.messages = []
        if 'summary' in st.session_state: del st.session_state.summary
        if 'summary_facts' in st.session_state: del st.session_state.summary_facts
        st.session_state.current_patient_id = selected_patient_id

    st.sidebar.divider()
    
    if st.sidebar.button("Generate Summary", type="primary"):
        # --- THE FIX: Clear chat history when regenerating summary ---
        st.session_state.messages = [] 
        
        summary, facts_used = generate_summary(
            selected_patient_id,
            selected_specialist,
            all_facts,
            all_embeddings
        )
        st.session_state.summary = summary
        st.session_state.summary_facts = facts_used
        
        # Force a refresh so the chat history visually disappears immediately
        st.rerun()

    # --- Main Display Area ---
    
    # 1. Display Summary Results 
    if 'summary' in st.session_state:
        st.divider()
        st.subheader(f"Medical Summary for: {selected_specialist}")
        st.caption(f"Patient ID: {selected_patient_id}")
        st.markdown(st.session_state.summary)
        
        with st.expander("View Source Records Used for Summary"):
            if 'summary_facts' in st.session_state and st.session_state.summary_facts:
                for fact in st.session_state.summary_facts:
                    st.text(f"[{fact.event_date}] {fact.event_type} ({fact.source_file}): {fact.description}")
            else:
                 st.write("No source records available.")
    
    # 2. Q&A Interface
    st.divider()
    st.subheader("❓ Ask a Patient Question (RAG Chat)")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about specific medications, lab results, or diagnoses..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_text = rag_qa_response(selected_patient_id, prompt, all_facts, all_embeddings)
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()