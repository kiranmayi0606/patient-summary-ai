import os
import streamlit as st
import pandas as pd
import openai
import numpy as np
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from dotenv import load_dotenv

# --- 1. Configurations and Prompts ---
load_dotenv()

# --- API KEY Initialization ---
try:
    API_KEY = os.environ.get("OPENAI_API_KEY")
    
    # Check for environment, then secrets
    if not API_KEY or "sk-" not in API_KEY:
        if "OPENAI_API_KEY" in st.secrets:
            API_KEY = st.secrets["OPENAI_API_KEY"]
        elif not API_KEY:
             st.error("OPENAI_API_KEY is missing. Please set it in your environment or .env file.", icon="🚨")
             st.stop()
        
    client = openai.OpenAI(api_key=API_KEY)
except Exception as e:
    st.error(f"Configuration Error: {e}", icon="🚨")
    st.stop()

# --- SPECIALIST PROFILES (Used for Summary Generation) ---
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

# --- PROMPT TEMPLATE FOR SUMMARY GENERATION (Static Report) ---
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

# --- PROMPT TEMPLATE FOR CHAT Q&A (Reasoning Enabled) ---
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

# --- 2. Data Loading and Structuring ---

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
        except KeyError as e: print(f"Skipping {filename}: Missing column {e}") # Handle case where a column might be missing

    # Load data from various files using the reusable function
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

# --- 3. Core AI Logic and Retrieval Functions ---

def get_relevant_context(patient_id, query_texts, all_facts, all_embeddings, top_k=15):
    """Retrieves the most semantically relevant facts for a given patient and query."""
    
    patient_indices = [i for i, fact in enumerate(all_facts) if fact.patient_id == patient_id]
    if not patient_indices:
        return "Error: No records found for this Patient ID in the loaded dataset.", None
    
    patient_facts = [all_facts[i] for i in patient_indices]
    patient_embeddings = all_embeddings[patient_indices]

    if isinstance(query_texts, str):
        query_texts = [query_texts]
        
    try:
        response = client.embeddings.create(input=query_texts, model="text-embedding-ada-002")
        query_embeddings = np.array([item.embedding for item in response.data])
    except Exception as e:
        return f"OpenAI API Error during embedding: {e}", None

    # Semantic Search using SciPy
    distances = cdist(query_embeddings, patient_embeddings, 'cosine')
    min_distances = np.min(distances, axis=0)
    
    k = min(top_k, len(patient_facts))
    nearest_indices = np.argsort(min_distances)[:k]
    
    relevant_facts = [patient_facts[i] for i in nearest_indices]

    # Prepare Context String
    relevant_facts.sort(key=lambda x: x.event_date, reverse=True)
    context_lines = []
    for fact in relevant_facts:
        line = f"Source: {fact.source_file} | Date: {fact.event_date} | Type: {fact.event_type} | Text: {fact.description}"
        context_lines.append(line)
    context = "\n".join(context_lines)
    
    return context, relevant_facts

def generate_summary(patient_id, specialist_type, all_facts, all_embeddings):
    """Generates the static, cited summary report."""
    
    with st.spinner(f"Analyzing records for {specialist_type} relevance and generating summary..."):
        
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

def rag_qa_response(patient_id, question, all_facts, all_embeddings):
    """Handles the RAG pipeline for a single user question (Hybrid Reasoning)."""
    
    with st.spinner("Analyzing patient history and formulating answer..."):
        
        # A. Get Core Context (Conditions/Meds) - Forced retrieval for medical reasoning base
        core_context_str, _ = get_relevant_context(
            patient_id, 
            ["Active medical conditions diagnoses", "Current medications list"], 
            all_facts, 
            all_embeddings, 
            top_k=15
        )
        
        # B. Get Question-Specific Context
        specific_context_str, _ = get_relevant_context(
            patient_id, 
            question, 
            all_facts, 
            all_embeddings, 
            top_k=10
        )
        
        # Combine contexts
        full_context = f"""
--- CORE PATIENT HISTORY (Diagnoses & Meds) ---
{core_context_str}
--- RELEVANT RECORDS FOR QUESTION ---
{specific_context_str}
"""

        # Fallback if no data is retrieved
        if "Error" in core_context_str and "Error" in specific_context_str:
             return "I cannot find any records relevant to that question in the patient's available history."

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

def create_download_content(summary, patient_id, specialist, source_facts):
    """Formats the entire report content into an HTML string for clean PDF printing."""
    
    source_html = "<h4>Source Records Used:</h4><ul>"
    if source_facts:
        for fact in source_facts:
            source_html += f"<li>**{fact.event_type}** ({fact.event_date}) from *{fact.source_file}*: {fact.description}</li>"
    else:
        source_html += "<li>No specific facts were retrieved for this summary.</li>"
    source_html += "</ul>"
    
    # Final HTML Structure
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Summary - {patient_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
            h1 {{ border-bottom: 2px solid #ccc; padding-bottom: 5px; }}
            h4 {{ margin-top: 20px; color: #555; }}
            .summary-content {{ margin-top: 15px; border: 1px solid #eee; padding: 15px; border-radius: 8px; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin-bottom: 5px; border-left: 3px solid #007bff; padding-left: 10px; }}
        </style>
    </head>
    <body>
        <h1>Medical Summary: {specialist}</h1>
        <p><strong>Patient ID:</strong> {patient_id}</p>
        <p><strong>Generated By:</strong> AI Summary Agent</p>
        
        <h4>AI Generated Narrative Summary:</h4>
        <div class="summary-content">
            {summary}
        </div>
        
        {source_html}
    </body>
    </html>
    """
    return html_content


# --- 4. UI: Main Application ---

def main():
    st.set_page_config(page_title="Patient Summary Assistant", layout="wide")
    st.title("🧑‍⚕️ Patient Summary & Q&A Assistant")
    st.markdown("### AI-Powered Specialist Handover Tool")
    
    # Initialize Session State
    if 'messages' not in st.session_state: st.session_state.messages = []
    if 'current_patient_id' not in st.session_state: st.session_state.current_patient_id = None
    
    # Summary Report State
    if 'summary' not in st.session_state: st.session_state.summary = None
    if 'summary_facts' not in st.session_state: st.session_state.summary_facts = None
    
    # Load Data
    all_facts, all_embeddings = build_live_index()
    unique_patient_ids = sorted(list(set(fact.patient_id for fact in all_facts)))
    
    # --- Sidebar for Context Selection ---
    st.sidebar.header("Consultation Context")
    
    selected_patient_id = st.sidebar.selectbox(
        "Select Patient ID",
        options=unique_patient_ids,
        help="Only patients with records in the loaded sample are shown."
    )
    
    selected_specialist = st.sidebar.selectbox(
        "Target Specialist",
        options=list(SPECIALIST_PROFILES.keys())
    )

    # Update patient context if selection changes
    if st.session_state.current_patient_id != selected_patient_id:
        st.session_state.messages = []
        st.session_state.summary = None
        st.session_state.summary_facts = None
        st.session_state.current_patient_id = selected_patient_id
        st.session_state.messages.append({"role": "assistant", "content": f"Context updated to Patient ID: **{selected_patient_id}**. Use the button to generate the report."})

    st.sidebar.divider()
    
    # --- Generate Summary Button ---
    if st.sidebar.button(f"Generate {selected_specialist} Summary", type="primary"):
        # Clear chat history when regenerating summary
        st.session_state.messages = [] 
        
        summary, facts_used = generate_summary(
            selected_patient_id,
            selected_specialist,
            all_facts,
            all_embeddings
        )
        st.session_state.summary = summary
        st.session_state.summary_facts = facts_used
        
        # Display the summary and then jump to the Q&A section
        st.rerun()

    # --- Main Display Area ---
    
    # 1. Display Summary Results 
    if st.session_state.summary:
        st.divider()
        st.subheader(f"✅ Specialist Report: {selected_specialist}")
        st.caption(f"Patient ID: {selected_patient_id}")
        
        st.markdown(st.session_state.summary)
        
        # Download Button
        download_content = create_download_content(
            st.session_state.summary, 
            selected_patient_id, 
            selected_specialist, 
            st.session_state.summary_facts
        )
        st.download_button(
            label="⬇️ Download Report (HTML for PDF)",
            data=download_content,
            file_name=f"Summary_{selected_patient_id}_{selected_specialist}.html",
            mime="text/html"
        )
        
        # Source Expander
        with st.expander("View Source Records Used for Summary"):
            if st.session_state.summary_facts:
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
