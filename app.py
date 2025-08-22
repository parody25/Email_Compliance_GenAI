import streamlit as st
import pandas as pd
import json
import math
from openai import OpenAI
import os
from dotenv import load_dotenv
import traceback
import datetime
import re
import ast
import io

load_dotenv()
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY_GPT4o")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_ID = "groupe-77d4b91e-b82f-4632-802b-158a6dbb4176"

client = OpenAI(
    api_key=AZURE_API_KEY,
    base_url=f"{AZURE_ENDPOINT}/openai/deployments/{DEPLOYMENT_ID}",
    default_query={"api-version": "2025-01-01-preview"},
    default_headers={"api-key": AZURE_API_KEY}
)

def calculate_priority(row):
    score = (
        row["RegulatoryImpact"] * 0.4 +
        row["BusinessImpact"] * 0.25 +
        row["ReputationRisk"] * 0.2 +
        row["Frequency"] * 0.15
    )
    if score >= 3.8:
        priority = "High"
    elif score >= 2.8:
        priority = "Medium"
    else:
        priority = "Low"
    return round(score, 2), priority

def chunk_emails(df, chunk_size=10):
    """
    Strictly chunk emails into batches of size `chunk_size`.
    Each batch will contain exactly `chunk_size` rows,
    except possibly the last one if rows are not divisible.
    """
    total = len(df)
    for i in range(0, total, chunk_size):
        batch = df.iloc[i:i+chunk_size]
        print(f"[DEBUG] Creating batch from row {i} to {i+len(batch)-1}, size={len(batch)}")
        yield batch

def clean_and_parse_json(text: str):
    if not text:
        return []

    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())

    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        pyish = text.replace("true", "True").replace("false", "False").replace("null", "None")
        parsed = ast.literal_eval(pyish)
        return parsed
    except Exception:
        return []

def analyze_emails_llm(email_group, group_index):
    try:
        examples = [
            {
                "SerialNo.": 1, "IsCompliant": True, "Category": "None",
                "Reason": "Routine meeting scheduling with no compliance risks.",
                "SourceLines": "Let's meet at 3 PM tomorrow to discuss project updates.",
                "RegulatoryImpact": 1, "BusinessImpact": 1, "ReputationRisk": 1, "Frequency": 1
            },
            {
                "SerialNo.": 2, "IsCompliant": False, "Category": "Secrecy",
                "Reason": "Sharing confidential client financial data with unauthorized recipients.",
                "SourceLines": "Attached is the client’s confidential financial report.",
                "RegulatoryImpact": 5, "BusinessImpact": 4, "ReputationRisk": 5, "Frequency": 3
            },
            {
                "SerialNo.": 3, "IsCompliant": False, "Category": "Market Manipulation/Misconduct",
                "Reason": "The email suggests spreading false information to influence stock price.",
                "SourceLines": "Let’s leak a rumor to drive up demand before earnings.",
                "RegulatoryImpact": 5, "BusinessImpact": 5, "ReputationRisk": 5, "Frequency": 2
            },
            {
                "SerialNo.": 4, "IsCompliant": False, "Category": "Market Bribery",
                "Reason": "The email proposes offering a gift to gain preferential treatment.",
                "SourceLines": "We should send him expensive tickets so he approves the deal.",
                "RegulatoryImpact": 4, "BusinessImpact": 4, "ReputationRisk": 5, "Frequency": 2
            },
            {
                "SerialNo.": 5, "IsCompliant": False, "Category": "Change in communication",
                "Reason": "The sender requests to shift to unofficial communication channels.",
                "SourceLines": "Let’s take this discussion to WhatsApp instead.",
                "RegulatoryImpact": 3, "BusinessImpact": 2, "ReputationRisk": 3, "Frequency": 2
            },
            {
                "SerialNo.": 6, "IsCompliant": False, "Category": "Complaints",
                "Reason": "The email contains a formal complaint from a client regarding service quality.",
                "SourceLines": "I am extremely dissatisfied with how my request was handled.",
                "RegulatoryImpact": 3, "BusinessImpact": 4, "ReputationRisk": 4, "Frequency": 3
            },
            {
                "SerialNo.": 7, "IsCompliant": False, "Category": "Employee ethics",
                "Reason": "The email shows a conflict of interest by promoting a family business.",
                "SourceLines": "Please consider using my cousin’s company for this project.",
                "RegulatoryImpact": 2, "BusinessImpact": 3, "ReputationRisk": 3, "Frequency": 2
            }
        ]

        prompt = f"""
You are a compliance analyst. Analyze emails for compliance risks.

Each email has a unique "SerialNo.". Always include it unchanged in your output.

Scoring (1=very low risk, 5=very high risk):
- RegulatoryImpact
- BusinessImpact
- ReputationRisk
- Frequency

Also return:
- IsCompliant: true/false
- Category: ["Secrecy", "Market Manipulation/Misconduct", "Market Bribery", "Change in communication", "Complaints", "Employee ethics"]
- Reason: Short, specific explanation.
- SourceLines: Key phrases that triggered the decision.

⚠️ Do NOT mark every email as non-compliant.
⚠️ Benign or routine emails should be marked compliant with very low scores.
⚠️ Only mark non-compliant when there is a clear compliance issue.
⚠️ The Category should fall into the 6 categories mentioned above or it should be None of them.

Return ONLY a valid JSON array. 
No explanations, no markdown, no code fences.

Here are some examples of expected format:
{json.dumps(examples, indent=2)}

Now analyze these emails:
{json.dumps(email_group.to_dict(orient="records"), indent=2)}
"""

        st.write(f"🔹 Sending group {group_index+1} with {len(email_group)} emails to LLM...")
        print(f"[DEBUG] Group {group_index+1} prompt:\n", prompt[:500], "...")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=4000
        )

        raw_output = response.choices[0].message.content or ""
        print(f"[DEBUG] Raw LLM output for group {group_index+1}:\n", raw_output)

        parsed_json = clean_and_parse_json(raw_output)

        if not parsed_json:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            err_file = f"llm_error_group_{group_index+1}_{ts}.txt"
            with open(err_file, "w", encoding="utf-8") as f:
                f.write(raw_output)
            print(f"[ERROR] JSON parsing failed for group {group_index+1}. Saved raw output to {err_file}")
            return []

        return parsed_json

    except Exception as e:
        print(f"[ERROR] API call failed for group {group_index+1}: {e}")
        print(traceback.format_exc())
        return []

st.title("📧 Email Compliance Analyzer with Priority Matrix")

uploaded_file = st.file_uploader("Upload Excel with Emails", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Uploaded Data:", df.head(10))

    if not all(col in df.columns for col in ["Date", "From", "To", "Subject", "Body"]):
        st.error("Uploaded file must have columns: Date, From, To, Subject, Body")
    else:
        all_results = []

        with st.spinner("Analyzing emails..."):
            for idx, group_df in enumerate(chunk_emails(df, chunk_size=10)):
                st.write(f"📦 Processing batch {idx+1} with {len(group_df)} emails")
                result = analyze_emails_llm(group_df, idx)
                all_results.extend(result)

        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df["PriorityScore"], results_df["Priority"] = zip(*results_df.apply(calculate_priority, axis=1))

            st.success("Analysis Completed!")
            st.dataframe(results_df.sort_values(by="PriorityScore", ascending=False))

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                results_df.to_excel(writer, index=False, sheet_name="Compliance_Results")

            st.download_button(
                label="Download Results Excel",
                data=excel_buffer.getvalue(),
                file_name="email_compliance_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error("No results were generated. Check debug logs in the terminal for details.")
