from typing import TypedDict, List, Optional, Dict, Any
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

LLM = os.getenv("GEMINI_API_KEY")

DEFAULT_LLM = "gemini-2.5-flash"

class Metadata(BaseModel):
    
    "Standard header metadata for incoming medical reports, provides tracking, HIPAA target names, and safety flags"
    category: str = Field(default="medical_report", description="Document type")
    report_id: str = Field(..., description="Unique report identifier")
    patient_name: str = Field(..., description="Patient full name ")
    visit_date: str = Field(..., description="Date of clinical visit")
    doctor_name: str = Field(..., description="Treating doctor or clinician")
    diagnosis: str = Field(..., description="Initial visit diagnosis or chief complaint")
    synthetic_only: bool = Field(default=True, description="Safety flag confirming synthetic benchmark data")
    sample_id: str = Field(..., description="Dataset tracking ID")

class DemoGraphics(BaseModel):
    "Basic Patient Detials"
    age: Optional[int] = Field(None, description = "Take the Patients age in year")
    gender: Optional[str] = Field(None, description = "Ask Gender(male, female, other)")


class ClinicalFact (BaseModel):
    "Represents one single clinical fact extracted from medical text or OCR for example A diesease, a precription or a lab blood test"
    name: str = Field(..., description = "Name of the entry like Metaformin or Type 1 Diabeties")
    category: str = Field(..., description = "Must be: 'Condition' 'Medication' 'Lab' or 'Demographics ")
    value: Optional[str] = Field(None, description = "Dosages or measurement like 100mg twice a daily" )
    standard_code: Optional[str] = Field(None, description="Standard code (e.g. ICD-10 'E11' RxNorm '6809')")
    temporal_status: Optional[str] = Field("active", description="Is this 'active' 'historical' or 'resolved'?")


class ClinicalFactTree(BaseModel):
    "The structured bundle of medical data extracted from a text or photo "
    demographics: DemoGraphics
    conditions: List[ClinicalFact] = Field(default_factory=list, description="Diagnoses and diseases")
    medications: List[ClinicalFact] = Field(default_factory=list, description="Active prescriptions")
    labs: List[ClinicalFact] = Field(default_factory=list, description="Lab test values and bloodwork")

class TrailCriteria(BaseModel):
    "Represents an inclusion or exclusion rule from a clinical trial protocol"
    criterion_id: str = Field(..., description="Unique ID for the rule, like 'INC_01' 'EXC_02')")
    criteria_type: str = Field(..., description="'inclusion' or 'exclusion'")
    description: str = Field(..., description="The full sentence describing the rule")
    target_category: str = Field(..., description="Category targeted: 'Condition' 'Medication' 'Lab' or 'Age'")
    threshold_value: Optional[str] = Field(None, description="Numerical cutoff if any (e.g. '< 30')")

class EvaluationResult(BaseModel):
    "The scorecard produced by the Self-Correction Evaluator Agent and If any one of the check fails the workflow loops back to fix the draft"
    is_grounded: bool = Field(..., description="True if every claim has a source citation. False if any statement was made up")
    is_complete: bool = Field(..., description="True if all required trial rules were evaluated. False if required tests were missed")
    is_relevant: bool = Field(..., description="True if retrieved facts actually answer the trial eligibility requirements")
    missing_entities: List[str] = Field(default_factory=list, description="List of required items not found ( ['HbA1c lab result'])")
    critique: str = Field(..., description="Clear constructive feedback explaining what went wrong and how to fix the draft")

class Citation(BaseModel):
    "It tracks each and every fact mentioned came from so that every claim can be verified"
    source_id: str 
    source_type: str
    snippet: str 

class ReportGeneration(BaseModel):
    "The memory state of our workflow, every agent will reads from this dictionary, updates its piece and passes it forward"
    report_metadata: Optional[Metadata] = None

    raw_intake_text: Optional[str] = None
    image_path: Optional[str] = None
    ocr_text: Optional[str] = None
    ocr_status: Optional[str] = None
    sanitized_text: str = ""

    extracted_data: Optional[ClinicalFactTree] = None

    deterministic_rule_passed: bool = True
    deterministic_reason: str = ""

    retrieved_graph_facts: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_vector_chunks: List[str] = Field(default_factory=list)
    provenance_citations: List[Citation] = Field(default_factory=list)

    draft_report: str = ""
    evaluation: Optional[EvaluationResult] = None
    iteration_count: int = 0

    final_report: Optional[str] = None 


SAMPLE_REPORT_METADATA_DATA = {
    "category": "medical_report",
    "report_id": "MED-327KSAUA",
    "patient_name": "Mark Cannon",
    "visit_date": "2026-09-06",
    "doctor_name": "Dr. Gina Wright",
    "diagnosis": "Hypertension follow-up",
    "synthetic_only": True,
    "sample_id": "medical_report_0001"
}

SAMPLE_PATIENT_NOTE = """
PATIENT INTAKE NOTE:
Patient Name: Mark Cannon
Visit Date: 2026-09-06
Doctor: Dr. Gina Wright
Diagnosis: Hypertension follow-up

HISTORY OF PRESENT ILLNESS:
54-year-old male with a history of Type 2 Diabetes Mellitus and Essential Hypertension.
Patient is currently taking Metformin 1000 mg twice daily and Lisinopril 20 mg once daily.
Complains of occasional fatigue.

RECENT LABS:
- Serum Creatinine: 2.1 mg/dL
- eGFR: 28 mL/min/1.73m2 (Stage 4 Chronic Kidney Disease)
- Blood Pressure: 138/86 mmHg
(Note: HbA1c lab result is missing from current encounter).
""" .strip()

SAMPLE_TRIAL_PROTOCOL = """
CLINICAL TRIAL PROTOCOL: CARDIO-RENAL SGLT2 STUDY (STUDY-CR-401)
INCLUSION CRITERIA:
1. Adults aged 18 to 75 years at time of screening.
2. Documented diagnosis of Type 2 Diabetes Mellitus.
3. Documented HbA1c between 7.0% vnd 10.5% within 30 days prior to screening.
EXCLUSION CRITERIA:
1. Severe renal impairment defined as eGFR < 30 mL/min/1.73m2.
2. Concurrent use of medications with severe contraindications in renal failure.
""".strip()

def testing():
        """Basic validation that models and state instantiate without error."""
        meta = Metadata(**SAMPLE_REPORT_METADATA_DATA)
        state = ReportGeneration(
            report_metadata=meta,
            raw_intake_text=SAMPLE_PATIENT_NOTE
        )
        eval_res = EvaluationResult(
            is_grounded=True,
            is_complete=False,
            is_relevant=True,
            missing_entities=["HbA1c lab result"],
            critique="Trial requires HbA1c, but patient blood draw is not completed."
        )
        assert state.report_metadata.report_id == meta.report_id
        assert eval_res.is_grounded is True
        print("Schema definitions and sample state validated successfully.")


if __name__ == "__main__":
    testing()