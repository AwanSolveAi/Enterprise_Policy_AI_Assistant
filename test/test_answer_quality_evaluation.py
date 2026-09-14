import json
from pathlib import Path

import pytest

from evaluation.answer_quality import (
    deterministic_generator, evidence_coverage, evaluate_case, load_benchmark,
    grounding_evaluations, summarize, unsupported_claims,
)
from src.generation.context_builder import _resolve_jurisdictions, build_context
from src.generation.rag_engine import build_prompt

def result(cid="c1", document="d1", category="Benefits", text="BUPA provides private health insurance"):
    return {"chunk_id":cid,"chunk":{"chunk_id":cid,"document_id":document,"category":category,"title":"Policy","chunk_text":text,"source_url":"https://example.test"}}

class FixedRetriever:
    def __init__(self, results): self.results=results
    def search(self,*args,**kwargs): return {"routes":[],"hybrid_results":self.results}

def test_benchmark_schema_size_and_unanswerable_count():
    cases=load_benchmark()
    assert 25 <= len(cases) <= 30
    assert sum(not case["answerable"] for case in cases) >= 5
    assert all(case.get("expected_abstention") for case in cases if not case["answerable"])

def test_deterministic_generator_uses_only_present_evidence():
    context=build_context([result()])
    case={"answerable":True,"evidence_terms":["BUPA","private health insurance"]}
    answer=deterministic_generator(case,context)
    assert "[S1]" in answer and evidence_coverage(answer,case["evidence_terms"])==1
    case["evidence_terms"]=["not present"]
    assert "couldn't find enough" in deterministic_generator(case,context)

def test_context_selects_query_evidence_beyond_first_five_and_stays_bounded():
    results=[result(f"c{i}",text=("generic policy text "*120),document="d1",category="Offboarding") for i in range(1,7)]
    results.append(result("target",text=("prefix "*200)+"benefits will cease on the last day",document="d1",category="Offboarding"))
    for rank,item in enumerate(results,1): item["rank"]=rank; item["chunk"]["chunk_index"]=rank
    context=build_context(results,max_chars=3000,max_chunks=5,query="When will benefits cease on the last day?")
    assert "target" in context.chunk_ids and "benefits will cease" in context.text
    assert len(context.text)<=3000 and len(context.sources)<=5

def test_bounded_excerpt_preserves_query_critical_policy_label():
    text=("general leave guidance "*90)+"Enter the time away in Workday by selecting the label \u2018Out Sick\u2019. "+("other policy details "*90)
    item=result("sick",document="leave",category="Leave & Time Off",text=text)
    item["rank"]=1; item["chunk"]["chunk_index"]=1; item["chunk"]["country"]="Global"
    context=build_context([item],max_chars=700,max_chunks=1,query="How should a team member record sick time?")
    assert "Enter the time away in Workday by selecting the label \u2018Out Sick\u2019" in context.text
    assert len(context.text)<=700

def test_context_keeps_dominant_document_overview_for_broad_question():
    results=[]
    for rank,index in enumerate([19,4,18,9,1],1):
        item=result(f"c{index}",document="offboarding",category="Offboarding",text="departure procedure")
        item["rank"]=rank; item["chunk"]["chunk_index"]=index; results.append(item)
    context=build_context(results,max_chunks=3,query="What happens when an employee leaves?")
    assert "c1" in context.chunk_ids

def test_generic_context_prefers_global_when_relevant_global_evidence_exists():
    australia=result("au",document="au",category="Benefits",text="Select Out Sick in Workday")
    australia["chunk"]["country"]="Australia"; australia["rank"]=1; australia["chunk"]["chunk_index"]=1
    global_result=result("global",document="global",category="Leave & Time Off",text="Enter sick time in Workday as Out Sick")
    global_result["chunk"]["country"]="Global"; global_result["rank"]=2; global_result["chunk"]["chunk_index"]=1
    context=build_context([australia,global_result],query="How should a team member record sick time?")
    assert context.chunk_ids==("global",)

def test_explicit_australia_context_prefers_australian_evidence():
    australia=result("au",document="au",category="Benefits",text="Australian sick leave uses Workday")
    australia["chunk"]["country"]="Australia"; australia["rank"]=2; australia["chunk"]["chunk_index"]=1
    global_result=result("global",text="Global sick leave uses Workday")
    global_result["chunk"]["country"]="Global"; global_result["rank"]=1; global_result["chunk"]["chunk_index"]=1
    context=build_context([global_result,australia],query="How does sick leave work in Australia?")
    assert "au" in context.chunk_ids and "global" in context.chunk_ids

def test_explicit_france_context_keeps_france_and_bounds():
    results=[]
    for rank in range(1,8):
        item=result(f"fr{rank}",document="fr",category="Remote Work",text="France equipment return policy")
        item["chunk"]["country"]="France"; item["rank"]=rank; item["chunk"]["chunk_index"]=rank; results.append(item)
    other=result("au",text="Australia equipment policy"); other["chunk"]["country"]="Australia"; other["rank"]=8
    context=build_context(results+[other],max_chars=1200,max_chunks=3,query="What happens to equipment in France?")
    assert len(context.chunk_ids)==3 and "au" not in context.chunk_ids and len(context.text)<=1200

@pytest.mark.parametrize(("query","expected"),[
    ("US employee","United States"),
    ("USA employee","United States"),
    ("American employee","United States"),
    ("Australian team member","Australia"),
    ("French employee","France"),
    ("Irish employee","Ireland"),
    ("Indian employee","India"),
])
def test_jurisdiction_aliases_resolve_to_metadata_country(query,expected):
    assert _resolve_jurisdictions(query)=={expected}

def test_generic_sick_leave_has_no_resolved_jurisdiction():
    assert _resolve_jurisdictions("How should a team member record sick time?")==set()

def test_unsupported_claim_detector_checks_cited_source():
    context=build_context([result()])
    assert not unsupported_claims("BUPA provides private health insurance [S1].",context)
    assert unsupported_claims("Employees receive a private helicopter every week [S1].",context)
    assert unsupported_claims("An uncited assertion.",context)

def test_narrow_contact_and_privacy_phrase_normalization():
    context=build_context([result(text="For payroll questions, reach out to payroll@example.test. An involuntary departure cannot be shared because it affects privacy.")])
    answer="Payroll questions are directed to payroll@example.test. An involuntary departure is not disclosed because it affects privacy. [S1]"
    assert not unsupported_claims(answer,context)

def test_prompt_qualifies_country_specific_evidence_for_generic_question():
    context=build_context([result(text="US Team Members\nBenefits end after the final eligible month.",document="us")])
    prompt=build_prompt("When can benefits end?",context)
    assert "explicitly limited to: United States" in prompt
    assert "explicitly qualify every such statement" in prompt

def test_prompt_allows_direct_answer_for_explicit_jurisdiction_question():
    context=build_context([result(text="US Team Members may request records in writing.",document="us")])
    prompt=build_prompt("How may a US employee request records?",context)
    assert "explicitly limited to: United States" not in prompt

def test_prompt_does_not_add_country_qualification_for_global_evidence():
    context=build_context([result(text="This global policy applies to all team members regardless of location.")])
    prompt=build_prompt("Who may use this policy?",context)
    assert "The context contains provisions explicitly limited to:" not in prompt

def test_prompt_requires_direct_citation_for_factual_introductory_sentence():
    prompt=build_prompt("How is sick time recorded?",build_context([result(text="Enter sick time in Workday.")]))
    assert "Every factual introductory or summary sentence" in prompt
    assert "citations on later bullets do not support an uncited summary" in prompt

def test_prompt_requires_all_material_contact_options():
    prompt=build_prompt("Who should I contact?",build_context([result(text="Contact a manager or People Operations.")]))
    assert "include every materially relevant alternative" in prompt

def test_semantic_grounding_supports_sick_leave_paraphrases():
    context=build_context([result(text="Enter the time away in Workday by selecting the label 'Out Sick'. Enter caregiver leave using the label 'Caregiver Sick Time'.")])
    answer="Record sick time in Workday and select the Out Sick label [S1]. Select Caregiver Sick Time for caregiver leave [S1]."
    assert not unsupported_claims(answer,context)

def test_direct_containment_supports_short_list_fragment():
    context=build_context([result(text="Leave covers recovery from a serious health condition, caring for family, and serving in the military.")])
    assert not unsupported_claims("serving in the military [S1]",context)

@pytest.mark.parametrize("response",["Yes.","No."])
def test_leading_yes_no_is_associated_with_supported_explanation(response):
    context=build_context([result(text="The policy applies to contractors and employees in all locations.")])
    evaluations=grounding_evaluations(response+" The policy applies to contractors and employees in all locations [S1].",context)
    assert len(evaluations)==1 and evaluations[0]["status"]=="SUPPORTED"

def test_semantic_completeness_accepts_faithful_paraphrase():
    assert evidence_coverage("Bereavement leave is provided through Flexible PTO.",["time off"])==1.0

def test_adjacent_cited_sentences_support_compound_claim():
    context=build_context([result(text="Employees need not work with the alleged harasser during investigation. The alleged harasser may be suspended or transferred.")])
    assert not unsupported_claims("The parties may be kept apart during investigation by suspending or transferring the alleged harasser [S1].",context)

def test_jurisdiction_specific_provision_is_not_generalized():
    context=build_context([result(text="For employees in Ireland, managers must follow this complaint process.")])
    assert unsupported_claims("All employees must follow this complaint process [S1].",context)

def test_semantic_grounding_supports_offboarding_privacy_paraphrase():
    context=build_context([result(text="An involuntary departure generally cannot be shared since it affects the individual's privacy.")])
    assert not unsupported_claims("An involuntary departure is not disclosed publicly to protect privacy [S1].",context)

def test_semantic_grounding_rejects_jurisdiction_generalization():
    context=build_context([result(text="For employees in UAE, dependent visas must be cancelled before departure.")])
    assert unsupported_claims("The offboarding process for all employees includes visa cancellation [S1].",context)

def test_explicit_query_jurisdiction_propagates_to_answer_claims():
    context=build_context([result(text="US employees may discuss their own pay with coworkers.")])
    assert not unsupported_claims("Team members may discuss their own pay with coworkers [S1].",context,"May US employees discuss their pay?")

def test_explicit_global_applicability_overrides_incidental_country_reference():
    context=build_context([result(text="This policy applies to contractors and employees in all locations. Local laws in the United States must still be followed.")])
    assert not unsupported_claims("The policy applies to contractors and employees worldwide [S1].",context)

def test_pronoun_and_scope_continue_from_preceding_discourse():
    context=build_context([result(text="Current and former US team members may request records. Team members can access records anytime through Workday.")])
    answer="Current and former US team members may request records [S1]. They can access records any time through Workday [S1]."
    assert not unsupported_claims(answer,context)

def test_supported_causal_paraphrase_is_accepted():
    context=build_context([result(text="An involuntary departure cannot be shared since it affects the individual's privacy.")])
    assert not unsupported_claims("The departure is not shared because of the individual's privacy [S1].",context)

def test_semantic_grounding_rejects_causal_inference():
    context=build_context([result(text="Equipment must be returned. Misuse of equipment may result in dismissal.")])
    assert unsupported_claims("Failure to return equipment results in dismissal [S1].",context)

def test_grounding_distinguishes_uncited_and_invalid_citation():
    context=build_context([result(text="Employees must record leave in Workday.")])
    assert grounding_evaluations("Employees record leave in Workday.",context)[0]["status"]=="UNCITED"
    assert grounding_evaluations("Employees record leave in Workday [S99].",context)[0]["status"]=="UNSUPPORTED"

def test_unsupported_claim_detector_normalizes_format_and_paraphrase():
    context=build_context([result(text="The offboarding process is facilitated by the People Operations team. All equipment provided by the Company remains its property. The team member must return it at the end of the employment contract without delay.")])
    answer="The off\u2011boarding process is coordinated by People Operations. Company-provided equipment remains company property and must be returned when employment ends. [ S1 ]"
    assert not unsupported_claims(answer,context)

def test_unsupported_claim_detector_keeps_inference_and_transferred_consequences():
    context=build_context([result(text="Equipment must be returned when employment ends. Misuse of equipment may result in dismissal.")])
    assert unsupported_claims("Failure to return equipment may result in dismissal [S1].",context)
    assert unsupported_claims("This rule applies equally to every kind of termination [S1].",context)

def test_diagnosed_supported_reporting_and_sick_leave_paraphrases():
    context=build_context([result(text="If behavior does not cease, report the misconduct to a People Business Partner. Enter sick time in Workday by selecting the label Out Sick.")])
    answer="If the behavior continues, report it to a People Business Partner [S1]. Record sick time in Workday as Out Sick [S1]."
    assert not unsupported_claims(answer,context)

def test_citation_only_line_attaches_to_immediately_preceding_claim():
    context=build_context([result(text="The employee must return company-provided equipment at the end of the employment contract without delay.")])
    answer="Company-provided equipment must be returned immediately.\n[S1]"
    assert not unsupported_claims(answer,context)
    assert unsupported_claims("An uncited assertion.\nCompany-provided equipment must be returned immediately.\n[S1]",context)==["An uncited assertion."]

def test_markdown_heading_is_structural_and_abbreviation_is_not_split():
    context=build_context([result(text="GitLab France S.A.S. requires equipment to be returned without delay.")])
    answer="**Support contacts** \u2013\nGitLab France S.A.S. requires equipment to be returned immediately. [S1]"
    assert not unsupported_claims(answer,context)

def test_answerable_case_scores_all_dimensions():
    case={"id":"health","group":"benefits","question":"Who provides health insurance?","answerable":True,
          "expected_document_id":"d1","expected_category":"Benefits","evidence_terms":["BUPA","private health insurance"]}
    row=evaluate_case(case,FixedRetriever([result()]))
    assert row["passed"] and row["citation_valid"] and row["citation_source_correct"]
    assert row["retrieval_source_correct"] and not row["unsupported_claims"]
    assert row["raw_model_answer"]==row["answer"] and row["completeness"]==1

def test_unanswerable_case_requires_abstention():
    case={"id":"unknown","group":"unanswerable","question":"Unknown?","answerable":False,"expected_abstention":True,"evidence_terms":[]}
    row=evaluate_case(case,FixedRetriever([result()]))
    assert row["passed"] and row["abstention_correct"] and not row["answer"].startswith("BUPA")

def test_summary_reports_failures_and_latency():
    rows=[{"group":"x","answerable":True,"retrieval_source_correct":True,"grounded_answer_accuracy":True,"citation_valid":True,
           "citation_source_correct":True,"abstention_correct":True,"answer_relevant":True,"unsupported_claims":[],"completeness":1.0,"total_ms":2.0,"passed":True,"id":"ok"}]
    summary=summarize(rows,"deterministic_logic")
    assert summary["passed"]==1 and summary["failed"]==0 and summary["latency_ms"]["mean"]==2.0

def test_invalid_small_benchmark_rejected(tmp_path: Path):
    path=tmp_path/"cases.json"; path.write_text(json.dumps([]),encoding="utf-8")
    with pytest.raises(ValueError): load_benchmark(path)

def test_run_reports_incremental_progress():
    from evaluation.answer_quality import run
    case={"id":"unknown","group":"unanswerable","question":"Unknown?","answerable":False,"expected_abstention":True,"evidence_terms":[]}
    progress=[]; report=run(FixedRetriever([result()]),[case],on_case=lambda index,total,rows:progress.append((index,total,len(rows))))
    assert progress==[(1,1,1)] and report["summary"]["cases"]==1
