from pathlib import Path

import pytest

from evaluation.groq_benchmark import retry_after_seconds, run_cases, run_with_retries


class RateLimited(Exception):
    status_code=429
    def __init__(self,retry_after=None):
        self.response=type("Response",(),{"headers":{} if retry_after is None else {"Retry-After":retry_after}})()


def stats():
    return {"successful_evaluated_cases":0,"api_attempts":0,"rate_limit_attempts":0,"retries":0,"permanently_failed_requests":0}


def test_immediate_success():
    counters=stats(); counters["api_attempts"]+=1
    assert run_with_retries(lambda:{"passed":True},counters)=={"passed":True}
    assert counters["rate_limit_attempts"]==0 and counters["retries"]==0


def test_429_then_success():
    counters=stats(); calls=[]
    def operation():
        counters["api_attempts"]+=1; calls.append(1)
        if len(calls)==1: raise RateLimited()
        return {"passed":True}
    assert run_with_retries(operation,counters,sleep=lambda _:None)["passed"]
    assert counters["api_attempts"]==2 and counters["rate_limit_attempts"]==1 and counters["retries"]==1


def test_repeated_429_exhausts_three_retries():
    counters=stats()
    def operation(): counters["api_attempts"]+=1; raise RateLimited()
    with pytest.raises(RateLimited): run_with_retries(operation,counters,max_retries=3,sleep=lambda _:None)
    assert counters["api_attempts"]==4 and counters["rate_limit_attempts"]==4 and counters["retries"]==3


def test_retry_after_is_honored():
    counters=stats(); delays=[]; calls=[]
    def operation():
        counters["api_attempts"]+=1; calls.append(1)
        if len(calls)==1: raise RateLimited("7.5")
        return {"passed":True}
    run_with_retries(operation,counters,sleep=delays.append)
    assert delays==[7.5] and retry_after_seconds(RateLimited("7.5"))==7.5


def test_checkpoint_resume_skips_completed_cases(tmp_path: Path):
    output=tmp_path/"checkpoint.json"; calls=[]; counters=stats()
    cases=[{"id":"a"},{"id":"b"}]
    def evaluate(case): counters["api_attempts"]+=1; calls.append(case["id"]); return {"id":case["id"],"passed":True,"group":"x","answerable":True,
        "retrieval_source_correct":True,"grounded_answer_accuracy":True,"citation_valid":True,"citation_source_correct":True,
        "abstention_correct":True,"answer_relevant":True,"unsupported_claims":[],"completeness":1.0,"total_ms":1.0}
    first=run_cases(cases,evaluate,output,pacing_seconds=0,sleep=lambda _:None,stats=counters)
    second=run_cases(cases,evaluate,output,pacing_seconds=0,sleep=lambda _:None,stats=counters)
    assert first["complete"] and second["complete"] and calls==["a","b"]
    assert second["completed_case_ids"]==["a","b"]
