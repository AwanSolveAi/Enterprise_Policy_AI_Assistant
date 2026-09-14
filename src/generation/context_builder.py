"""Bounded, query-aware context construction with stable citation labels."""
from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

STOPWORDS={"a","an","and","are","as","at","be","by","does","for","from","gitlab","how","in","is","it","of","on","or","that","the","their","to","what","when","where","which","who","with"}
JURISDICTION_PATTERNS={
    "United States": re.compile(r"(?<!\w)(?:united\s+states(?:\s+of\s+america)?|u\.?\s*s\.?\s*a\.?|u\.?\s*s\.?|american)(?!\w)",re.I),
    "Australia": re.compile(r"(?<!\w)australia(?:n)?(?!\w)",re.I),
    "France": re.compile(r"(?<!\w)(?:france|french)(?!\w)",re.I),
    "Ireland": re.compile(r"(?<!\w)(?:ireland|irish)(?!\w)",re.I),
    "India": re.compile(r"(?<!\w)(?:india|indian)(?!\w)",re.I),
}

@dataclass(frozen=True)
class ContextBundle:
    text: str
    sources: tuple[dict[str, str], ...]
    chunk_ids: tuple[str, ...]

def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9']+",text.lower()) if len(word)>2 and word not in STOPWORDS}

def _query_tokens(query: str) -> set[str]:
    tokens=_tokens(query)
    if tokens & {"how","where","record","request","submit","report","contact"} or re.search(r"\b(how|where)\b",query,re.I):
        tokens.update({"workday","submit","select","email","notify","helplab","reporting"})
    return tokens

def _text(result: dict[str,Any]) -> str:
    chunk=result.get("chunk",result)
    return str(chunk.get("chunk_text",chunk.get("text",""))).strip()

def _resolve_jurisdictions(query: str, available: set[str] | None=None) -> set[str]:
    resolved={country for country,pattern in JURISDICTION_PATTERNS.items() if pattern.search(query)}
    return resolved if available is None else resolved & available

def _select(results: list[dict[str,Any]], query: str, max_chunks: int) -> list[dict[str,Any]]:
    """Balance lexical evidence, reranked order, and one document overview."""
    unique=[]; seen=set()
    for result in results:
        chunk=result.get("chunk",result); chunk_id=str(result.get("chunk_id") or chunk.get("chunk_id","")).strip()
        if chunk_id and chunk_id not in seen and _text(result): seen.add(chunk_id); unique.append(result)
    if not query: return unique[:max_chunks]
    countries={str(item.get("chunk",item).get("country","")).strip() for item in unique}
    jurisdictions=_resolve_jurisdictions(query,countries)
    if jurisdictions:
        eligible=[item for item in unique if str(item.get("chunk",item).get("country","")).strip() in jurisdictions|{"Global"}]
        unique=eligible or unique
    else:
        global_items=[item for item in unique if str(item.get("chunk",item).get("country","")).strip()=="Global"]
        query_terms=_tokens(query)
        relevant_global=[item for item in global_items if query_terms & _tokens(" ".join([str(item.get("chunk",item).get("title","")),_text(item)]))]
        if relevant_global:
            unique=global_items
    query_tokens=_query_tokens(query)
    def relevance(item):
        chunk=item.get("chunk",item); searchable=" ".join([str(chunk.get("title","")),str(chunk.get("category","")),_text(item)])
        overlap=len(query_tokens & _tokens(searchable))/max(1,len(query_tokens)); rank=int(item.get("rank",len(unique)+1))
        country=str(chunk.get("country","")).strip()
        jurisdiction_preference=1 if jurisdictions and country in jurisdictions else 0
        return (jurisdiction_preference,overlap,1/rank)
    documents=Counter(str(item.get("chunk",item).get("document_id","")) for item in unique)
    dominant=documents.most_common(1)[0][0] if documents else ""
    overviews=[item for item in unique if str(item.get("chunk",item).get("document_id",""))==dominant]
    selected=sorted(unique,key=relevance,reverse=True)[:max_chunks]
    if overviews:
        overview=min(overviews,key=lambda item:int(item.get("chunk",item).get("chunk_index") or 10**9))
        if overview not in selected:
            replaceable=[item for item in selected if str(item.get("chunk",item).get("document_id",""))!=dominant]
            victim=min(replaceable or selected,key=relevance)
            selected[selected.index(victim)]=overview
    return sorted(selected,key=lambda item:int(item.get("rank",10**9)))

def _excerpt(text: str, query: str, limit: int) -> str:
    if len(text)<=limit: return text
    base_tokens=_tokens(query); tokens=_query_tokens(query); occurrences=[]; counts={}
    for token in tokens:
        matches=[match.start() for match in re.finditer(re.escape(token),text,re.I)]
        counts[token]=len(matches)
        occurrences.extend((position,token) for position in matches)
    if not occurrences: windows=[(0,limit)]
    else:
        window_size=max(250,limit//2)
        starts=sorted({max(0,position-window_size//12) for position,_ in occurrences})
        def window_score(candidate):
            included=[(position,token) for position,token in occurrences if candidate<=position<candidate+window_size]
            included_tokens={token for _,token in included}
            unique=(3*len(included_tokens & base_tokens))+len(included_tokens-base_tokens)
            rarity=sum(1/max(1,counts[token]) for _,token in included)
            return (unique,rarity,-candidate)
        ranked=sorted(starts,key=window_score,reverse=True); chosen=[ranked[0]]
        if re.search(r"\b(contact|concerns?)\b",query,re.I):
            tail=max(0,len(text)-window_size)
            if all(abs(tail-other)>=window_size//2 for other in chosen): chosen.append(tail)
        else:
            for start in ranked:
                if all(abs(start-other)>=window_size//2 for other in chosen): chosen.append(start); break
        windows=[(start,min(len(text),start+window_size)) for start in sorted(chosen)]
    excerpts=[]
    for start,end in windows:
        if start:
            boundary=text.find(" ",start); start=boundary+1 if boundary>=0 else start
        if end<len(text):
            boundary=text.rfind(" ",start,end); end=boundary if boundary>start else end
        excerpts.append(("… " if start else "")+text[start:end].strip()+(" …" if end<len(text) else ""))
    return "\n".join(excerpts)

def build_context(results: list[dict[str, Any]], max_chars=7000, max_chunks=5, query: str="") -> ContextBundle:
    chosen=_select(results,query,max_chunks); blocks=[]; sources=[]; used=0
    body_limit=max(300,(max_chars-(max_chunks*110))//max(1,len(chosen)))
    for result in chosen:
        chunk=result.get("chunk",result); chunk_id=str(result.get("chunk_id") or chunk.get("chunk_id","")).strip()
        label=f"S{len(sources)+1}"; header=f"[{label}] {chunk.get('title','Untitled')} (chunk_id={chunk_id})"
        allowance=min(body_limit,max_chars-used-len(header)-2)
        if allowance<=0: break
        block=f"{header}\n{_excerpt(_text(result),query,allowance)}"; blocks.append(block); used+=len(block)+2
        sources.append({"citation":label,"chunk_id":chunk_id,"title":str(chunk.get("title","Untitled")),"source_url":str(chunk.get("source_url",""))})
    return ContextBundle("\n\n".join(blocks),tuple(sources),tuple(source["chunk_id"] for source in sources))
