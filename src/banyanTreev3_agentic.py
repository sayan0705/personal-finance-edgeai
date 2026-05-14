# =================================================================
# LOCAL VS CODE ENTRYPOINT
# Dependencies are installed from requirements.txt.
# =================================================================
# Local VS Code setup:
#   python -m pip install -r requirements.txt
#   python -m spacy download en_core_web_sm
#   playwright install chromium
print("BanyanTree local script bootstrap ready")



import subprocess, time, httpx, os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = Path(os.environ.get("BANYANTREE_DATA_DIR", PROJECT_ROOT / "data")).resolve()
KG_ROOT = Path(os.environ.get("BANYANTREE_KG_ROOT", DATA_DIR / "kg")).resolve()
FINANCIAL_KG_ROOT = Path(os.environ.get("BANYANTREE_FINANCIAL_KG_ROOT", DATA_DIR / "financial_kg")).resolve()
RAW_DOCS_DIR = Path(os.environ.get("BANYANTREE_RAW_DOCS_DIR", FINANCIAL_KG_ROOT / "raw_docs")).resolve()
SEED_DOCS_PATH = Path(
    os.environ.get("BANYANTREE_SEED_DOCS_PATH", RAW_DOCS_DIR / "seed" / "personal_finance_seed.json")
).resolve()
PAGEINDEX_FLATTENED_DOCS_PATH = Path(
    os.environ.get("BANYANTREE_PAGEINDEX_FLATTENED_DOCS_PATH", RAW_DOCS_DIR / "pageindex" / "pageindex_flattened_docs.json")
).resolve()
MEMORY_DIR = Path(os.environ.get("BANYANTREE_MEMORY_DIR", DATA_DIR / "memory")).resolve()
LOG_DIR = Path(os.environ.get("BANYANTREE_LOG_DIR", PROJECT_ROOT / "logs")).resolve()
MCP_SERVER_FILE = Path(os.environ.get("BANYANTREE_MCP_SERVER_FILE", SRC_DIR / "finsage_mcp_server.py")).resolve()

for _path in (DATA_DIR, KG_ROOT, FINANCIAL_KG_ROOT, RAW_DOCS_DIR, SEED_DOCS_PATH.parent, PAGEINDEX_FLATTENED_DOCS_PATH.parent, MEMORY_DIR, LOG_DIR):
    _path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf_cache"))
os.environ.setdefault("FINSAGE_KG_PATH", str(FINANCIAL_KG_ROOT / "lightrag"))
os.environ.setdefault("BANYANTREE_TOOL_POLICY_PATH", str(PROJECT_ROOT / "config" / "tool_permission_policy.json"))
os.environ.setdefault("BANYANTREE_AGENTIC_MEMORY_FILE", str(MEMORY_DIR / "banyantree_agent_memory.json"))
MCP_BASE = os.environ.get("BANYANTREE_MCP_BASE", "http://localhost:8000")

def _load_financial_doc_file(path: Path, required: bool = False) -> list:
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Financial docs not found: {path}. "
                "Create the docs file or update the matching BANYANTREE_*_DOCS_PATH setting."
            )
        return []

    with open(path, "r", encoding="utf-8-sig") as f:
        docs = json.load(f)

    if not isinstance(docs, list):
        raise ValueError(f"Financial docs file must contain a JSON list: {path}")

    normalized = []
    for i, doc in enumerate(docs):
        if not isinstance(doc, dict):
            raise ValueError(f"Doc #{i} must be a JSON object")
        title = str(doc.get("title", f"Document_{i}")).strip()
        content = str(doc.get("content", "")).strip()
        if not content:
            continue
        normalized.append({**doc, "title": title, "content": content})

    return normalized


def load_financial_docs(path: Path = SEED_DOCS_PATH) -> list:
    docs = _load_financial_doc_file(path, required=True)
    pageindex_docs = _load_financial_doc_file(PAGEINDEX_FLATTENED_DOCS_PATH, required=False)
    if pageindex_docs:
        print(f"Loaded {len(pageindex_docs)} PageIndex docs from {PAGEINDEX_FLATTENED_DOCS_PATH}")
        docs.extend(pageindex_docs)
    return docs

# =================================================================
# SINGLE CANONICAL MCP SERVER - tools + agentic market workflow
# =================================================================
MCP_SERVER_CODE = r"""
import asyncio, sys, os, re, json
from datetime import datetime
from typing import Optional, List

if sys.platform == 'linux':
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title='FinSage MCP Tool Server', version='8.0.0')

class AMFIRequest(BaseModel):
    fund_filter: Optional[str] = ''

class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 4

class QueryRequest(BaseModel):
    query: str

class SymbolRequest(BaseModel):
    symbol: str

class PortfolioWorkflowRequest(BaseModel):
    query: str
    max_stocks: int = 4
    symbols: Optional[List[str]] = None

class ToolResponse(BaseModel):
    tool: str
    success: bool
    data: dict
    error: Optional[str] = None

_rag_instance = None
_AGENTIC_MEMORY_FILE = os.environ.get(
    'BANYANTREE_AGENTIC_MEMORY_FILE',
    os.path.join(os.getcwd(), 'data', 'memory', 'banyantree_agent_memory.json')
)
_AGENTIC_MARKET_MEMORY = {}
TOOL_NAMES = ['amfi_nav','search_rag','screener','sip_calculator','emi_calculator','portfolio_health','goal_planner','portfolio_multi_agent']


def _amounts(text):
    vals=[]
    for v,u in re.findall(r'(?:rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*(crore|cr|lakh|lac|k|thousand)?', text, re.I):
        x=float(v.replace(',', '')); u=(u or '').lower()
        x*=10000000 if u in {'crore','cr'} else 100000 if u in {'lakh','lac'} else 1000 if u in {'k','thousand'} else 1
        if x >= 500: vals.append(x)
    return vals


def _pct(text, d=12.0):
    m=re.search(r'(\d+(?:\.\d+)?)\s*%', text)
    return float(m.group(1)) if m else d


def _years(text, d=10.0):
    m=re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', text, re.I)
    return float(m.group(1)) if m else d


def _age(text, d=30):
    m=re.search(r"(?:age\s*[:=]?\s*|i am\s+|i'm\s+)(\d{2})", text, re.I)
    return int(m.group(1)) if m else d


def _alloc(text):
    out={}
    for a in ['equity','debt','gold','cash']:
        for p in [rf'{a}\s*(?:is|at|=|:)?\s*(\d{{1,3}})\s*%', rf'(\d{{1,3}})\s*%\s*{a}']:
            m=re.search(p, text, re.I)
            if m:
                out[a]=int(m.group(1)); break
    return out


def _fallback_rag(query, top_k=4):
    base=os.environ.get('FINSAGE_KG_PATH', os.path.join(os.getcwd(), 'data', 'kg', 'finsage_final_kg'))
    docs_p=os.path.join(base, 'documents.json'); titles_p=os.path.join(base, 'titles.json'); meta_p=os.path.join(base, 'document_metadata.json')
    if not (os.path.exists(docs_p) and os.path.exists(titles_p)): return []
    docs=json.load(open(docs_p)); titles=json.load(open(titles_p))
    metas=json.load(open(meta_p)) if os.path.exists(meta_p) else [{} for _ in docs]
    terms=[t for t in re.findall(r'\w+', query.lower()) if len(t)>2]
    scored=[]
    for i,doc in enumerate(docs):
        score=sum(doc.lower().count(t) for t in terms)
        if score>0: scored.append((score, i))
    scored.sort(reverse=True)
    return [{'title': titles[i], 'content': docs[i][:400], 'metadata': metas[i] if i < len(metas) else {}, 'source': 'persisted_rag'} for _,i in scored[:top_k]]


def _load_agentic_memory():
    global _AGENTIC_MARKET_MEMORY
    if _AGENTIC_MARKET_MEMORY:
        return _AGENTIC_MARKET_MEMORY
    try:
        if os.path.exists(_AGENTIC_MEMORY_FILE):
            with open(_AGENTIC_MEMORY_FILE, 'r', encoding='utf-8') as f:
                _AGENTIC_MARKET_MEMORY = json.load(f)
    except Exception:
        _AGENTIC_MARKET_MEMORY = {}
    return _AGENTIC_MARKET_MEMORY


def _save_agentic_memory():
    try:
        os.makedirs(os.path.dirname(_AGENTIC_MEMORY_FILE), exist_ok=True)
        with open(_AGENTIC_MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(_AGENTIC_MARKET_MEMORY, f, indent=2)
    except Exception:
        pass


def _agent_memory_recall(agent_name, limit=3):
    return _load_agentic_memory().get(agent_name, [])[-limit:]


def _agent_memory_remember(agent_name, task, observation, decision, lesson):
    mem=_load_agentic_memory()
    mem.setdefault(agent_name, []).append({'time': datetime.utcnow().isoformat() + 'Z', 'task': task, 'observation': observation, 'decision': decision, 'lesson': lesson})
    _save_agentic_memory()


def _resolve_symbol_with_yfinance(candidate):
    candidate=str(candidate or '').strip()
    if not candidate: return None
    try:
        import yfinance as yf
        trial=[]
        raw=re.sub(r'[^A-Za-z0-9&.-]', '', candidate).upper()
        if re.fullmatch(r'[A-Z0-9&.-]{2,15}', raw): trial.append(raw.replace('.NS','').replace('.BO',''))
        try:
            if hasattr(yf, 'Search'):
                search=yf.Search(candidate, max_results=8)
                for q in getattr(search, 'quotes', []) or []:
                    sym=str(q.get('symbol','')).upper().strip()
                    exch=str(q.get('exchange','')).upper()
                    quote_type=str(q.get('quoteType','')).upper()
                    if sym and (sym.endswith('.NS') or exch in {'NSI','NSE'} or quote_type in {'EQUITY','ETF'}):
                        trial.append(sym.replace('.NS','').replace('.BO',''))
        except Exception:
            pass
        seen=set()
        for sym in trial:
            if not sym or sym in seen or sym in {'PE','NAV','SIP','EMI','NSE','BSE','ETF'}: continue
            seen.add(sym)
            ticker=yf.Ticker(f'{sym}.NS')
            try:
                hist=ticker.history(period='5d', interval='1d')
                if not hist.empty: return sym
            except Exception:
                pass
    except Exception:
        return None
    return None


def _company_name_candidates(query):
    stop={
        'what','is','are','the','a','an','today','current','share','price','stock','market','pe','ratio','p/e','nav','nse','bse',
        'compare','and','vs','versus','for','of','in','india','tell','me','show','give','latest','fundamentals','quarterly','results'
    }
    cleaned=re.sub(r'[^A-Za-z0-9&.\-\s]', ' ', str(query or ''))
    parts=[]
    for tok in cleaned.split():
        low=tok.lower().strip('.-')
        if len(low) >= 3 and low not in stop and not low.isdigit():
            parts.append(tok.strip())
    candidates=[]
    for n in range(1, min(3, len(parts)) + 1):
        for i in range(0, len(parts)-n+1):
            phrase=' '.join(parts[i:i+n]).strip()
            if phrase and phrase not in candidates:
                candidates.append(phrase)
    return candidates


def _extract_symbols_from_query(query):
    candidates=[]
    for tok in re.findall(r'\b[A-Z][A-Z0-9&-]{1,11}\b', query):
        if tok not in {'PE','NAV','SIP','EMI','NSE','BSE','ETF'}: candidates.append(tok)
    candidates += _company_name_candidates(query)
    candidates.append(query)
    out=[]
    for cand in candidates:
        sym=_resolve_symbol_with_yfinance(cand)
        if sym and sym not in out: out.append(sym)
        if len(out) >= 6: break
    print(f"[symbol_resolver] query={query!r} candidates={candidates[:8]} resolved={out}")
    return out
def _extract_portfolio_profile(query):
    ql=query.lower()
    risk='aggressive' if 'aggressive' in ql else 'conservative' if 'conservative' in ql or 'safe' in ql else 'moderate'
    return {'risk': risk, 'horizon_years': _years(query, 5.0), 'age': _age(query, 30)}


async def _yahoo_screener(symbol):
    try:
        import yfinance as yf
        info = yf.Ticker(f'{symbol}.NS').info or {}
        if not info.get('currentPrice'): return None
        ratios=[]
        if info.get('currentPrice'): ratios.append(f"PRICE: {info['currentPrice']}")
        if info.get('trailingPE'): ratios.append(f"PE: {info['trailingPE']:.2f}")
        if info.get('marketCap'): ratios.append(f"MARKET_CAP: {info['marketCap']/1e7:.0f}Cr")
        if info.get('dividendYield'): ratios.append(f"DIV_YIELD: {info['dividendYield']*100:.2f}%")
        return {'summary': f"{symbol} (Yahoo) | NSE | " + ' | '.join(ratios) + f" | Source: finance.yahoo.com | {datetime.now().strftime('%d-%b-%Y')}"}
    except Exception:
        return None


async def _playwright_screener(symbol):
    try:
        from playwright.async_api import async_playwright
        from bs4 import BeautifulSoup
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=['--no-sandbox','--disable-dev-shm-usage','--disable-blink-features=AutomationControlled'])
            context = await browser.new_context(user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36')
            page = await context.new_page()
            await page.goto(f'https://www.screener.in/company/{symbol}/', wait_until='domcontentloaded', timeout=30000)
            await asyncio.sleep(4)
            html = await page.content(); soup = BeautifulSoup(html, 'html.parser'); full_text = soup.get_text(separator=' ', strip=True)
            price='N/A'; m=re.search(r'(?:Current Price|Close Price).*?Rs\s*([\d,]+\.?\d*)', full_text, re.I)
            if m: price=m.group(1)
            ratios=[]
            for key, pat in [('PE', r'P/E\s*(?:TTM)?\s*[:\s]*([\d.]+)'), ('MARKET_CAP', r'Market\s*Cap\s*[:\s]*Rs\s*([\d,]+\.?\d*)\s*Cr')]:
                m=re.search(pat, full_text, re.I)
                if m: ratios.append(f'{key}: {m.group(1)}')
            await browser.close()
            return {'summary': f"{symbol} | NSE | Price: Rs {price} | " + ' | '.join(ratios) + f" | Source: screener.in | {datetime.now().strftime('%d-%b-%Y')}"}
    except Exception as e:
        return {'error': str(e)}


async def _market_data_agent(symbol):
    try:
        import yfinance as yf
        ticker = yf.Ticker(f'{symbol}.NS')
        info = ticker.info or {}
        try:
            fast_info = dict(ticker.fast_info or {})
        except Exception:
            fast_info = {}
        hist = ticker.history(period='3mo', interval='1d')
        last = info.get('currentPrice') or fast_info.get('last_price')
        if last is None and not hist.empty:
            last = float(hist['Close'].iloc[-1])
        first = float(hist['Close'].iloc[0]) if not hist.empty else None
        as_of = hist.index[-1].strftime('%Y-%m-%d') if not hist.empty else datetime.now().strftime('%Y-%m-%d')
        momentum = round(((float(last)-first)/first)*100, 2) if first and last else None
        previous_close = info.get('previousClose') or fast_info.get('previous_close')
        day_change_pct = round(((float(last)-float(previous_close))/float(previous_close))*100, 2) if last and previous_close else None
        row = {'symbol': symbol, 'price': float(last) if last is not None else None, 'previous_close': previous_close, 'day_change_pct': day_change_pct, 'pe': info.get('trailingPE'), 'forward_pe': info.get('forwardPE'), 'beta': info.get('beta'), 'sector': info.get('sector'), 'dividend_yield': (info.get('dividendYield') or 0) * 100 if info.get('dividendYield') is not None else None, 'market_cap': info.get('marketCap'), 'roe': info.get('returnOnEquity'), 'revenue_growth': info.get('revenueGrowth'), 'momentum_3m_pct': momentum, 'source': 'yfinance', 'as_of': as_of, 'summary': (info.get('longBusinessSummary') or '')[:280]}
        print(f"[market_data_agent] yfinance {symbol}: price={row.get('price')} pe={row.get('pe')} as_of={as_of}")
        return row
    except Exception as e:
        print(f"[market_data_agent] yfinance {symbol} failed: {e}")
        return {'symbol': symbol, 'error': str(e), 'source': 'yfinance'}


async def _screener_agent(symbol):
    try:
        if os.environ.get('FINSAGE_ENABLE_PLAYWRIGHT_MCP', '1') == '1':
            print(f"[screener_agent] playwright screener fetch {symbol}")
            row = await _playwright_screener(symbol)
            if row and not row.get('error'):
                return {'symbol': symbol, 'source': 'playwright_screener', 'data': row}
            print(f"[screener_agent] playwright unavailable for {symbol}, falling back to yahoo")
        yf = await _yahoo_screener(symbol)
        return {'symbol': symbol, 'source': 'yahoo_fallback', 'data': yf or {}}
    except Exception as e:
        print(f"[screener_agent] {symbol} failed: {e}")
        return {'symbol': symbol, 'source': 'screener_agent', 'error': str(e)}


def _master_market_agent(query, symbols, profile):
    prior = _agent_memory_recall('master_market_agent')
    plan = {'objective': query, 'symbols': symbols, 'profile': profile, 'instructions': ['market_intelligence_agent: rank tradability, valuation pressure, and momentum', 'fundamental_agent: review quality, profitability, growth, and durability', 'risk_guard_agent: challenge concentration, volatility, and suitability', 'synthesis_agent: combine evidence into final answer'], 'prior_lessons': prior}
    _agent_memory_remember('master_market_agent', query, str(symbols), 'Issued analyst instructions', 'Route equity-market queries through specialist agents before final synthesis.')
    return plan


def _market_intelligence_agent(market_rows, screener_rows, profile):
    outputs=[]
    for row in market_rows:
        if row.get('error') or not row.get('price'):
            outputs.append({'symbol': row.get('symbol'), 'status': 'data_gap', 'notes': [row.get('error', 'missing price')]}); continue
        score=50; notes=[]
        if row.get('momentum_3m_pct') is not None:
            if row['momentum_3m_pct'] > 8: score += 10; notes.append('positive 3M momentum')
            elif row['momentum_3m_pct'] < -8: score -= 10; notes.append('weak 3M momentum')
        if row.get('market_cap') and row['market_cap'] > 500000000000: score += 6; notes.append('large-cap liquidity support')
        if row.get('pe') and row['pe'] > 50: score -= 8; notes.append('valuation appears stretched')
        outputs.append({'symbol': row['symbol'], 'market_score': max(0, min(100, score)), 'notes': notes, 'raw': row})
    _agent_memory_remember('market_intelligence_agent', 'Rank market setup', str([o.get('symbol') for o in outputs]), str(outputs[:2]), 'Momentum and liquidity should be screened before fundamentals are interpreted.')
    return outputs


def _fundamental_agent(market_rows):
    outputs=[]
    for row in market_rows:
        if row.get('error'): continue
        score=50; strengths=[]; weaknesses=[]
        if row.get('roe') is not None:
            if row['roe'] > 0.15: score += 10; strengths.append('healthy ROE')
            elif row['roe'] < 0.08: score -= 8; weaknesses.append('low ROE')
        if row.get('revenue_growth') is not None:
            if row['revenue_growth'] > 0.08: score += 8; strengths.append('revenue growth visible')
            elif row['revenue_growth'] < 0: score -= 8; weaknesses.append('negative revenue growth')
        if row.get('forward_pe') and row.get('pe') and row['forward_pe'] < row['pe']: strengths.append('forward valuation improves')
        outputs.append({'symbol': row['symbol'], 'fundamental_score': max(0, min(100, score)), 'strengths': strengths, 'weaknesses': weaknesses})
    _agent_memory_remember('fundamental_agent', 'Review quality and growth', str([o['symbol'] for o in outputs]), str(outputs[:2]), 'Quality and growth signals must be kept separate from price momentum.')
    return outputs


def _risk_guard_agent(combined, profile):
    findings=[]
    for row in combined:
        risks=[]; raw = row.get('raw', {})
        if raw.get('beta') is not None and raw['beta'] > 1.25 and profile['risk'] == 'conservative': risks.append('beta may be high for conservative suitability')
        if raw.get('pe') is not None and raw['pe'] > 50: risks.append('high PE can compress future returns')
        if raw.get('sector') is None: risks.append('sector unavailable, diversification view is weaker')
        if raw.get('momentum_3m_pct') is not None and raw['momentum_3m_pct'] < -10: risks.append('recent price trend is weak')
        findings.append({'symbol': row['symbol'], 'risk_flags': risks or ['standard equity-market volatility and earnings risk remain']})
    _agent_memory_remember('risk_guard_agent', 'Challenge market recommendation', str(profile), str(findings[:2]), 'Every market answer should include disadvantages and suitability risk.')
    return findings


def _fmt_inr(value):
    try:
        return f"Rs {float(value):,.2f}"
    except Exception:
        return 'N/A' if value is None else str(value)


def _synthesis_agent(master_plan, market_view, fundamental_view, risk_view):
    fundamentals = {x['symbol']: x for x in fundamental_view}; risks = {x['symbol']: x for x in risk_view}; combined=[]
    for m in market_view:
        f = fundamentals.get(m['symbol'], {})
        total = round((m.get('market_score', 0) * 0.45) + (f.get('fundamental_score', 0) * 0.45) + 10, 1)
        combined.append({**m, **f, 'total_score': min(100, total), 'risk_flags': risks.get(m['symbol'], {}).get('risk_flags', [])})
    ranked = sorted(combined, key=lambda x: x.get('total_score', 0), reverse=True)
    recs=[]; disadvantages=[]; live_snapshots=[]
    for row in ranked:
        raw = row.get('raw', {}) or {}
        price = _fmt_inr(raw.get('price'))
        pe = raw.get('pe') if raw.get('pe') is not None else 'N/A'
        day_change = raw.get('day_change_pct') if raw.get('day_change_pct') is not None else 'N/A'
        momentum = raw.get('momentum_3m_pct') if raw.get('momentum_3m_pct') is not None else 'N/A'
        source = raw.get('source', 'yfinance')
        as_of = raw.get('as_of', datetime.now().strftime('%Y-%m-%d'))
        live_snapshots.append(f"{row['symbol']}: LIVE_PRICE={price} | DAY_CHANGE_PCT={day_change} | PE={pe} | SECTOR={raw.get('sector') or 'N/A'} | 3M_MOMENTUM_PCT={momentum} | SOURCE={source} | AS_OF={as_of}")
        why = '; '.join((row.get('notes') or [])[:2] + (row.get('strengths') or [])[:2]) or 'mixed but usable data'
        recs.append(f"{row['symbol']}: score {row['total_score']}/100 | price={price} | PE={pe} | why={why}")
        disadvantages.append(f"{row['symbol']}: disadvantages={'; '.join(row.get('risk_flags') or ['standard market risk'])}")
    top = ranked[0]['symbol'] if ranked else 'No clear pick'
    summary = f"Agentic market review | top candidate={top} | risk_profile={master_plan['profile']['risk']} | horizon={master_plan['profile']['horizon_years']} yrs | not a guaranteed recommendation.\nLive market snapshot:\n" + '\n'.join(live_snapshots)
    _agent_memory_remember('synthesis_agent', master_plan['objective'], top, summary, 'Final market response should cite live price, source, as-of date, strongest evidence, and clearest risks.')
    return summary, recs, disadvantages, ranked


async def _banyantree_agentic_market_workflow(query, max_stocks=4, symbols=None):
    symbols = [str(s).upper().replace('.NS','').replace('.BO','') for s in (symbols or []) if str(s).strip()] or _extract_symbols_from_query(query)
    if not symbols: return {'ok': False, 'error': 'No valid equity tickers/company names found in query'}
    profile = _extract_portfolio_profile(query)
    master_plan = _master_market_agent(query, symbols[:max_stocks], profile)
    market_rows = await asyncio.gather(*[_market_data_agent(sym) for sym in symbols[:max_stocks]])
    screener_rows = await asyncio.gather(*[_screener_agent(sym) for sym in symbols[:max_stocks]])
    valid_rows = [row for row in market_rows if not row.get('error') and row.get('price')]
    if not valid_rows: return {'ok': False, 'error': 'Could not fetch usable market data for the provided stocks'}
    market_view = _market_intelligence_agent(valid_rows, screener_rows, profile)
    fundamental_view = _fundamental_agent(valid_rows)
    risk_view = _risk_guard_agent([{**m, 'raw': m.get('raw', {})} for m in market_view], profile)
    summary, recommendations, disadvantages, ranked = _synthesis_agent(master_plan, market_view, fundamental_view, risk_view)
    return {'ok': True, 'summary': summary + '\n' + '\n'.join(recommendations[:4]) + '\n' + '\n'.join(disadvantages[:4]), 'profile': profile, 'master_plan': master_plan, 'recommendations': recommendations, 'disadvantages': disadvantages, 'agent_outputs': {'market_data_agent': market_rows, 'screener_agent': screener_rows, 'market_intelligence_agent': market_view, 'fundamental_agent': fundamental_view, 'risk_guard_agent': risk_view, 'synthesis_agent': ranked}, 'memory_file': _AGENTIC_MEMORY_FILE}


@app.on_event('startup')
async def startup_event():
    print('FinSage MCP server started - canonical tools: ' + ', '.join(TOOL_NAMES))


@app.post('/tools/amfi_nav', response_model=ToolResponse)
async def amfi_nav(req: AMFIRequest):
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get('https://www.amfiindia.com/spages/NAVAll.txt')
        lines = resp.text.split('\n')
        if req.fund_filter: lines = [l for l in lines if req.fund_filter.lower() in l.lower()]
        sample='\n'.join(lines[:30])
        return ToolResponse(tool='amfi_nav', success=True, data={'summary': 'AMFI NAV for ' + (req.fund_filter or 'all funds') + ' as of ' + datetime.now().strftime('%d-%b-%Y') + ':\n' + sample[:500], 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        return ToolResponse(tool='amfi_nav', success=False, data={}, error=str(e))


@app.post('/tools/search_rag', response_model=ToolResponse)
async def search_rag(req: RAGSearchRequest):
    try:
        if _rag_instance is not None:
            docs, community_ctx, paths = _rag_instance.retrieve(req.query, k=req.top_k)
            summary='\n'.join(d['title'] + ': ' + d['content'][:200] for d in docs)
            return ToolResponse(tool='search_rag', success=True, data={'summary': summary, 'community_ctx': community_ctx[:2], 'kg_paths': [' -> '.join(p) for p in paths[:3]]})
        docs=_fallback_rag(req.query, req.top_k)
        if not docs: return ToolResponse(tool='search_rag', success=False, data={}, error='RAG not initialized and no persisted KG snapshot found')
        return ToolResponse(tool='search_rag', success=True, data={'summary': '\n'.join(d['title'] + ': ' + d['content'][:200] for d in docs), 'community_ctx': [], 'kg_paths': [], 'mode': 'persisted_rag_fallback'})
    except Exception as e:
        return ToolResponse(tool='search_rag', success=False, data={}, error=str(e))


@app.post('/tools/sip_calculator', response_model=ToolResponse)
async def sip_calculator(req: QueryRequest):
    monthly=_amounts(req.query)[0] if _amounts(req.query) else 10000; rate=_pct(req.query)/1200.0; years=_years(req.query); months=int(years*12)
    fv=monthly*months if rate == 0 else monthly*(((1+rate)**months-1)/rate)*(1+rate)
    invested=monthly*months
    return ToolResponse(tool='sip_calculator', success=True, data={'summary': f'SIP plan | Monthly: Rs {monthly:,.0f} | Horizon: {years:.1f} years | Return: {_pct(req.query):.2f}% | Invested: Rs {invested:,.0f} | Estimated value: Rs {fv:,.0f}'})


@app.post('/tools/emi_calculator', response_model=ToolResponse)
async def emi_calculator(req: QueryRequest):
    principal=_amounts(req.query)[0] if _amounts(req.query) else 5000000; rate=_pct(req.query, 8.5)/1200.0; years=_years(req.query, 20.0); months=int(years*12)
    emi=principal/months if rate == 0 else principal*rate*((1+rate)**months)/(((1+rate)**months)-1)
    total=emi*months
    return ToolResponse(tool='emi_calculator', success=True, data={'summary': f'EMI plan | Loan: Rs {principal:,.0f} | Rate: {_pct(req.query, 8.5):.2f}% | Tenure: {years:.1f} years | EMI: Rs {emi:,.0f} | Total interest: Rs {total-principal:,.0f}'})


@app.post('/tools/portfolio_health', response_model=ToolResponse)
async def portfolio_health(req: QueryRequest):
    age=_age(req.query, 30); risk='aggressive' if 'aggressive' in req.query.lower() else 'conservative' if 'conservative' in req.query.lower() else 'moderate'; alloc=_alloc(req.query) or {'equity':60,'debt':25,'gold':10,'cash':5}
    eq=max(20, min(85, 100-age + (10 if risk == 'aggressive' else -15 if risk == 'conservative' else 0))); debt=max(10, 100-eq-10); warn=[]
    if alloc.get('equity', 0) > eq + 15: warn.append('equity above model range')
    if alloc.get('debt', 0) < max(5, debt - 10): warn.append('debt cushion looks light')
    return ToolResponse(tool='portfolio_health', success=True, data={'summary': f"Portfolio review | Age: {age} | Risk: {risk} | Current: {alloc} | Model mix: equity {eq}%, debt {debt}%, gold 10%, cash {max(5,100-eq-debt-10)}% | {' | '.join(warn) if warn else 'Allocation broadly aligned'}"})


@app.post('/tools/goal_planner', response_model=ToolResponse)
async def goal_planner(req: QueryRequest):
    target=_amounts(req.query)[0] if _amounts(req.query) else 20000000; rate=_pct(req.query)/1200.0; years=_years(req.query, 15.0); months=int(years*12)
    sip=target/months if rate == 0 else target/((((1+rate)**months-1)/rate)*(1+rate))
    return ToolResponse(tool='goal_planner', success=True, data={'summary': f'Goal plan | Target corpus: Rs {target:,.0f} | Horizon: {years:.1f} years | Return: {_pct(req.query):.2f}% | Required monthly SIP: Rs {sip:,.0f}'})


@app.post('/tools/screener', response_model=ToolResponse)
async def screener(req: SymbolRequest):
    symbol=req.symbol.upper().strip(); notes=[]
    if os.environ.get('FINSAGE_ENABLE_PLAYWRIGHT_MCP', '1') == '1':
        pw = await _playwright_screener(symbol)
        if pw and not pw.get('error'): return ToolResponse(tool='screener', success=True, data={**pw, 'runtime_note': 'served via MCP server using Playwright'})
        if pw and pw.get('error'): notes.append('playwright_failed=' + pw['error'][:180])
    yf = await _yahoo_screener(symbol)
    if yf: return ToolResponse(tool='screener', success=True, data={**yf, 'runtime_note': 'served via MCP server using Yahoo fallback', 'notes': notes})
    return ToolResponse(tool='screener', success=False, data={'notes': notes}, error='Screener MCP endpoint failed across Playwright and Yahoo fallback')


@app.post('/tools/portfolio_multi_agent', response_model=ToolResponse)
async def portfolio_multi_agent(req: PortfolioWorkflowRequest):
    try:
        result = await _banyantree_agentic_market_workflow(req.query, req.max_stocks, req.symbols)
        if not result.get('ok'):
            return ToolResponse(tool='portfolio_multi_agent', success=False, data={}, error=result.get('error', 'Agentic market workflow failed'))
        return ToolResponse(tool='portfolio_multi_agent', success=True, data={'summary': result['summary'], 'profile': result['profile'], 'master_plan': result['master_plan'], 'recommendations': result['recommendations'], 'disadvantages': result['disadvantages'], 'agent_outputs': result['agent_outputs'], 'memory_file': result['memory_file']})
    except Exception as e:
        return ToolResponse(tool='portfolio_multi_agent', success=False, data={}, error=str(e))


@app.get('/health')
async def health():
    return {'status': 'ok', 'rag_loaded': _rag_instance is not None, 'tools': TOOL_NAMES, 'server': 'canonical_single_mcp_server', 'timestamp': datetime.now().isoformat()}


@app.get('/tools')
async def list_tools():
    return {'server': 'FinSage MCP Tool Server', 'architecture': 'Single MCP server for RAG, planners, market tools, and agentic market workflow.', 'tools': [
        {'name':'amfi_nav','endpoint':'/tools/amfi_nav','method':'POST'},
        {'name':'search_rag','endpoint':'/tools/search_rag','method':'POST'},
        {'name':'screener','endpoint':'/tools/screener','method':'POST'},
        {'name':'sip_calculator','endpoint':'/tools/sip_calculator','method':'POST'},
        {'name':'emi_calculator','endpoint':'/tools/emi_calculator','method':'POST'},
        {'name':'portfolio_health','endpoint':'/tools/portfolio_health','method':'POST'},
        {'name':'goal_planner','endpoint':'/tools/goal_planner','method':'POST'},
        {'name':'portfolio_multi_agent','endpoint':'/tools/portfolio_multi_agent','method':'POST'}]}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=False, workers=1, loop='asyncio')
"""
# â”€â”€ Write to disk â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with open(MCP_SERVER_FILE, "w", encoding="utf-8") as f:
    f.write(MCP_SERVER_CODE)
print("âœ… Server file written")

# â”€â”€ Verify syntax â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
result = subprocess.run(
    [sys.executable, "-c",
     f"import ast; ast.parse(open(r'{MCP_SERVER_FILE}', encoding='utf-8').read())"],
    capture_output=True, text=True
)
if result.returncode == 0:
    print("âœ… Syntax OK")
else:
    print(f"âŒ Syntax error: {result.stderr}")
    import sys; sys.exit()

# â”€â”€ Kill any existing server â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if sys.platform.startswith("win"):
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | "
            "Select-Object -ExpandProperty OwningProcess -Unique | "
            "ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }",
        ],
        capture_output=True,
        text=True,
    )
else:
    subprocess.run(["fuser", "-k", "8000/tcp"], capture_output=True)
time.sleep(2)

# â”€â”€ Start server â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
proc = subprocess.Popen(
    [sys.executable, str(MCP_SERVER_FILE)],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=str(PROJECT_ROOT),
)
time.sleep(5)

# â”€â”€ Verify â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    r = httpx.get(f"{MCP_BASE}/health", timeout=5)
    print(f"âœ… MCP server running: {r.json()}")
    t = httpx.get(f"{MCP_BASE}/tools", timeout=5)
    print(f"âœ… Tools: {[x['name'] for x in t.json()['tools']]}")

    print("\nðŸ§ª Testing AMFI...")
    a = httpx.post(f"{MCP_BASE}/tools/amfi_nav",
                   json={"fund_filter": "ELSS"}, timeout=20)
    print(f"âœ… AMFI: {a.json()['data']['summary'][:100]}")

    print("\nðŸ§ª Testing SIP calculator...")
    s = httpx.post(f"{MCP_BASE}/tools/sip_calculator",
                   json={"query": "SIP 5000 per month 10 years 12%"}, timeout=10)
    print(f"âœ… SIP: {s.json()['data']['summary']}")

    print("\nðŸ§ª Testing EMI calculator...")
    e = httpx.post(f"{MCP_BASE}/tools/emi_calculator",
                   json={"query": "home loan 50 lakh 8.5% 20 years"}, timeout=10)
    print(f"âœ… EMI: {e.json()['data']['summary']}")

except Exception as ex:
    out, err = proc.stdout.read(), proc.stderr.read()
    print(f"âŒ {ex}\nSTDERR: {err.decode()[:1000]}")
	
# =================================================================
# LOCAL DEPENDENCY CHECK + IMPORTS
# =================================================================
try:
    import torch
    import faiss
    import numpy as np
    import spacy
    import networkx as nx
    from sklearn.cluster import KMeans
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError as exc:
    missing = exc.name or "required package"
    raise SystemExit(
        "\nMissing Python dependency: "
        f"{missing}\n\n"
        "This usually means VS Code is running global Python instead of this project's .venv, "
        "or setup has not been run yet.\n\n"
        "Fix from PowerShell:\n"
        f"  cd {PROJECT_ROOT}\n"
        "  powershell -ExecutionPolicy Bypass -File .\\scripts\\setup.ps1\n"
        "  powershell -ExecutionPolicy Bypass -File .\\scripts\\run.ps1\n\n"
        "In VS Code, also select interpreter:\n"
        "  .venv\\Scripts\\python.exe\n"
    )

import asyncio, json, re
from datetime import datetime
from collections import defaultdict
from typing import Optional, List
try:
    from langgraph.types import interrupt as _langgraph_interrupt
except Exception:
    _langgraph_interrupt = None

os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf_cache"))

MCP_BASE = os.environ.get("BANYANTREE_MCP_BASE", "http://localhost:8000")
QWEN_MODEL_ID = os.environ.get("BANYANTREE_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
EMBEDDING_MODEL_ID = os.environ.get("BANYANTREE_EMBEDDING_MODEL", "BAAI/bge-m3")
LLM_PROVIDER = os.environ.get("BANYANTREE_LLM_PROVIDER", "local").strip().lower()
API_MODEL_ID = os.environ.get("BANYANTREE_API_MODEL", "gpt-4o-mini")
API_BASE_URL = os.environ.get("BANYANTREE_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
API_KEY = os.environ.get("BANYANTREE_API_KEY") or os.environ.get("OPENAI_API_KEY")
USE_API_LLM = LLM_PROVIDER in {"api", "openai", "openai-compatible", "openai_compatible"}

if USE_API_LLM:
    print(f"LLM    : API model ({API_MODEL_ID})")
elif not torch.cuda.is_available():
    print("\nCUDA GPU not available.")
    print("To run without a GPU, edit .env and set BANYANTREE_LLM_PROVIDER=api plus BANYANTREE_API_KEY.")
    raise SystemExit(0)
else:
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy  : {np.__version__}")

def build_qwen_prompt(tokenizer, system: str, user: str) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def format_retrieved_doc_for_prompt(doc: dict, index: int, max_chars: int = 420) -> str:
    src = doc.get("source", "vector+bm25")
    metadata = doc.get("metadata") or {}
    label = " [LIVE DATA]" if src == "mcp_tool" else ""
    lines = [f"[Doc {index}{label}]", f"Title: {doc.get('title', 'Untitled')}", f"Source: {src}"]
    source_type = metadata.get("source_type")
    if source_type:
        lines.append(f"Source type: {source_type}")
    category = metadata.get("category")
    if category:
        lines.append(f"Category: {category}")
    section_path = metadata.get("section_path") or []
    if section_path:
        if isinstance(section_path, list):
            lines.append("Section path: " + " > ".join(str(part) for part in section_path))
        else:
            lines.append(f"Section path: {section_path}")
    page_start, page_end = metadata.get("page_start"), metadata.get("page_end")
    if page_start or page_end:
        lines.append(f"Pages: {page_start or '?'}-{page_end or page_start or '?'}")
    node_id = metadata.get("node_id")
    if node_id:
        lines.append(f"Node ID: {node_id}")
    content = str(doc.get("content", "")).replace(chr(10), " ").strip()[:max_chars]
    lines.append(f"Content: {content}")
    return "\n".join(lines)

class OpenAICompatibleChatLLM:
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1"):
        if not api_key or api_key.lower() in {"replace_me", "your_api_key_here"}:
            raise SystemExit(
                "\nAPI model mode is enabled, but no API key is configured.\n\n"
                "Edit .env and set:\n"
                "  BANYANTREE_API_KEY=your_real_api_key\n\n"
                "Then rerun:\n"
                "  powershell -ExecutionPolicy Bypass -File .\\scripts\\run.ps1\n"
            )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        print(f"API LLM ready: {self.model} via {self.base_url}")

    def complete(self, system: str, user: str, max_tokens: int = 160, temperature: float = 0.0) -> str:
        # Prepend /no_think to suppress Qwen3 extended thinking (<think> blocks).
        # This is a Qwen3 control token; it has no effect on other models.
        user_content = f"/no_think\n{user}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_exc: Exception | None = None
        for attempt in range(1, 4):  # 3 retries — mirrors LLMClient in app/api/llm_client.py
            try:
                with httpx.Client(timeout=120) as client:  # 120 s — same as LLMClient
                    response = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    raw_content = data["choices"][0]["message"]["content"].strip()
                    # Strip Qwen3 extended-thinking blocks before returning.
                    # Handle both complete (<think>...</think>) and truncated (no closing tag).
                    raw_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL)
                    raw_content = re.sub(r"<think>.*$", "", raw_content, flags=re.DOTALL)
                    return raw_content.strip()
            except Exception as exc:
                last_exc = exc
                print(f"API model call attempt {attempt}/3 failed: {exc}")
                if attempt < 3:
                    time.sleep(2 ** attempt)  # 2 s, 4 s backoff
        raise RuntimeError(f"API model call failed: {last_exc}") from last_exc



# =================================================================
# MODULE 1 : PII REDACTOR
# =================================================================
class FinancialPIIRedactor:
    def __init__(self):
        try:
            from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig
            _nlp_config = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            _nlp_engine = NlpEngineProvider(nlp_configuration=_nlp_config).create_engine()
            self.analyzer       = AnalyzerEngine(nlp_engine=_nlp_engine)
            self.anonymizer     = AnonymizerEngine()
            self.OperatorConfig = OperatorConfig
            self._add_indian_patterns()
            self.enabled = True
            print("PII Redactor ready")
        except Exception as e:
            print(f"Presidio not available ({e}), using regex fallback")
            self.enabled = False

    def _add_indian_patterns(self):
        from presidio_analyzer import PatternRecognizer, Pattern
        for entity, regex, score in [
            ("AADHAAR_NUMBER", r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b", 0.95),
            ("PAN_NUMBER",     r"\b[A-Z]{5}\d{4}[A-Z]{1}\b",       0.95),
            ("INDIAN_PHONE",   r"\b[6-9]\d{9}\b",                   0.85),
            ("IFSC_CODE",      r"\b[A-Z]{4}0[A-Z0-9]{6}\b",         0.90),
        ]:
            self.analyzer.registry.add_recognizer(
                PatternRecognizer(supported_entity=entity, patterns=[Pattern(name=entity.lower(), regex=regex, score=score)])
            )

    def redact(self, text: str) -> tuple:
        if not self.enabled:
            text = re.sub(r'\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b', '<AADHAAR>', text)
            text = re.sub(r'\b[A-Z]{5}\d{4}[A-Z]{1}\b',       '<PAN>',     text)
            text = re.sub(r'\b[6-9]\d{9}\b',                   '<PHONE>',   text)
            return text, []
        entities = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "AADHAAR_NUMBER", "PAN_NUMBER", "INDIAN_PHONE", "IFSC_CODE"]
        results = self.analyzer.analyze(text=text, entities=entities, language="en")
        ops = {e: self.OperatorConfig("replace", {"new_value": f"<{e}>"}) for e in entities}
        anon = self.anonymizer.anonymize(text=text, analyzer_results=results, operators=ops)
        return anon.text, [{"type": r.entity_type, "score": r.score} for r in results]

# =================================================================
# MODULE 2 : FINANCIAL ENTITY EXTRACTOR
# =================================================================
class FinancialEntityExtractor:
    def __init__(self, nlp=None):
        self.nlp = nlp
        self.financial_patterns = {
            "INVESTMENT_TYPE": r"\b(SIP|PPF|ELSS|NPS|FD|RD|Mutual Fund|Equity|Debt Fund)\b",
            "TAX_SECTION": r"\b(Section\s*\d+[A-Za-z]?|80C|80D|80CCD|24(b)|87A)\b",
            "AMOUNT": r"Ã¢â€šÂ¹?\s*[\d,]+(?:\s*(?:lakh|crore|thousand))?\b",
            "FINANCIAL_INSTITUTION": r"\b(HDFC|ICICI|SBI|Axis|Kotak|PPF|EPF|NPS|SEBI|RBI)\b",
            "FINANCIAL_TERM": r"\b(EMI|LTCG|STCG|CAGR|NAV|XIRR|Asset Allocation|Rebalancing)\b",
            "LOAN_TYPE": r"\b(Home Loan|Car Loan|Personal Loan|Education Loan)\b",
            "INSURANCE_TYPE": r"\b(Term Insurance|Health Insurance|Life Insurance|ULIP)\b",
            "STOCK_TICKER": r"\b(NSE|BSE):\s*[A-Z]{2,10}\b",
        }

    def extract(self, text: str) -> set:
        entities = set()
        for entity_type, pattern in self.financial_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.add((match.group().lower(), entity_type))
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "MONEY", "DATE"]:
                    entities.add((ent.text.lower(), f"NER_{ent.label_}"))
        return entities

# =================================================================
# MODULE 3 : QWEN CLASSIFIER
# =================================================================
class QwenClassifier:
    # Guardrail-only: BLOCK or OK. No intent, no routing, no tool selection.
    FEW_SHOT = """Examples:
Q: "How do I hide income from IT department?"
A: {"guardrail":"BLOCK","reason":"TAX_EVASION"}
Q: "What is TCS share price today?"
A: {"guardrail":"OK","reason":""}
Q: "guaranteed 30% returns on crypto"
A: {"guardrail":"BLOCK","reason":"SCAM_RETURNS"}
Q: "Compare Infosys and Wipro PE ratio"
A: {"guardrail":"OK","reason":""}
Q: "How do I do pump and dump on penny stocks?"
A: {"guardrail":"BLOCK","reason":"MARKET_MANIPULATION"}
Q: "insider tip on HDFC Bank before results"
A: {"guardrail":"BLOCK","reason":"INSIDER_TRADING"}
Q: "pyramid scheme investment opportunity"
A: {"guardrail":"BLOCK","reason":"PONZI"}
Q: "non public information about reliance merger"
A: {"guardrail":"BLOCK","reason":"INSIDER_TRADING"}
Q: "How to start SIP in ELSS funds?"
A: {"guardrail":"OK","reason":""}
Q: "how to calculate EMI for home loan?"
A: {"guardrail":"OK","reason":""}"""

    SYSTEM = (
        "You are a safety guardrail for an Indian personal finance app.\n"
        "Output ONLY a compact JSON with exactly two keys:\n"
        "  guardrail : OK or BLOCK\n"
        "  reason    : if BLOCK, one of: TAX_EVASION, SCAM_RETURNS, INSIDER_TRADING, "
        "MARKET_MANIPULATION, PONZI -- else empty string\n"
        "BLOCK only for illegal/fraudulent/unethical requests. "
        "All legitimate finance questions (stocks, SIP, tax, loans, budgeting) are OK.\n"
        "Output ONLY the JSON. No explanation. No markdown."
    )

    GUARDRAIL_FALLBACK = (
        "I cannot help with that request.\n\n"
        "I can assist with: budgeting, legal tax saving (80C/80D/NPS), goal-based investing, "
        "mutual funds, debt management, and reading live stock/market data.\n\n"
        "For personalized advice, consult a SEBI-registered advisor."
    )

    def __init__(self, tokenizer, generator, device: str, api_client=None):
        self.tokenizer  = tokenizer
        self.generator  = generator
        self.device     = device
        self.api_client = api_client
        print("API Guardrail ready" if self.api_client else "Qwen2.5 Guardrail ready")

    def classify(self, query: str) -> dict:
        user_msg = f"{self.FEW_SHOT}\n\nNow classify:\nQ: \"{query}\"\nA:"
        if self.api_client:
            raw = self.api_client.complete(self.SYSTEM, user_msg, max_tokens=400, temperature=0.0)
        else:
            prompt = build_qwen_prompt(self.tokenizer, self.SYSTEM, user_msg)
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2000,
                                    truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.generator.generate(
                    **inputs, max_new_tokens=30, temperature=0.0, do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id)
            raw = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"Guardrail: {raw}")
        return self._parse(raw, query)

    def _parse(self, raw: str, query: str) -> dict:
        # After think-block stripping in complete(), raw should be clean JSON.
        # We intentionally do NOT do keyword matching on raw text — the LLM's
        # <think> reasoning may contain words like "scam" or "insider" while
        # correctly concluding the query is OK. Keyword fallback caused false BLOCKs.
        try:
            m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
            if m:
                result    = json.loads(m.group())
                guardrail = str(result.get("guardrail", "OK")).upper()
                reason    = str(result.get("reason", ""))
                if guardrail not in ("OK", "BLOCK"):
                    guardrail = "OK"
                return {"guardrail": guardrail, "reason": reason}
        except Exception:
            pass
        # No JSON found (e.g. think block exceeded token limit, stripped to empty).
        # Default to OK — it is safer to let a legitimate query through than to
        # false-block based on heuristics. Real BLOCK cases always return explicit JSON.
        print("Guardrail parse failed — defaulting to OK")
        return {"guardrail": "OK", "reason": ""}


# =================================================================
# MODULE 4 : QWEN MCP TOOL SELECTOR
# =================================================================
class QwenMCPToolSelector:
    TOOL_MANIFEST = """Available tools Ã¢â‚¬â€ output JSON array of calls needed:
Tool: screener
When: user asks about share price, PE ratio, 52W high/low, fundamentals, revenue, profit, market cap, quarterly results
Call: {"tool":"screener","symbol":"TICKER"}
Note: Use NSE ticker symbols; when company names are ambiguous, rely on yfinance-validated symbol resolution.
Tool: amfi_nav
When: user asks about mutual fund NAV or scheme prices
Call: {"tool":"amfi_nav","fund_filter":"FUND_NAME"}
Tool: search_rag
When: personal finance concepts Ã¢â‚¬â€ SIP, PPF, tax, budget, EMI, insurance
Call: {"tool":"search_rag","query":"YOUR_QUERY"}
Examples:
Q: "TCS share price today"
A: [{"tool":"portfolio_multi_agent","query":"TCS share price today"}]
Q: "Compare INFY and WIPRO PE ratio"
A: [{"tool":"screener","symbol":"INFY"},{"tool":"screener","symbol":"WIPRO"}]
Q: "ELSS NAV today"
A: [{"tool":"amfi_nav","fund_filter":"ELSS"}]
Q: "What is 80C tax limit?"
A: [{"tool":"search_rag","query":"80C tax deduction limit"}]
Q: "TCS PE and fundamentals"
A: [{"tool":"screener","symbol":"TCS"}]
Q: "hdfc bank share price"
A: [{"tool":"screener","symbol":"HDFCBANK"}]
Q: "Mirae ELSS NAV"
A: [{"tool":"amfi_nav","fund_filter":"Mirae ELSS"}]
Q: "nifty 50 index value"
A: [{"tool":"screener","symbol":"NIFTY"}]"""

    def __init__(self, tokenizer, generator, device: str, api_client=None):
        self.tokenizer = tokenizer
        self.generator = generator
        self.device    = device
        self.api_client = api_client
        print("API MCP Tool Selector ready" if self.api_client else "Qwen2.5 MCP Tool Selector ready")

    def select_tools(self, query: str) -> list:
        system_msg = "You are a tool selector. Given a user query, output ONLY a JSON array of tool calls needed. No explanation. No markdown."
        user_msg = f"{self.TOOL_MANIFEST}\n\nQuery: {query}\nTool calls needed:"
        if self.api_client:
            raw = self.api_client.complete(system_msg, user_msg, max_tokens=400, temperature=0.0)
        else:
            prompt = build_qwen_prompt(self.tokenizer, system_msg, user_msg)
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2000, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.generator.generate(**inputs, max_new_tokens=120, temperature=0.0, do_sample=False,
                                                  eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id)
            raw = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"tool selection: {raw}")
        return self._parse_tool_calls(raw)

    def _parse_tool_calls(self, raw: str) -> list:
        try:
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if match:
                calls = json.loads(match.group())
                if isinstance(calls, list): return [c for c in calls if isinstance(c, dict) and "tool" in c][:4]
        except Exception: pass
        try:
            match = re.search(r'\{[^}]+\}', raw)
            if match:
                call = json.loads(match.group())
                if "tool" in call: return [call]
        except Exception: pass
        print("Tool selection parse failed empty list")
        return []

# =================================================================
# MODULE 5 MCP TOOL CLIENT
# =================================================================
class LangGraphToolApprovalMiddleware:
    """Central policy layer for LangGraph-style tool approval and guardrails."""

    DEFAULT_POLICY = {
        "requires_permission": ["NA"],
        "auto_approved": ["search_rag", "sip_calculator", "emi_calculator", "portfolio_health", "goal_planner","portfolio_multi_agent", "screener", "amfi_nav"],
        "dangerous_patterns": [
            r"rm\s+-rf", r"del\s+/[sq]", r"format\s+[a-z]:", r"powershell", r"cmd\.exe",
            r"subprocess", r"os\.system", r"eval\s*\(", r"exec\s*\(", r"curl\s+", r"wget\s+",
            r"download\s+and\s+run", r"shell\s+command", r"insider\s+tip", r"inside\s+information",
            r"pump\s+and\s+dump", r"manipulate\s+stock", r"hide\s+income", r"tax\s+evasion",
        ],
    }

    def __init__(self, policy_path: str | None = None, trace_path: str | None = None, registered_tools: set | None = None):
        base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        self.policy_path = policy_path or os.environ.get("BANYANTREE_TOOL_POLICY_PATH", os.path.join(base_dir, "tool_permission_policy.json"))
        self.trace_path = trace_path or os.environ.get("BANYANTREE_TRACE_PATH", os.path.join(os.getcwd(), "banyantree_tool_trace.jsonl"))
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.registered_tools = set(registered_tools or [])
        self.policy = self._load_policy()

    def _load_policy(self) -> dict:
        policy = dict(self.DEFAULT_POLICY)
        try:
            if os.path.exists(self.policy_path):
                with open(self.policy_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                for key in ("requires_permission", "auto_approved", "dangerous_patterns"):
                    if isinstance(loaded.get(key), list):
                        policy[key] = loaded[key]
        except Exception as e:
            print(f"Tool approval policy load failed, using defaults: {e}")
        policy["requires_permission"] = set(policy.get("requires_permission", []))
        policy["auto_approved"] = set(policy.get("auto_approved", []))
        return policy

    def before_tool_call(self, tool_call: dict) -> dict:
        """LangGraph middleware entry point called immediately before each tool execution."""
        tool = str(tool_call.get("tool", "")).strip()
        self._trace("tool_queued", tool_call)

        if not tool:
            return self._decision(False, "Missing tool name", tool_call)
        if self.registered_tools and tool not in self.registered_tools:
            return self._decision(False, f"Unknown or unregistered tool: {tool}", tool_call)

        blocked_reason = self._dangerous_payload_reason(tool_call)
        if blocked_reason:
            return self._decision(False, blocked_reason, tool_call)

        if tool in self.policy["auto_approved"]:
            return self._decision(True, "auto-approved by JSON policy", tool_call, permission_required=False)
        if tool in self.policy["requires_permission"]:
            approved, reason = self._request_human_approval(tool_call)
            return self._decision(approved, reason, tool_call, permission_required=True)

        return self._decision(False, f"Tool {tool} is not classified in JSON policy", tool_call)

    def _dangerous_payload_reason(self, tool_call: dict) -> str | None:
        payload = json.dumps(tool_call, default=str).lower()
        for pattern in self.policy.get("dangerous_patterns", []):
            if re.search(pattern, payload, re.I):
                return f"Blocked dangerous payload pattern: {pattern}"
        return None

    def _request_human_approval(self, tool_call: dict) -> tuple[bool, str]:
        if os.environ.get("BANYANTREE_REQUIRE_TOOL_APPROVAL", "1") == "0":
            return True, "auto-approved by BANYANTREE_REQUIRE_TOOL_APPROVAL=0"

        approval_payload = {
            "kind": "tool_approval",
            "message": "Approve the next BanyanTree tool execution?",
            "tool_call": tool_call,
        }
        if _langgraph_interrupt is not None:
            try:
                response = _langgraph_interrupt(approval_payload)
                if isinstance(response, dict):
                    return bool(response.get("approve") or response.get("approved")), str(response.get("reason", "langgraph human approval"))
                if isinstance(response, str):
                    return response.strip().lower() in {"y", "yes", "approve", "approved"}, "langgraph human approval"
            except Exception as e:
                self._trace("langgraph_interrupt_fallback", tool_call, error=str(e))

        try:
            print("\nHuman approval required before tool execution")
            print(json.dumps(tool_call, indent=2, default=str))
            answer = input("Approve this tool call? [y/N]: ").strip().lower()
            return answer in {"y", "yes", "approve", "approved"}, "console approval"
        except Exception as e:
            return False, f"approval unavailable: {e}"

    def _decision(self, allowed: bool, reason: str, tool_call: dict, permission_required: bool = False) -> dict:
        event = "tool_allowed" if allowed else "tool_blocked"
        self._trace(event, tool_call, reason=reason, permission_required=permission_required)
        return {"allowed": allowed, "reason": reason, "permission_required": permission_required}

    def _trace(self, event: str, tool_call: dict | None = None, **extra) -> None:
        record = {
            "ts": datetime.now().isoformat(),
            "run_id": self.run_id,
            "middleware": "LangGraphToolApprovalMiddleware",
            "event": event,
            "tool_call": tool_call or {},
            **extra,
        }
        try:
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            print(f"Middleware trace write failed: {e}")

class MCPToolClient:
    MCP_ENDPOINTS = {"amfi_nav": "/tools/amfi_nav", "search_rag": "/tools/search_rag"}

    def __init__(self, mcp_base: str = "http://localhost:8000"):
        self.mcp_base = mcp_base
        self.tool_middleware = LangGraphToolApprovalMiddleware(registered_tools=self._registered_tools())

    def _registered_tools(self) -> set:
        policy_tools = set(LangGraphToolApprovalMiddleware.DEFAULT_POLICY["requires_permission"])
        policy_tools |= set(LangGraphToolApprovalMiddleware.DEFAULT_POLICY["auto_approved"])
        return set(self.MCP_ENDPOINTS.keys()) | policy_tools

    # Screener: local Playwright + BeautifulSoup + Regex Fallback #
    async def _local_screener(self, symbol: str) -> dict | None:
        from playwright.async_api import async_playwright
        from bs4 import BeautifulSoup
        import pandas as pd
        from io import StringIO

        symbol = symbol.upper().strip()
        print(f" Local Playwright at Screener {symbol}")

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True, args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"]
                )
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                )
                page = await context.new_page()
                await page.goto(f"https://www.screener.in/company/{symbol}/", wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(4)

                html = await page.content()
                if "Error 404" in html or "No CompanyCode matches" in html:
                    print(f"{symbol} not found on screener.in")
                    await browser.close()
                    return {"summary": f"{symbol} is not available on screener.in. Check NSE/BSE directly.", "symbol": symbol, "key_ratios": "", "pl_summary": ""}

                soup = BeautifulSoup(html, 'html.parser')
                full_text = soup.get_text(separator=' ', strip=True)

                # Extract Price
                price = "N/A"
                price_el = soup.select_one("div.price, span.price, .current-price, h1 + p")
                if price_el:
                    m = re.search(r'Ã¢â€šÂ¹\s*([\d,]+\.?\d*)', price_el.text)
                    if m: price = m.group(1)
                if price == "N/A":
                    m = re.search(r'(?:Current Price|Close Price).*?Ã¢â€šÂ¹\s*([\d,]+\.?\d*)', full_text, re.I)
                    if not m: m = re.search(r'Ã¢â€šÂ¹\s*([\d,]+\.?\d*)\s*[-Ã¢â‚¬â€œ]', full_text)
                    if m: price = m.group(1)

                # Extract Change %
                change = "N/A"
                change_el = soup.select_one("span.change, .percent-change")
                if change_el: change = change_el.text.strip()

                # Extract Ratios
                ratios = {}
                ratios_section = soup.find(id="top-ratios")
                if ratios_section:
                    table = ratios_section.find('table')
                    if table:
                        for row in table.find_all('tr'):
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 2:
                                key = cells[0].text.strip().lower().replace(':', '')
                                val = cells[1].text.strip()
                                if key and val: ratios[key] = val

                # Regex fallback for missing ratios
                if len(ratios) < 3:
                    for key, pattern in [("pe", r'P/E\s*(?:TTM)?\s*[:\s]*([\d.]+)'), ("market cap", r'Market\s*Cap\s*[:\s]*Ã¢â€šÂ¹\s*([\d,]+\.?\d*)\s*Cr'),
                                         ("price to book", r'Price\s*to\s*Book\s*[:\s]*([\d.]+)'), ("dividend yield", r'Dividend\s*Yield\s*[:\s]*([\d.]+%?)')]:
                        if key not in ratios:
                            m = re.search(pattern, full_text, re.I)
                            if m: ratios[key] = m.group(1).strip()

                # Build RAG Summary
                clean_ratios = []
                for k in ['pe', 'market cap', 'price to book', 'dividend yield']:
                    if k in ratios: clean_ratios.append(f"{k.upper().replace(' ', '_')}: {ratios[k]}")

                summary = (f"{symbol} | NSE | Price: Ã¢â€šÂ¹{price} ({change}) | " + " | ".join(clean_ratios) +
                           f" | Source: screener.in | {datetime.now().strftime('%d-%b-%Y')}")[:1200]
                print(f"   screener {symbol}: Price=Ã¢â€šÂ¹{price}, PE={ratios.get('pe', 'N/A')}")
                await browser.close()
                return {"summary": summary, "symbol": symbol, "key_ratios": " | ".join(clean_ratios), "pl_summary": ""}

        except Exception as e:
            print(f"Local Screener failed ({symbol}): {e}")
            return None

    # Yahoo Finance Fallback #
    async def _yahoo_finance_fallback(self, symbol: str) -> dict | None:
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            info = ticker.info
            if not info.get('currentPrice'): return None
            ratios = []
            if info.get('currentPrice'): ratios.append(f"PRICE: {info['currentPrice']}")
            if info.get('trailingPE'): ratios.append(f"PE: {info['trailingPE']:.2f}")
            if info.get('marketCap'): ratios.append(f"MARKET_CAP: {info['marketCap']/1e7:.0f}Cr")
            if info.get('dividendYield'): ratios.append(f"DIV_YIELD: {info['dividendYield']*100:.2f}%")
            summary = (f"{symbol} (Yahoo) | NSE | " + " | ".join(ratios) + f" | Source: finance.yahoo.com | {datetime.now().strftime('%d-%b-%Y')}")
            print(f"   Yahoo fallback {symbol}: {ratios[:2]}")
            return {"summary": summary, "symbol": symbol, "key_ratios": " | ".join(ratios), "pl_summary": ""}
        except Exception as e:
            print(f"Yahoo fallback failed: {e}")
            return None

    # AMFI + search_rag: HTTP to MCP server #
    async def _call_mcp(self, tool_call: dict) -> dict | None:
        tool = tool_call.get("tool", "")
        if tool not in self.MCP_ENDPOINTS:
            print(f"Unknown MCP tool: {tool}")
            return None
        body = {k: v for k, v in tool_call.items() if k != "tool"}
        print(f" MCP Ã¢â€ â€™ POST {self.mcp_base}{self.MCP_ENDPOINTS[tool]} {body}")
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(f"{self.mcp_base}{self.MCP_ENDPOINTS[tool]}", json=body)
                result = resp.json()  # read inside context so body is available
            if result.get("success"):
                print(f"   {tool}: success")
                return result["data"]
            else:
                print(f"{tool} failed: {result.get('error')}")
                return None
        except httpx.ConnectError:
            print(f" MCP server not reachable at {self.mcp_base}")
            return None
        except Exception as e:
            print(f"MCP call failed ({tool}): {e}")
            return None

    # Route each tool call #
    async def _call_one(self, tool_call: dict) -> dict | None:
        tool = tool_call.get("tool", "")
        if tool == "screener":
            result = await self._local_screener(tool_call.get("symbol", ""))
            if not result or "not available" in result.get("summary", "").lower():
                print(f" Falling back to Yahoo Finance for {tool_call.get('symbol')}")
                return await self._yahoo_finance_fallback(tool_call.get("symbol", ""))
            return result
        return await self._call_mcp(tool_call)

    # Execute tool calls through the LangGraph middleware policy before invoking tools.
    async def execute(self, tool_calls: list) -> list:
        if not tool_calls: return []
        docs = []
        for tc in tool_calls:
            decision = self.tool_middleware.before_tool_call(tc)
            if not decision.get("allowed"):
                print(f"Tool execution skipped by middleware: {decision.get('reason')}")
                continue
            try:
                self.tool_middleware._trace("tool_start", tc)
                result = await self._call_one(tc)
                self.tool_middleware._trace("tool_end", tc, success=bool(result))
            except Exception as e:
                self.tool_middleware._trace("tool_error", tc, error=str(e))
                continue
            if not isinstance(result, dict) or not result:
                continue
            tool = tc.get("tool", "unknown")
            subject = tc.get("symbol", tc.get("query", tc.get("fund_filter", ""))[:40])
            docs.append({"title": f"{tool}: {subject}", "content": result.get("summary", str(result)[:800]), "source": "mcp_tool"})
        print(f"   MCPToolClient: {len(docs)} docs from {len(tool_calls)} requested tool calls")
        return docs
# =================================================================
# MODULE 6 Ã¢â‚¬â€ OUTPUT GUARDRAIL
# =================================================================
class OutputGuardrail:
    BLOCK_OUT = [(r"guaranteed?\s+\d+%\s+(return|profit)", "HALLUCINATED_RETURN"), (r"i\s+am\s+(a\s+)?sebi.?registered", "FALSE_CLAIM")]
    DISCLAIMER = "\nConsult a SEBI-registered advisor for personalized financial advice."
    def check(self, response: str) -> tuple:
        for pattern, cat in self.BLOCK_OUT:
            if re.search(pattern, response, re.IGNORECASE): return False, cat
        return True, "OK"

# =================================================================
# MODULE 7 Ã¢â‚¬â€ CONVERSATION MEMORY
# =================================================================
class ConversationMemory:
    def __init__(self, max_turns: int = 5):
        self.history, self.summary, self.max_turns = [], "", max_turns
    def add(self, question: str, answer: str):
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_turns:
            old = self.history.pop(0)
            self.summary += f"Discussed: {old['question'][:80]}. Key: {old['answer'][:120]}. "
            if len(self.summary) > 500: self.summary = self.summary[-500:]
    def get_context(self) -> str:
        if not self.history and not self.summary: return ""
        parts = []
        if self.summary: parts.append(f"[Earlier summary]\n{self.summary}")
        if self.history:
            parts.append("[Recent exchanges]")
            for ex in self.history[-3:]:
                parts.append(f"User: {ex['question']}")
                parts.append(f"FinSage: {ex['answer'][:200]}...")
        return "\n".join(parts)
    def clear(self):
        self.history, self.summary = [], ""

# =================================================================
# MAIN CLASS Ã¢â‚¬â€ FINANCIAL HIERARCHICAL LIGHT RAG
# =================================================================
class FINANCIAL_HIERARCHICAL_LIGHT_RAG:
    def __init__(self, kg_db_path: str = "finsage_kg_database"):
        self.device = "api" if USE_API_LLM else ("cuda" if torch.cuda.is_available() else "cpu")
        llm_note = f"API model: {API_MODEL_ID}" if USE_API_LLM else "Local Qwen model"
        print(f"\n{'='*60}\n  FinSage Financial LightRAG Final\n  Device  : {self.device}\n  Screener: local Playwright (notebook process)\n  AMFI    : MCP server (httpx)\n  LLM     : {llm_note}\n{'='*60}")
        self.financial_kg_root = FINANCIAL_KG_ROOT
        self.kg_db_path = str((FINANCIAL_KG_ROOT / "lightrag").resolve())
        self.graph_db_path = str((FINANCIAL_KG_ROOT / "graph").resolve())
        self.legacy_kg_db_path = str((KG_ROOT / kg_db_path).resolve())
        os.makedirs(self.kg_db_path, exist_ok=True)
        os.makedirs(self.graph_db_path, exist_ok=True)
        self.pii_redactor = FinancialPIIRedactor()
        self.output_guard = OutputGuardrail()
        self.memory = ConversationMemory(max_turns=5)
        try: self.nlp = spacy.load("en_core_web_sm")
        except Exception: self.nlp = None; print("spaCy not found")
        self.entity_extractor = FinancialEntityExtractor(self.nlp)

        print(f"\nLoading embedding model: {EMBEDDING_MODEL_ID}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_ID, cache_folder=str(DATA_DIR / "sentence_transformers"))
        print("BGE-M3 loaded (1024-dim)")

        self.llm_client = None
        if USE_API_LLM:
            self.tokenizer = None
            self.generator = None
            self.llm_client = OpenAICompatibleChatLLM(API_KEY, API_MODEL_ID, API_BASE_URL)
            self._build_prompt = lambda system, user: f"System:\n{system}\n\nUser:\n{user}"
        else:
            print(f"\nLoading {QWEN_MODEL_ID} in 4-bit...")
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
            self.tokenizer.padding_side = "right"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.generator = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            self.generator.eval()
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"{QWEN_MODEL_ID} loaded | VRAM: {used:.1f}/{total:.1f} GB")
            self._build_prompt = lambda system, user: build_qwen_prompt(self.tokenizer, system, user)

        self.classifier = QwenClassifier(self.tokenizer, self.generator, self.device, self.llm_client)
        self.tool_selector = QwenMCPToolSelector(self.tokenizer, self.generator, self.device, self.llm_client)
        self.mcp_client = MCPToolClient(mcp_base=MCP_BASE)
        print(f"MCP client Ã¢â€ â€™ {MCP_BASE}")

        try:
            from flashrank import Ranker
            self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=str(DATA_DIR / "reranker"))
            self.rerank_enabled = True
            print("FlashRank re-ranker loaded")
        except Exception as e:
            self.rerank_enabled = False
            print(f"Re-ranker unavailable ({e})")

        self.documents, self.titles, self.document_metadata = [], [], []
        self.document_embeddings = None
        self.neighbor_index = None
        self.community_index = None
        self.communities = []
        self.community_summaries = {}
        self.kg = nx.Graph()
        self.entity_contexts = defaultdict(list)
        self.entity_cooccurrence = defaultdict(int)
        self.query_history = []
        self.bm25 = None
        self.raptor_root = ""
        self.load_kg_database()
        print(f"\nFinSage ready\n")

    def save_kg_database(self):
        print("Saving...")
        try:
            graph_saves = {
                "knowledge_graph.json": nx.node_link_data(self.kg),
                "entity_contexts.json": dict(self.entity_contexts),
                "entity_cooccurrence.json": {f"{k[0]}||{k[1]}": v for k, v in self.entity_cooccurrence.items()},
            }
            lightrag_saves = {
                "query_history.json": self.query_history[-100:],
                "community_summaries.json": {str(k): v for k, v in self.community_summaries.items()},
                "documents.json": self.documents, "titles.json": self.titles,
                "document_metadata.json": self.document_metadata,
                "metadata.json": {"last_updated": datetime.now().isoformat(), "total_entities": self.kg.number_of_nodes(),
                                  "total_relationships": self.kg.number_of_edges(), "total_documents": len(self.documents),
                                  "lightrag_path": self.kg_db_path, "graph_path": self.graph_db_path}
            }
            for fname, data in graph_saves.items():
                with open(f"{self.graph_db_path}/{fname}", "w") as f: json.dump(data, f, indent=2)
            for fname, data in lightrag_saves.items():
                with open(f"{self.kg_db_path}/{fname}", "w") as f: json.dump(data, f, indent=2)
            if self.neighbor_index is not None: faiss.write_index(self.neighbor_index, f"{self.kg_db_path}/faiss_index.index")
            if self.document_embeddings is not None: np.save(f"{self.kg_db_path}/document_embeddings.npy", self.document_embeddings)
            if len(self.communities) > 0: np.save(f"{self.kg_db_path}/communities.npy", np.array(self.communities))
            print(f"   {self.kg.number_of_nodes()} entities | {self.kg.number_of_edges()} relationships")
        except Exception as e: print(f"Save error: {e}")

    def load_kg_database(self):
        print("Loading saved KG database (if exists)...")
        try:
            loaders = {
                "query_history.json": lambda d: setattr(self, "query_history", d),
                "community_summaries.json": lambda d: setattr(self, "community_summaries", {int(k): v for k, v in d.items()}),
                "document_metadata.json": lambda d: setattr(self, "document_metadata", d if isinstance(d, list) else []),
                "documents.json": lambda d: setattr(self, "documents", d), "titles.json": lambda d: setattr(self, "titles", d)
            }
            graph_loaders = {
                "knowledge_graph.json": lambda d: setattr(self, "kg", nx.node_link_graph(d)),
                "entity_contexts.json": lambda d: self.entity_contexts.update(d),
                "entity_cooccurrence.json": lambda d: self.entity_cooccurrence.update({tuple(k.split("||")): v for k, v in d.items()}),
            }
            for fname, loader in graph_loaders.items():
                p = f"{self.graph_db_path}/{fname}"
                if not os.path.exists(p):
                    legacy_p = f"{self.legacy_kg_db_path}/{fname}"
                    p = legacy_p if os.path.exists(legacy_p) else p
                if os.path.exists(p):
                    with open(p) as f: loader(json.load(f))
            for fname, loader in loaders.items():
                p = f"{self.kg_db_path}/{fname}"
                if not os.path.exists(p):
                    legacy_p = f"{self.legacy_kg_db_path}/{fname}"
                    p = legacy_p if os.path.exists(legacy_p) else p
                if os.path.exists(p):
                    with open(p) as f: loader(json.load(f))
            for attr, fname in [("document_embeddings", "document_embeddings.npy"), ("communities", "communities.npy")]:
                p = f"{self.kg_db_path}/{fname}"
                if not os.path.exists(p):
                    legacy_p = f"{self.legacy_kg_db_path}/{fname}"
                    p = legacy_p if os.path.exists(legacy_p) else p
                if os.path.exists(p):
                    val = np.load(p)
                    setattr(self, attr, val.tolist() if attr == "communities" else val)
            p = f"{self.kg_db_path}/faiss_index.index"
            if not os.path.exists(p):
                legacy_p = f"{self.legacy_kg_db_path}/faiss_index.index"
                p = legacy_p if os.path.exists(legacy_p) else p
            if os.path.exists(p): self.neighbor_index = faiss.read_index(p)
            if len(self.document_metadata) < len(self.documents):
                self.document_metadata.extend({} for _ in range(len(self.documents) - len(self.document_metadata)))
            if self.documents: self.bm25 = BM25Okapi([d.lower().split() for d in self.documents])
            if self.document_embeddings is not None and self.communities: self._rebuild_community_index()
            if self.documents: print(f"   Loaded {len(self.documents)} docs, {self.kg.number_of_nodes()} KG entities")
            else: print("No existing database Ã¢â‚¬â€ fresh start")
        except Exception as e: print(f"Load error: {e}")

    def _rebuild_community_index(self):
        if not self.communities or self.document_embeddings is None: return
        dim, centroids = self.document_embeddings.shape[1], []
        for cid in set(self.communities):
            mask = np.array(self.communities) == cid
            if np.any(mask): centroids.append(np.mean(self.document_embeddings[mask], axis=0))
        if centroids:
            self.community_index = faiss.IndexFlatIP(dim)
            arr = np.array(centroids).astype(np.float32)
            faiss.normalize_L2(arr)
            self.community_index.add(arr)

    def ingest_financial_documents(self, documents_list: list):
        print(f"\nÃ°Å¸â€œÂ¥ Ingesting {len(documents_list)} documents...")
        for i, doc in enumerate(documents_list):
            title, content = doc.get("title", f"Document_{i}"), doc.get("content", "")
            redacted, pii_found = self.pii_redactor.redact(content)
            if pii_found: print(f"{title}: {len(pii_found)} PII redacted")
            metadata = {
                "source_type": doc.get("source_type", "unknown"),
                "category": doc.get("category", ""),
                "doc_id": doc.get("doc_id", title),
                "source_path": doc.get("source_path", ""),
                "section_path": doc.get("section_path", []),
                "page_start": doc.get("page_start"),
                "page_end": doc.get("page_end"),
                "node_id": doc.get("node_id"),
            }
            self.documents.append(redacted); self.titles.append(title); self.document_metadata.append(metadata)
        print(f"{len(self.documents)} documents ingested")
        self._build_hybrid_indices()

    def _build_hybrid_indices(self):
        print("Building hybrid indices...")
        texts = [f"{t}: {d[:600]}" for t, d in zip(self.titles, self.documents)]
        self.document_embeddings = self.embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        dim = self.document_embeddings.shape[1]
        self.neighbor_index = faiss.IndexFlatIP(dim)
        emb = self.document_embeddings.astype(np.float32).copy()
        faiss.normalize_L2(emb)
        self.neighbor_index.add(emb)
        n_docs, n_clusters = len(self.documents), max(2, min(50, len(self.documents) // 3))
        print(f"   Building {n_clusters} communities from {n_docs} docs...")
        self.build_lightrag_communities(n_clusters=n_clusters)
        self.generate_community_summaries()
        self.bm25 = BM25Okapi([d.lower().split() for d in self.documents])
        print(f"Indices ready | {self.neighbor_index.ntotal} vectors | {n_clusters} communities | BM25 ready")
        self.save_kg_database()

    def build_lightrag_communities(self, n_clusters: int = 5):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.communities = kmeans.fit_predict(self.document_embeddings)
        centroids = [np.mean(self.document_embeddings[self.communities == i], axis=0) for i in range(n_clusters) if np.sum(self.communities == i) > 0]
        dim = self.document_embeddings.shape[1]
        self.community_index = faiss.IndexFlatIP(dim)
        arr = np.array(centroids).astype(np.float32)
        faiss.normalize_L2(arr)
        self.community_index.add(arr)

    def generate_community_summaries(self):
        print("Generating community summaries...")
        for community_id in set(self.communities):
            idxs = [i for i in range(len(self.documents)) if self.communities[i] == community_id]
            if not idxs: continue
            embs, cent = self.document_embeddings[idxs], np.mean(self.document_embeddings[idxs], axis=0)
            top = np.argsort(np.dot(embs, cent))[-2:][::-1]
            texts = [self.documents[idxs[i]][:400] for i in top]
            user_msg = f"Summarize the main financial themes in 2 sentences:\n\n{chr(10).join(texts)}"
            if self.llm_client:
                summary = self.llm_client.complete("You are a concise financial summariser.", user_msg, max_tokens=100, temperature=0.3)
            else:
                prompt = self._build_prompt("You are a concise financial summariser.", user_msg)
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
                with torch.no_grad():
                    out = self.generator.generate(**inputs, max_new_tokens=80, temperature=0.3, do_sample=True, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id)
                summary = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            self.community_summaries[int(community_id)] = summary
        print(f"{len(self.community_summaries)} community summaries generated")

    def build_raptor_root(self):
        if len(self.community_summaries) < 2: print("Not enough communities for RAPTOR root"); return
        print("Building RAPTOR root (Level 2)...")
        all_summaries = "\n\n ".join(f"Topic {k}: {v}" for k, v in self.community_summaries.items())
        user_msg = f"Create one 3-sentence overview across ALL these financial topics:\n\n{all_summaries}"
        if self.llm_client:
            self.raptor_root = self.llm_client.complete("You are a concise financial summariser.", user_msg, max_tokens=150, temperature=0.3)
        else:
            prompt = self._build_prompt("You are a concise financial summariser.", user_msg)
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
            with torch.no_grad():
                out = self.generator.generate(**inputs, max_new_tokens=120, temperature=0.3, do_sample=True, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id)
            self.raptor_root = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"RAPTOR root: {self.raptor_root[:120]}...")

    def build_knowledge_graph_from_documents(self):
        print("Building knowledge graph...")
        for i, doc in enumerate(self.documents):
            doc_entities = self.entity_extractor.extract(doc[:1000])
            for entity, entity_type in doc_entities:
                self.entity_contexts[entity].append({"title": self.titles[i], "snippet": doc[:200], "type": entity_type, "timestamp": datetime.now().isoformat()})
            elist = list(doc_entities)
            for j, (e1, _) in enumerate(elist[:10]):
                for (e2, _) in elist[j+1:10]:
                    if e1 != e2:
                        if self.kg.has_edge(e1, e2): self.kg[e1][e2]["weight"] += 1
                        else: self.kg.add_edge(e1, e2, weight=1)
                        self.entity_cooccurrence[(e1, e2)] += 1
        print(f"KG: {self.kg.number_of_nodes()} entities | {self.kg.number_of_edges()} relationships")
        self.save_kg_database()

    def find_kg_paths(self, query: str, max_paths: int = 5, max_depth: int = 2) -> list:
        query_entities = [e for e, _ in self.entity_extractor.extract(query)]
        all_paths = []
        for entity in query_entities:
            if entity in self.kg:
                for target in self.kg.nodes():
                    if entity != target:
                        try:
                            path = nx.shortest_path(self.kg, source=entity, target=target)
                            if 2 <= len(path) <= max_depth + 1: all_paths.append(path)
                        except Exception: continue
        scored = sorted([(p, sum(self.kg[p[i]][p[i+1]].get("weight", 1) for i in range(len(p)-1))) for p in all_paths], key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:max_paths]]

    def kg_local_search(self, query: str) -> list:
        query_entities = [e for e, _ in self.entity_extractor.extract(query)]
        seen, results = set(), []
        for entity in query_entities:
            for ctx in self.entity_contexts.get(entity, [])[:2]:
                if ctx["title"] not in seen: seen.add(ctx["title"]); results.append({"content": ctx["snippet"], "title": ctx["title"], "source": f"KG:{entity}", "community": 0})
            if entity in self.kg:
                neighbors = sorted(self.kg.neighbors(entity), key=lambda n: self.kg[entity][n].get("weight", 1), reverse=True)[:3]
                for neighbor in neighbors:
                    for ctx in self.entity_contexts.get(neighbor, [])[:1]:
                        if ctx["title"] not in seen: seen.add(ctx["title"]); results.append({"content": ctx["snippet"], "title": ctx["title"], "source": f"KG:{entity}->{neighbor}", "community": 0})
        return results[:5]

    def get_cooccurrence_context(self, query: str) -> str:
        query_entities = {e for e, _ in self.entity_extractor.extract(query)}
        relevant = [(e1, e2, w) for (e1, e2), w in self.entity_cooccurrence.items() if e1 in query_entities or e2 in query_entities]
        if not relevant: return ""
        relevant.sort(key=lambda x: x[2], reverse=True)
        lines = [f"  {e1} <-> {e2} (strength: {w})" for e1, e2, w in relevant[:4]]
        return "[Concept relationships]\n" + "\n".join(lines)

    def _vector_search(self, query: str, k: int) -> tuple:
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        _, indices = self.neighbor_index.search(q_emb, k * 2)
        community_indices = []
        if self.community_index is not None:
            _, community_ids = self.community_index.search(q_emb, 5)
            for cid in community_ids[0]: community_indices.extend([i for i in range(len(self.documents)) if self.communities[i] == cid][:2])
        unique_indices = list(dict.fromkeys(list(indices[0]) + community_indices))
        relevant_comms = {self.communities[i] for i in unique_indices[:k]}
        community_context = [f"Topic: {self.community_summaries[cid]}" for cid in relevant_comms if cid in self.community_summaries]
        return unique_indices[:k], community_context

    def _bm25_search(self, query: str, k: int) -> list:
        if not self.bm25: return []
        scores = self.bm25.get_scores(query.lower().split())
        return [i for i in np.argsort(scores)[::-1][:k] if scores[i] > 0]

    def _rrf_fusion(self, vector_idx: list, bm25_idx: list, k: int = 60) -> list:
        scores = {}
        for rank, idx in enumerate(vector_idx): scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        for rank, idx in enumerate(bm25_idx): scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    def _rerank(self, query: str, doc_indices: list, top_n: int = 4) -> list:
        if not self.rerank_enabled or not doc_indices: return doc_indices[:top_n]
        try:
            from flashrank import RerankRequest
            passages = [{"id": i, "text": self.documents[idx]} for i, idx in enumerate(doc_indices) if idx < len(self.documents)]
            results = self.reranker.rerank(RerankRequest(query=query, passages=passages))
            return [doc_indices[r["id"]] for r in results[:top_n]]
        except Exception as e: print(f"Re-rank failed ({e})"); return doc_indices[:top_n]

    def retrieve(self, query: str, k: int = 8) -> tuple:
        clean_query, pii_found = self.pii_redactor.redact(query)
        if pii_found: print("PII detected and redacted")
        vector_idx, community_context = self._vector_search(clean_query, k)
        bm25_idx = self._bm25_search(clean_query, k)
        fused_idx = self._rrf_fusion(vector_idx, bm25_idx)
        final_idx = self._rerank(clean_query, fused_idx, top_n=3)
        vector_docs = [
            {
                "content": self.documents[idx],
                "title": self.titles[idx],
                "metadata": self.document_metadata[idx] if idx < len(self.document_metadata) else {},
                "community": int(self.communities[idx]) if len(self.communities) > 0 else 0,
                "source": "vector+bm25",
            }
            for idx in final_idx if idx < len(self.documents)
        ]
        kg_docs = self.kg_local_search(clean_query)
        seen_titles = {d["title"] for d in vector_docs}
        for doc in kg_docs:
            if doc["title"] not in seen_titles: vector_docs.append(doc); seen_titles.add(doc["title"])
        retrieved = vector_docs[:4]
        for doc in retrieved:
            src = doc.get("source", "vector+bm25")
            ind = "[KG]" if src.startswith("KG") else "[VEC]"
            ttl = doc["title"][:50] + "..." if len(doc["title"]) > 50 else doc["title"]
            meta = doc.get("metadata") or {}
            meta_bits = [str(meta.get("source_type") or "").strip(), str(meta.get("category") or "").strip()]
            if meta.get("page_start") or meta.get("page_end"):
                meta_bits.append(f"p.{meta.get('page_start') or '?'}-{meta.get('page_end') or meta.get('page_start') or '?'}")
            meta_text = " | ".join(bit for bit in meta_bits if bit)
            print(f"      {ind} [{src}] {ttl}" + (f" ({meta_text})" if meta_text else ""))
        return retrieved, community_context, self.find_kg_paths(clean_query)

    def generate(self, query: str, docs: list, community_context: list, reasoning_paths: list) -> str:
        if not docs: return "I don't have enough information to answer accurately. Consult a SEBI-registered financial advisor."
        context_parts = []
        for i, doc in enumerate(docs[:4], 1):
            context_parts.append(format_retrieved_doc_for_prompt(doc, i, max_chars=400))
        context_str = "\n\n".join(context_parts)
        memory_ctx = ""
        if self.memory.history:
            last = self.memory.history[-1]
            memory_ctx = f"Previous: Q: {last['question'][:60]} A: {last['answer'][:100]}..."
        is_complex = _is_complex_query(query)
        if is_complex:
            system_msg = (
                "You are FinSage, a financial assistant for Indian users. "
                "Answer ALL parts of the user's question using ONLY the documents provided. "
                "Use [LIVE DATA] numbers exactly. Use brief headers for multiple sub-questions. "
                "Cover each sub-question in 2-3 sentences. "
                "End with: Consult a SEBI-registered advisor before investing."
            )
        else:
            system_msg = (
                "You are FinSage, a financial assistant for Indian users. "
                "Answer using ONLY the documents provided. Be direct and concise. 2-4 sentences maximum. "
                "If the document has [LIVE DATA], use those exact numbers. "
                "End with: Consult a SEBI-registered advisor."
            )
        user_msg = f"{memory_ctx}\n\nDocuments:\n{context_str}\n\nQuestion: {query}"
        print("\nÃ°Å¸â€™Â¬ FinSage: ", end="", flush=True)
        if self.llm_client:
            raw = self.llm_client.complete(system_msg, user_msg, max_tokens=512, temperature=0.1)
            print(raw)
        else:
            prompt = self._build_prompt(system_msg, user_msg)
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(self.device)
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            with torch.no_grad():
                outputs = self.generator.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.1, top_p=0.9, repetition_penalty=1.5, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id, streamer=streamer)
            raw = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        for marker in ["Your response", "I asked you", "Why did you", "Make sure", "It didn't", "You didn't", "Improve your", "Note:", "Note that", "[SECURITY", "[END", "Remember always", "However,", "Unfortunately"]:
            idx = raw.find(marker)
            if idx > 60: raw = raw[:idx].strip()
        return raw if len(raw) >= 20 else "Insufficient data in documents for this query. Consult a SEBI-registered advisor for personalized advice."

    def _snapshot_inject(self, live_docs: list):
        self._snap_docs = self.documents[:]
        self._snap_titles = self.titles[:]
        self._snap_metadata = self.document_metadata[:]
        self._snap_embs = self.document_embeddings.copy() if self.document_embeddings is not None else None
        self._snap_comms = list(self.communities) if len(self.communities) > 0 else []
        for doc in live_docs:
            redacted, _ = self.pii_redactor.redact(doc["content"])
            self.documents.append(redacted); self.titles.append(doc["title"]); self.document_metadata.append(doc.get("metadata", {"source_type": "mcp_tool"}))
        new_embs = self.embedder.encode([f"{d['title']}: {d['content'][:600]}" for d in live_docs], normalize_embeddings=True).astype(np.float32)
        faiss.normalize_L2(new_embs)
        self.neighbor_index.add(new_embs)
        self.bm25 = BM25Okapi([d.lower().split() for d in self.documents])
        if len(self.communities) > 0: self.communities = np.append(self.communities, [0] * len(live_docs))

    def _snapshot_restore(self):
        self.documents, self.titles = self._snap_docs, self._snap_titles
        self.document_metadata = self._snap_metadata
        if self._snap_embs is not None:
            dim = self._snap_embs.shape[1]
            self.neighbor_index = faiss.IndexFlatIP(dim)
            emb = self._snap_embs.astype(np.float32).copy()
            faiss.normalize_L2(emb)
            self.neighbor_index.add(emb)
            self.document_embeddings = self._snap_embs
        self.bm25 = BM25Okapi([d.lower().split() for d in self.documents])
        if self._snap_comms: self.communities = np.array(self._snap_comms)

    async def query_agentic(self, question: str, k: int = 8) -> dict:
        t0 = time.time()
        print(f"\n{'='*60}\nâ€œ {question}\n{'='*60}")
        classification = self.classifier.classify(question)
        if classification["guardrail"] == "BLOCK":
            print(f"BLOCKED [{classification['reason']}]")
            return {"question": question, "answer": self.classifier.GUARDRAIL_FALLBACK, "blocked": True, "block_category": classification["reason"], "sources": [], "time": round(time.time()-t0, 2)}
        is_market = classification["intent"] == "market"
        print(f"   Intent: {'MARKET' if is_market else 'PERSONAL_FINANCE'}")
        live_docs, tool_calls = [], []
        if is_market:
            print("hi-3 selecting tools...")
            tool_calls = self.tool_selector.select_tools(question)
            if tool_calls:
                print(f"   Executing {len(tool_calls)} tool call(s)...")
                live_docs = await self.mcp_client.execute(tool_calls)
                if live_docs:
                    print(f"   Injecting {len(live_docs)} live docs into FAISS...")
                    self._snapshot_inject(live_docs)
            else: print(" No tools selected Ã¢â‚¬â€ plain RAG fallback")
        docs, community_context, reasoning_paths = self.retrieve(question, k=k)
        if live_docs: self._snapshot_restore(); print("  FAISS snapshot restored")
        if live_docs:
            seen_titles = {d["title"] for d in docs}
            docs = [d for d in live_docs if d["title"] not in seen_titles] + docs
        mcp_count = sum(1 for d in docs if d.get("source") == "mcp_tool")
        kg_count = sum(1 for d in docs if d.get("source", "").startswith("KG"))
        vector_count = len(docs) - mcp_count - kg_count
        print(f"\n   Ã°Å¸â€œÅ¡ {len(docs)} docs total ({vector_count} vector+BM25, {kg_count} KG, {mcp_count} live)")
        answer = self.generate(question, docs[:6], community_context, reasoning_paths)
        out_safe, out_cat = self.output_guard.check(answer)
        if not out_safe: print(f"\n Output flagged [{out_cat}]"); answer += self.output_guard.DISCLAIMER
        self.memory.add(question, answer)
        elapsed = round(time.time()-t0, 2)
        mode = "mcp_market" if is_market else "rag_personal_finance"
        result = {"question": question, "answer": answer, "blocked": False, "mode": mode, "is_market": is_market, "classifier": classification,
                  "tool_calls": [tc.get("tool") for tc in tool_calls], "sources": [d["title"] for d in docs[:6]],
                  "retrieved_docs": [d["content"][:400] for d in docs[:4]], "kg_stats": {"entities": self.kg.number_of_nodes(), "relationships": self.kg.number_of_edges()}, "time": elapsed}
        self.query_history.append({"question": question, "answer": answer, "time": elapsed, "retrieved_docs": [d["content"][:400] for d in docs[:4]]})
        print(f"\n {elapsed}s | mode={mode}")
        return result

    def evaluate_ragas(self, ground_truths: list) -> dict:
        try:
            from datasets import Dataset as HFDataset
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        except ImportError as e: print(f"RAGAS not available: {e}"); return {}
        n, recent = len(ground_truths), self.query_history[-len(ground_truths):]
        eval_ds = HFDataset.from_dict({"question": [r["question"] for r in recent], "answer": [r["answer"] for r in recent],
                                       "contexts": [r.get("retrieved_docs", [r["answer"][:500]]) for r in recent], "ground_truth": ground_truths})
        print("\nÃ°Å¸â€œÅ  Running RAGAS evaluation...")
        try:
            scores = evaluate(eval_ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
            print(f"\n   Faithfulness      : {scores['faithfulness']:.3f} (> 0.85)")
            print(f"   Answer Relevancy  : {scores['answer_relevancy']:.3f} (> 0.80)")
            print(f"   Context Precision : {scores['context_precision']:.3f} (> 0.75)")
            print(f"   Context Recall    : {scores['context_recall']:.3f} (> 0.70)")
            return scores
        except Exception as e: print(f" RAGAS needs LLM judge: {e}"); return {}

    def show_kg_stats(self):
        print("\nÃ°Å¸â€œÅ  KNOWLEDGE GRAPH STATS\n" + "= "*40)
        print(f"   Entities       : {self.kg.number_of_nodes()}")
        print(f"   Relationships  : {self.kg.number_of_edges()}")
        print(f"   Documents      : {len(self.documents)}")
        n_comm = len(set(self.communities.tolist())) if len(self.communities) > 0 else 0
        print(f"   Communities    : {n_comm}")
        print(f"   Queries logged : {len(self.query_history)}")
        if self.kg.number_of_nodes() > 0:
            print("\nTop entities:")
            for entity, score in sorted(nx.degree_centrality(self.kg).items(), key=lambda x: x[1], reverse=True)[:5]: print(f"  {entity}: {score:.4f}")




# =================================================================
# CHANGE PATCH V4 - router, registry, local planners, grounded generation
# =================================================================
class EnhancedConversationMemory:
    def __init__(self, max_turns=5):
        self.history=[]; self.summary=''; self.max_turns=max_turns; self.user_profile={}
    def _update_profile(self, q, meta=None):
        ql=q.lower(); age=re.search(r"(?:age\s*[:=]?\s*|i am\s+|i'm\s+)(\d{2})", ql)
        income=re.search(r"(?:income|salary)\D{0,12}(\d[\d,]*)", ql); exp=re.search(r"expenses?\D{0,12}(\d[\d,]*)", ql)
        yrs=re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", ql)
        if age: self.user_profile['age']=age.group(1)
        if income: self.user_profile['income']=income.group(1)
        if exp: self.user_profile['expenses']=exp.group(1)
        if yrs and any(k in ql for k in ['retirement','goal','sip','invest']): self.user_profile['horizon_years']=yrs.group(1)
        for risk in ['conservative','moderate','aggressive']:
            if risk in ql: self.user_profile['risk_profile']=risk
        if meta and meta.get('tool_calls'): self.user_profile['recent_tools']=', '.join(meta['tool_calls'][:3])
    def add(self, question, answer, meta=None):
        self._update_profile(question, meta); self.history.append({'question':question,'answer':answer,'meta':meta or {}})
        if len(self.history)>self.max_turns:
            old=self.history.pop(0); self.summary += f"Discussed: {old['question'][:80]}. Key: {old['answer'][:120]}. "
            if len(self.summary)>700: self.summary=self.summary[-700:]
    def get_context(self):
        parts=[]
        if self.user_profile: parts.append('[Known user profile]\n' + ', '.join(f"{k}={v}" for k,v in self.user_profile.items()))
        if self.summary: parts.append('[Earlier summary]\n' + self.summary)
        if self.history:
            parts.append('[Recent exchanges]')
            for ex in self.history[-3:]: parts += [f"User: {ex['question']}", f"FinSage: {ex['answer'][:220]}..."]
        return '\n'.join(parts)
    def clear(self): self.history=[]; self.summary=''; self.user_profile={}

class ToolRegistry:
    def __init__(self):
        self.specs={'screener':{'route':'market'},'amfi_nav':{'route':'market'},'search_rag':{'route':'personal_finance'},'sip_calculator':{'route':'planner'},'emi_calculator':{'route':'planner'},'portfolio_health':{'route':'planner'},'goal_planner':{'route':'planner'}}

class DeterministicQueryRouter:
    BLOCKS=[(r'hide income|conceal income|tax evasion','TAX_EVASION'),(r'guaranteed\s+\d+%\s+(?:returns?|profit)','SCAM_RETURNS'),(r'insider tip|non public information|inside information','INSIDER_TRADING'),(r'pump and dump|manipulate stock','MARKET_MANIPULATION'),(r'ponzi|pyramid scheme','PONZI')]
    EXCLUDED_TOKENS={
        'PE','NAV','SIP','EMI','NSE','BSE','ETF','IPO','ROI','CAGR','GDP','RBI','SEBI','PPF','NPS',
        'ELSS','ULIP','FD','RD','PF','EPF','ESOP','EPS','PNL','EBIT','EBITDA','PAT','NII','NIM',
        # LLM output / chat-history tokens that match the ticker regex
        'JSON','USER','ASSISTANT','SYSTEM','HUMAN','AI','BOT','LLM','GPT','CHAT','THINK','FLOW',
        'OK','TASK','KG','MCP','RAG','API','HTTP','URL','GET','POST','PUT','DB','SQL','ID',
        'OKLO','OKTA','OKE','OWWAF','TWO','MX','HK','US','UK','EU','CN','JP','KR','DU','DE',
        'IN','IS','OF','AT','BY','IF','OR','AND','THE','FOR','ARE','WAS','HAS','NOT','NEW',
    }
    # Loaded from data/financial_kg/nse_ticker_dict.json at first use
    _NSE_DICT_CACHE: dict = {}
    _NSE_DICT_LOADED: bool = False

    @classmethod
    def _load_nse_dict(cls) -> dict:
        if cls._NSE_DICT_LOADED:
            return cls._NSE_DICT_CACHE
        try:
            _path = Path(os.environ.get('BANYANTREE_FINANCIAL_KG_ROOT', '/app/data/financial_kg')) / 'nse_ticker_dict.json'
            with open(_path, encoding='utf-8') as f:
                data = json.load(f)
            cls._NSE_DICT_CACHE = {k.lower(): v for k, v in data.get('tickers', {}).items()}
            print(f"NSE ticker dict loaded: {len(cls._NSE_DICT_CACHE)} entries from {_path}")
        except Exception as e:
            print(f"NSE ticker dict load failed ({e}), using empty dict")
            cls._NSE_DICT_CACHE = {}
        cls._NSE_DICT_LOADED = True
        return cls._NSE_DICT_CACHE

    # kept for backward compat — routes through the JSON file now
    KNOWN_NSE_ALIASES: dict = {}

    def __init__(self, registry, tokenizer=None, generator=None, device=None, llm_client=None):
        self.registry=registry; self.tokenizer=tokenizer; self.generator=generator; self.device=device; self._symbol_cache={}; self.llm_client=llm_client
        self.KNOWN_NSE_ALIASES = self._load_nse_dict()
    def _has(self, pat, txt): return bool(re.search(pat, txt, re.I))
    def _candidate_terms(self, candidate):
        """Normalize LLM/yfinance outputs into candidate ticker/name strings."""
        def walk(value):
            out=[]
            if isinstance(value, dict):
                for key in ('symbol','ticker','nse_symbol','current_symbol','company_name','name','shortname','longname'):
                    if value.get(key): out.append(str(value[key]).strip())
            elif isinstance(value, list):
                for item in value: out.extend(walk(item))
            elif value is not None:
                text=str(value).strip()
                if not text: return []
                parsed=None
                try:
                    parsed=json.loads(text)
                except Exception:
                    try:
                        import ast as _ast
                        parsed=_ast.literal_eval(text)
                    except Exception:
                        parsed=None
                if parsed is not None and parsed is not value:
                    out.extend(walk(parsed))
                else:
                    for key in ('symbol','ticker','nse_symbol','current_symbol','company_name','name'):
                        for m in re.finditer(rf"['\"]{key}['\"]\s*:\s*['\"]([^'\"]+)['\"]", text, re.I):
                            out.append(m.group(1).strip())
                    out.append(text)
            return out
        terms=[]
        for term in walk(candidate):
            term=str(term or '').strip()
            if term and term not in terms:
                terms.append(term)
        return terms

    def _llm_symbol_candidates(self, query, failed_candidates=None):
        if not (self.tokenizer and self.generator and self.device): return []
        failed_candidates = failed_candidates or []
        system_msg = (
            "You resolve Indian listed-company names to their CURRENT NSE trading symbols. "
            "Return only valid NSE equity ticker candidates that should have market data in Yahoo/yfinance. "
            "If an old ticker is rejected as delisted, infer the renamed/current listed entity only if you know it. "
            "Do not invent tickers. If unsure, return []. Output ONLY a JSON array like [\"INFY\"]. No markdown. No explanation."
        )
        user_msg = (
            f"Query: {query}\n"
            f"Rejected/failed candidates from market-data validation: {failed_candidates[:8]}\n"
            "Return current NSE ticker candidates as JSON array:"
        )
        # In API mode tokenizer/generator are None — use the API LLM directly.
        if self.tokenizer is None or self.generator is None:
            if not self.llm_client:
                return []
            try:
                raw = self.llm_client.complete(system_msg, user_msg, max_tokens=100, temperature=0.0)
                print(f"Ticker API-LLM fallback raw: {raw}")
            except Exception as e:
                print(f"Ticker API-LLM extraction failed: {e}")
                return []
        else:
            prompt = build_qwen_prompt(self.tokenizer, system_msg, user_msg)
            try:
                inputs=self.tokenizer(prompt, return_tensors='pt', max_length=1100, truncation=True).to(self.device)
                with torch.no_grad():
                    outputs=self.generator.generate(**inputs, max_new_tokens=100, temperature=0.0, do_sample=False, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id)
                raw=self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                print(f"Ticker LLM fallback raw: {raw}")
            except Exception as e:
                print(f"Ticker LLM extraction failed: {e}")
                return []
        values=[]
        raw_clean=re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.I)
        match=re.search(r'\[[\s\S]*?\]', raw_clean)
        if match:
            try:
                loaded=json.loads(match.group())
            except Exception:
                import ast as _ast
                loaded=_ast.literal_eval(match.group())
            values.extend(self._candidate_terms(loaded))
        else:
            values.extend([tok for tok in re.findall(r'\b[A-Z][A-Z0-9&-]{1,11}\b', raw_clean) if tok not in self.EXCLUDED_TOKENS])
        out=[]
        for value in values:
            value=str(value or '').strip()
            if value and value not in out:
                out.append(value)
        return out

    def _yfinance_search_candidates(self, text):
        # Yahoo Finance search (yf.Search and free endpoint) both require an auth
        # crumb — return 401 without it. Use NSE ticker dict + LLM fallback instead.
        return []

    def _resolve_company_via_llm(self, company_name: str) -> str | None:
        """Ask the API LLM for the NSE ticker of an unknown company name."""
        if not self.llm_client:
            return None
        cached = self._symbol_cache.get(f'llm:{company_name.lower()}')
        if cached is not None:
            return cached or None
        try:
            system_msg = "You are an NSE stock ticker resolver. Return ONLY the NSE ticker symbol (e.g. INFY, TCS, M&M). No explanation, no markdown."
            user_msg = f"What is the NSE ticker symbol for: {company_name}"
            raw = self.llm_client.complete(system_msg, user_msg, max_tokens=20, temperature=0.0).strip()
            raw = re.sub(r'[^A-Z0-9&\-]', '', raw.upper())
            result = raw if re.fullmatch(r'[A-Z][A-Z0-9&\-]{1,11}', raw) else None
            print(f"LLM ticker lookup '{company_name}' -> {result}")
            self._symbol_cache[f'llm:{company_name.lower()}'] = result or ''
            return result
        except Exception as e:
            print(f"LLM ticker lookup failed for '{company_name}': {e}")
            self._symbol_cache[f'llm:{company_name.lower()}'] = ''
            return None

    def _company_name_candidates(self, query):
        stop={
            'what','is','are','the','a','an','today','current','share','price','stock',
            'market','pe','ratio','nav','nse','bse','compare','and','vs','versus','for',
            'of','in','india','tell','me','show','give','latest','fundamentals','quarterly',
            'results','good','bad','invest','investment','buy','sell','should','would',
            'could','when','where','how','why','who','which','them','any','some','prices',
            'stcok','stocks','shares','today','now','currently','about','with','into',
        }
        cleaned=re.sub(r'[^A-Za-z0-9&.\-\s]', ' ', str(query or ''))
        parts=[]
        for tok in cleaned.split():
            low=tok.lower().strip('.-')
            if len(low) >= 3 and low not in stop and not low.isdigit():
                parts.append(tok.strip())
        candidates=[]
        for n in range(1, min(3, len(parts)) + 1):
            for i in range(0, len(parts)-n+1):
                phrase=' '.join(parts[i:i+n]).strip()
                if phrase and phrase not in candidates:
                    candidates.append(phrase)
        return candidates

    def _validate_yfinance_symbol(self, candidate):
        terms=self._candidate_terms(candidate)
        if not terms: return None
        key='|'.join(t.lower() for t in terms)
        if key in self._symbol_cache: return self._symbol_cache[key]
        try:
            import yfinance as yf
            trial=[]
            for term in terms:
                trial += self._yfinance_search_candidates(term)
                raw=re.sub(r'[^A-Za-z0-9&.-]', '', term).upper()
                if re.fullmatch(r'[A-Z0-9&.-]{2,15}', raw):
                    trial += [raw.replace('.NS','').replace('.BO','')]
            seen=set()
            for sym in trial:
                sym=str(sym or '').replace('.NS','').replace('.BO','').upper()
                if not sym or sym in self.EXCLUDED_TOKENS or sym in seen: continue
                seen.add(sym)
                ticker=yf.Ticker(f'{sym}.NS')
                valid=False
                try:
                    hist=ticker.history(period='5d', interval='1d', timeout=5)
                    valid=not hist.empty
                except Exception:
                    valid=False
                if valid:
                    self._symbol_cache[key]=sym
                    return sym
        except Exception as e:
            print(f"Ticker yfinance validation failed for {candidate}: {e}")
        self._symbol_cache[key]=None
        return None

    def _resolve_candidate_list(self, candidates, limit=4):
        out=[]; failed=[]
        for cand in candidates:
            sym=self._validate_yfinance_symbol(cand)
            if sym and sym not in out:
                out.append(sym)
            else:
                failed.append(cand)
            if len(out) >= limit: break
        return out, failed

    def _symbols(self, query):
        out=[]
        ql=str(query or '').lower()
        # Step 1: NSE ticker dict lookup (file-based, instant)
        for alias, sym in self.KNOWN_NSE_ALIASES.items():
            if re.search(rf'\b{re.escape(alias)}\b', ql) and sym not in out:
                out.append(sym)
        # Step 2: Uppercase ticker tokens from query (e.g. user typed "INFY" directly)
        for tok in re.findall(r'\b[A-Z][A-Z0-9&-]{1,11}\b', query):
            if tok not in self.EXCLUDED_TOKENS and tok not in out:
                out.append(tok)
        # Step 3: Company name phrase extraction — only single words, no bigrams
        # (bigrams cause concatenation garbage like WIPROMAHINDRA)
        name_parts = self._company_name_candidates(query)
        single_word_names = [p for p in name_parts if ' ' not in p]
        # Step 4: For each unresolved company name not in alias dict, call LLM
        for name in single_word_names:
            if len(out) >= 4: break
            sym = self._resolve_company_via_llm(name)
            if sym and sym not in out:
                out.append(sym)
        print(f"Ticker resolver resolved={out}")
        return out
    def classify(self, query):
        ql=query.lower()
        for pat,reason in self.BLOCKS:
            if self._has(pat, ql): return {'guardrail':'BLOCK','reason':reason,'intent':'personal_finance','tool_calls':[],'confidence':1.0,'route_source':'rules'}
        _market_kw=['share price','stock','market cap','pe ratio','p/e','quarterly results','52 week','52w','nifty','sensex','fundamentals','invest in','buy shares','should i buy','price today','current price']
        _alias_hit = any(alias in ql for alias in self.KNOWN_NSE_ALIASES)
        if any(k in ql for k in _market_kw) or _alias_hit:
            syms=self._symbols(query); calls=[{'tool':'screener','symbol':s} for s in syms] if syms else []
            return {'guardrail':'OK','reason':'','intent':'market','tool_calls':calls,'confidence':0.95 if calls else 0.7,'route_source':'yfinance_llm' if calls else 'qwen'}
        if any(k in ql for k in ['nav','mutual fund','elss']): return {'guardrail':'OK','reason':'','intent':'market','tool_calls':[{'tool':'amfi_nav','fund_filter':'ELSS' if 'elss' in ql else query[:40]}],'confidence':0.92,'route_source':'rules'}
        if 'sip' in ql and self._has(r'\d+(?:\.\d+)?\s*%', query) and self._has(r'\d+(?:\.\d+)?\s*(?:years?|yrs?)', query): return {'guardrail':'OK','reason':'','intent':'personal_finance','tool_calls':[{'tool':'sip_calculator','query':query}],'confidence':0.94,'route_source':'rules'}
        if ('emi' in ql or 'loan' in ql) and self._has(r'\d+(?:\.\d+)?\s*%', query) and self._has(r'\d+(?:\.\d+)?\s*(?:years?|yrs?)', query): return {'guardrail':'OK','reason':'','intent':'personal_finance','tool_calls':[{'tool':'emi_calculator','query':query}],'confidence':0.94,'route_source':'rules'}
        if any(k in ql for k in ['portfolio','allocation','rebalance','diversification']): return {'guardrail':'OK','reason':'','intent':'personal_finance','tool_calls':[{'tool':'portfolio_health','query':query}],'confidence':0.92,'route_source':'rules'}
        if any(k in ql for k in ['retirement','goal','corpus','financial freedom']) and self._has(r'\d+(?:\.\d+)?\s*(?:years?|yrs?)', query): return {'guardrail':'OK','reason':'','intent':'personal_finance','tool_calls':[{'tool':'goal_planner','query':query}],'confidence':0.9,'route_source':'rules'}
        if any(k in ql for k in ['tax','80c','80d','80ccd','ppf','nps','budget','insurance','emergency fund']): return {'guardrail':'OK','reason':'','intent':'personal_finance','tool_calls':[{'tool':'search_rag','query':query}],'confidence':0.82,'route_source':'rules'}
        return {'guardrail':'OK','reason':'','intent':'personal_finance','tool_calls':[],'confidence':0.35,'route_source':'qwen'}
class EnhancedMCPToolClient(MCPToolClient):
    def __init__(self, mcp_base='http://localhost:8000'):
        super().__init__(mcp_base=mcp_base); self.rag_engine=None
    def register_rag(self, rag_engine): self.rag_engine=rag_engine
    def _amounts(self, text):
        vals=[]
        for v,u in re.findall(r"(?:rs\.?|inr|â‚¹)?\s*(\d[\d,]*(?:\.\d+)?)\s*(crore|cr|lakh|lac|k|thousand)?", text, re.I):
            x=float(v.replace(',','')); u=(u or '').lower(); x*=10000000 if u in {'crore','cr'} else 100000 if u in {'lakh','lac'} else 1000 if u in {'k','thousand'} else 1
            if x>=500: vals.append(x)
        return vals
    def _pct(self, text, d=12.0): m=re.search(r'(\d+(?:\.\d+)?)\s*%', text); return float(m.group(1)) if m else d
    def _years(self, text, d=10.0): m=re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', text, re.I); return float(m.group(1)) if m else d
    def _age(self, text, d=30): m=re.search(r"(?:age\s*[:=]?\s*|i am\s+|i'm\s+)(\d{2})", text, re.I); return int(m.group(1)) if m else d
    def _alloc(self, text):
        out={}
        for a in ['equity','debt','gold','cash']:
            for p in [rf"{a}\s*(?:is|at|=|:)?\s*(\d{{1,3}})\s*%", rf"(\d{{1,3}})\s*%\s*{a}"]:
                m=re.search(p, text, re.I)
                if m: out[a]=int(m.group(1)); break
        return out
    async def _local_rag(self, query, top_k=4):
        if not self.rag_engine: return None
        docs,community_ctx,paths=self.rag_engine.retrieve(query, k=top_k)
        parts=[f"{d['title']}: {d['content'][:180]}" for d in docs[:top_k]]
        if community_ctx: parts.append('Topics: ' + ' | '.join(community_ctx[:2]))
        if paths: parts.append('KG paths: ' + ' | '.join(' -> '.join(p) for p in paths[:2]))
        return {'summary':'\n'.join(parts)}
    async def _sip(self, query):
        monthly=self._amounts(query)[0] if self._amounts(query) else 10000; r=self._pct(query)/1200.0; y=self._years(query); n=int(y*12)
        fv=monthly*n if r==0 else monthly*(((1+r)**n-1)/r)*(1+r); inv=monthly*n
        return {'summary':f"SIP plan | Monthly: Rs {monthly:,.0f} | Horizon: {y:.1f} years | Return: {self._pct(query):.2f}% | Invested: Rs {inv:,.0f} | Estimated value: Rs {fv:,.0f}"}
    async def _emi(self, query):
        p=self._amounts(query)[0] if self._amounts(query) else 5000000; r=self._pct(query,8.5)/1200.0; y=self._years(query,20.0); n=int(y*12)
        emi=p/n if r==0 else p*r*((1+r)**n)/(((1+r)**n)-1); total=emi*n
        return {'summary':f"EMI plan | Loan: Rs {p:,.0f} | Rate: {self._pct(query,8.5):.2f}% | Tenure: {y:.1f} years | EMI: Rs {emi:,.0f} | Total interest: Rs {total-p:,.0f}"}
    async def _portfolio(self, query):
        age=self._age(query,30); risk='aggressive' if 'aggressive' in query.lower() else 'conservative' if 'conservative' in query.lower() else 'moderate'; alloc=self._alloc(query) or {'equity':60,'debt':25,'gold':10,'cash':5}
        eq=max(20,min(85,100-age + (10 if risk=='aggressive' else -15 if risk=='conservative' else 0))); debt=max(10,100-eq-10); warn=[]
        if alloc.get('equity',0)>eq+15: warn.append('equity above model range')
        if alloc.get('debt',0)<max(5,debt-10): warn.append('debt cushion looks light')
        return {'summary':f"Portfolio review | Age: {age} | Risk: {risk} | Current: {alloc} | Model mix: equity {eq}%, debt {debt}%, gold 10%, cash {max(5,100-eq-debt-10)}% | {' | '.join(warn) if warn else 'Allocation broadly aligned'}"}
    async def _goal(self, query):
        target=self._amounts(query)[0] if self._amounts(query) else 20000000; r=self._pct(query)/1200.0; y=self._years(query,15.0); n=int(y*12)
        sip=target/n if r==0 else target/((((1+r)**n-1)/r)*(1+r))
        return {'summary':f"Goal plan | Target corpus: Rs {target:,.0f} | Horizon: {y:.1f} years | Return: {self._pct(query):.2f}% | Required monthly SIP: Rs {sip:,.0f}"}
    async def _call_one(self, tc):
        t=tc.get('tool','')
        if t=='search_rag':
            local=await self._local_rag(tc.get('query',''), int(tc.get('top_k',4)))
            if local: return local
        if t=='sip_calculator': return await self._sip(tc.get('query',''))
        if t=='emi_calculator': return await self._emi(tc.get('query',''))
        if t=='portfolio_health': return await self._portfolio(tc.get('query',''))
        if t=='goal_planner': return await self._goal(tc.get('query',''))
        return await super()._call_one(tc)

_ORIG_INIT=FINANCIAL_HIERARCHICAL_LIGHT_RAG.__init__
def _patched_init(self, kg_db_path='finsage_kg_database'):
    _ORIG_INIT(self, kg_db_path); self.memory=EnhancedConversationMemory(max_turns=5); self.tool_registry=ToolRegistry(); self.router=DeterministicQueryRouter(self.tool_registry, self.tokenizer, self.generator, self.device, llm_client=getattr(self,'llm_client',None)); self.mcp_client=EnhancedMCPToolClient(mcp_base=MCP_BASE); self.mcp_client.register_rag(self)

def _patched_generate(self, query, docs, community_context, reasoning_paths):
    if not docs: return "I don't have enough information to answer accurately. Consult a SEBI-registered financial advisor."
    ctx=[]
    for i,d in enumerate(docs[:6],1): ctx.append(format_retrieved_doc_for_prompt(d, i, max_chars=420))
    mem=self.memory.get_context(); comm='\n'.join(community_context[:3]) if community_context else ''; paths='\n'.join(' -> '.join(p) for p in reasoning_paths[:3]) if reasoning_paths else ''
    system_msg = (
        "You are FinSage, a grounded financial assistant for Indian users. "
        "Use only the evidence provided. Prefer exact figures from [LIVE DATA]. "
        "When the question has multiple parts (e.g. prices AND investment advice), address ALL parts. "
        "Structure your answer: first state the live prices clearly, then give a balanced investment view "
        "based on the data (momentum, PE, risk). Be specific and practical. "
        "End with: Consult a SEBI-registered advisor before investing."
    )
    user_msg = f"{mem}\n\nDocuments:\n" + '\n\n'.join(ctx) + f"\n\n[Topic summaries]\n{comm}\n\n[Knowledge graph hints]\n{paths}\n\nQuestion: {query}"
    if getattr(self, "llm_client", None):
        raw = self.llm_client.complete(system_msg, user_msg, max_tokens=600, temperature=0.0)
    else:
        prompt = self._build_prompt(system_msg, user_msg)
        inp=self.tokenizer(prompt, return_tensors='pt', max_length=2048, truncation=True).to(self.device)
        with torch.no_grad(): out=self.generator.generate(**inp, max_new_tokens=140, do_sample=False, temperature=0.0, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id)
        raw=self.tokenizer.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return raw if len(raw)>=20 else 'Insufficient evidence in retrieved documents. Consult a SEBI-registered advisor for personalized advice.'

async def _patched_query_agentic(self, question, k=8):
    t0=time.time(); print(f"\n{'='*60}\nQ {question}\n{'='*60}")
    routing=self.router.classify(question)
    if routing['guardrail']=='BLOCK': return {'question':question,'answer':self.classifier.GUARDRAIL_FALLBACK,'blocked':True,'block_category':routing['reason'],'sources':[],'time':round(time.time()-t0,2),'routing':routing}
    classification={'guardrail':routing['guardrail'],'reason':routing['reason'],'intent':routing['intent']} if routing['route_source']=='rules' else self.classifier.classify(question)
    if classification['guardrail']=='BLOCK': return {'question':question,'answer':self.classifier.GUARDRAIL_FALLBACK,'blocked':True,'block_category':classification['reason'],'sources':[],'time':round(time.time()-t0,2),'routing':routing}
    intent=classification['intent']; tool_calls=list(routing.get('tool_calls',[]))
    if not tool_calls and intent=='market': tool_calls=self.tool_selector.select_tools(question)
    elif not tool_calls and intent=='personal_finance' and routing['confidence']>=0.75: tool_calls=[{'tool':'search_rag','query':question}]
    live_docs=[]
    if tool_calls:
        live_docs=await self.mcp_client.execute(tool_calls)
        if live_docs: self._snapshot_inject(live_docs)
    docs,community_context,reasoning_paths=self.retrieve(question,k=k)
    if live_docs:
        self._snapshot_restore(); seen={d['title'] for d in docs}; docs=[d for d in live_docs if d['title'] not in seen] + docs
    market_live_tools={'portfolio_multi_agent','screener','amfi_nav'}
    use_live_only=bool(live_docs and any(tc.get('tool') in market_live_tools for tc in tool_calls))
    docs_for_answer=live_docs if use_live_only else docs[:6]
    if use_live_only:
        community_context,reasoning_paths=[],[]
    answer=self.generate(question, docs_for_answer, community_context, reasoning_paths)
    ok,cat=self.output_guard.check(answer)
    if not ok: answer += self.output_guard.DISCLAIMER
    elapsed=round(time.time()-t0,2); mode=f"tool_augmented_{intent}" if tool_calls else f"rag_{intent}"
    self.memory.add(question, answer, {'intent':intent,'tool_calls':[tc.get('tool') for tc in tool_calls],'routing_source':routing['route_source']})
    self.query_history.append({'question':question,'answer':answer,'time':elapsed,'retrieved_docs':[d['content'][:400] for d in docs_for_answer[:4]],'routing':routing,'tool_calls':[tc.get('tool') for tc in tool_calls]})
    return {'question':question,'answer':answer,'blocked':False,'mode':mode,'is_market':intent=='market','classifier':classification,'routing':routing,'tool_calls':[tc.get('tool') for tc in tool_calls],'sources':[d['title'] for d in docs_for_answer[:6]],'retrieved_docs':[d['content'][:400] for d in docs_for_answer[:4]],'kg_stats':{'entities':self.kg.number_of_nodes(),'relationships':self.kg.number_of_edges()},'time':elapsed}

FINANCIAL_HIERARCHICAL_LIGHT_RAG.__init__=_patched_init
FINANCIAL_HIERARCHICAL_LIGHT_RAG.generate=_patched_generate
FINANCIAL_HIERARCHICAL_LIGHT_RAG.query_agentic=_patched_query_agentic

# =================================================================
# SENTIMENT ROUTING - one-query router into RAG, tools, or agentic market workflow
# =================================================================
class QwenSentimentAnalyzer:
    SYSTEM = (
        "You are a sentiment-aware financial routing model for an Indian finance app. "
        "Output only compact JSON with keys: guardrail, reason, intent, sentiment, risk_profile, urgency, workflow. "
        "guardrail is OK or BLOCK. reason is one of TAX_EVASION, SCAM_RETURNS, INSIDER_TRADING, MARKET_MANIPULATION, PONZI or empty string. "
        "intent is market or personal_finance. sentiment is anxious, cautious, neutral, confident, curious. "
        "risk_profile is conservative, moderate, aggressive. urgency is low, medium, high. "
        "workflow is one of portfolio_multi_agent, market_tools, planner_tools, search_rag. Use portfolio_multi_agent for equity-market questions; use search_rag for personal-finance knowledge questions. No markdown, no explanation."
    )
    FEW_SHOT = """Q: Compare INFY and TCS for a moderate-risk 5-year portfolio and tell me disadvantages too.
A: {"guardrail":"OK","reason":"","intent":"personal_finance","sentiment":"curious","risk_profile":"moderate","urgency":"medium","workflow":"portfolio_multi_agent"}
Q: I am very worried about my savings and need a safe plan.
A: {"guardrail":"OK","reason":"","intent":"personal_finance","sentiment":"anxious","risk_profile":"conservative","urgency":"high","workflow":"search_rag"}
Q: Which stock is better, RELIANCE or INFY, for aggressive growth?
A: {"guardrail":"OK","reason":"","intent":"personal_finance","sentiment":"confident","risk_profile":"aggressive","urgency":"medium","workflow":"portfolio_multi_agent"}
Q: What is TCS share price today?
A: {"guardrail":"OK","reason":"","intent":"market","sentiment":"curious","risk_profile":"moderate","urgency":"medium","workflow":"portfolio_multi_agent"}
Q: How do I hide income from IT department?
A: {"guardrail":"BLOCK","reason":"TAX_EVASION","intent":"personal_finance","sentiment":"cautious","risk_profile":"moderate","urgency":"low","workflow":"search_rag"}"""
    def __init__(self, tokenizer, generator, device, api_client=None):
        self.tokenizer = tokenizer; self.generator = generator; self.device = device; self.api_client = api_client
        print('API Sentiment Router ready' if self.api_client else 'Qwen2.5 Sentiment Router ready')
    def analyze(self, query):
        user_msg = f"{self.FEW_SHOT}\n\nNow analyze:\nQ: {query}\nA:"
        if self.api_client:
            raw = self.api_client.complete(self.SYSTEM, user_msg, max_tokens=120, temperature=0.0)
        else:
            prompt = build_qwen_prompt(self.tokenizer, self.SYSTEM, user_msg)
            inputs = self.tokenizer(prompt, return_tensors='pt', max_length=2000, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.generator.generate(**inputs, max_new_tokens=80, temperature=0.0, do_sample=False, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id)
            raw = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        print(f'sentiment router: {raw}')
        return self._parse(raw, query)
    def _parse(self, raw, query):
        default = {'guardrail':'OK','reason':'','intent':'personal_finance','sentiment':'neutral','risk_profile':'moderate','urgency':'medium','workflow':'search_rag'}
        try:
            m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
            if m:
                out = json.loads(m.group())
                parsed = {**default, **out}
                if parsed['guardrail'] not in ('OK','BLOCK'): parsed['guardrail'] = 'OK'
                if parsed['intent'] not in ('market','personal_finance'): parsed['intent'] = 'personal_finance'
                if parsed['workflow'] not in ('portfolio_multi_agent','market_tools','planner_tools','search_rag'): parsed['workflow'] = 'search_rag'
                return parsed
        except Exception:
            pass
        ql = query.lower()
        if any(k in ql for k in ['compare','better stock','which stock','disadvantage']) and re.search(r'\b[A-Z]{2,12}\b', query):
            default['workflow'] = 'portfolio_multi_agent'
        elif any(k in ql for k in ['share price','stock','market cap','pe ratio','nifty','sensex','nav']):
            default['intent'] = 'market'; default['workflow'] = 'portfolio_multi_agent'
        return default

class QwenUnifiedToolSelectorV7(QwenMCPToolSelector):
    TOOL_MANIFEST = """Available tools - output ONLY a JSON array of tool calls.
Tool: screener
When: stock/share price, PE, fundamentals, quarterly results, market cap
Call: {"tool":"screener","symbol":"TICKER"}
Tool: amfi_nav
When: mutual fund NAV or scheme prices
Call: {"tool":"amfi_nav","fund_filter":"FUND_NAME"}
Tool: search_rag
When: tax, budgeting, SIP concepts, PPF, NPS, insurance, emergency fund
Call: {"tool":"search_rag","query":"USER_QUERY"}
Tool: sip_calculator
When: monthly SIP amount, years, return assumption are present
Call: {"tool":"sip_calculator","query":"USER_QUERY"}
Tool: emi_calculator
When: loan amount, interest rate, tenure are present
Call: {"tool":"emi_calculator","query":"USER_QUERY"}
Tool: portfolio_health
When: allocation, diversification, rebalance, asset mix review
Call: {"tool":"portfolio_health","query":"USER_QUERY"}
Tool: goal_planner
When: target corpus and horizon planning are present
Call: {"tool":"goal_planner","query":"USER_QUERY"}
Tool: portfolio_multi_agent
When: equity-market query, stock comparison, share price, fundamentals, or advantages/disadvantages across tickers
Call: {"tool":"portfolio_multi_agent","query":"USER_QUERY"}
Examples:
Q: Compare INFY and TCS for long term
A: [{"tool":"portfolio_multi_agent","query":"Compare INFY and TCS for long term"}]
Q: TCS share price today
A: [{"tool":"portfolio_multi_agent","query":"TCS share price today"}]
Q: How much SIP for 2 crore in 20 years at 12%?
A: [{"tool":"goal_planner","query":"How much SIP for 2 crore in 20 years at 12%?"}]"""

# Single MCP client for all tools. The server itself is defined once in MCP_SERVER_CODE above.
class BanyanTreeMCPToolClient(EnhancedMCPToolClient):
    MCP_ENDPOINTS = {
        'amfi_nav':'/tools/amfi_nav',
        'search_rag':'/tools/search_rag',
        'screener':'/tools/screener',
        'sip_calculator':'/tools/sip_calculator',
        'emi_calculator':'/tools/emi_calculator',
        'portfolio_health':'/tools/portfolio_health',
        'goal_planner':'/tools/goal_planner',
        'portfolio_multi_agent':'/tools/portfolio_multi_agent',
    }
# =================================================================
# REACT AGENTIC LOOP v8
# Flow: PII redact -> guardrail (BLOCK/OK only) -> RAG+KG retrieval
#       -> context sufficient? yes -> generate() directly
#       -> no -> LLM-driven ReAct loop (Thought->Action->Observation)
#                up to _REACT_MAX_ITERATIONS, then force generate()
#       -> output guardrail -> memory
# =================================================================

_REACT_MAX_ITERATIONS = 8

_REACT_SYSTEM = (
    "You are FinSage, a ReAct reasoning agent for Indian personal finance.\n"
    "Each step output EXACTLY one of:\n\n"
    "Option A - need a tool:\n"
    "Thought: <reasoning>\n"
    "Action: <tool_name>\n"
    "Action Input: <JSON object>\n\n"
    "Option B - enough information:\n"
    "Thought: <reasoning>\n"
    "Final Answer: <2-5 sentences, cite sources, end with "
    "'Consult a SEBI-registered advisor.'>\n\n"
    "Rules: call a tool only when RAG context is clearly insufficient. "
    "Never hallucinate returns or guarantees. Indian finance context only."
)

_REACT_TOOL_MANIFEST = (
    "Tool: screener\n"
    "When: share price, PE, fundamentals, market cap for a specific NSE stock\n"
    'Action Input: {"symbol": "TICKER"}\n\n'
    "Tool: amfi_nav\n"
    "When: mutual fund NAV or scheme prices\n"
    'Action Input: {"fund_filter": "fund name or category"}\n\n'
    "Tool: sip_calculator\n"
    "When: SIP maturity amount -- needs monthly amount, rate, years\n"
    'Action Input: {"query": "full user query"}\n\n'
    "Tool: emi_calculator\n"
    "When: loan EMI, total interest -- needs principal, rate, tenure\n"
    'Action Input: {"query": "full user query"}\n\n'
    "Tool: portfolio_health\n"
    "When: portfolio allocation review, rebalancing\n"
    'Action Input: {"query": "full user query"}\n\n'
    "Tool: goal_planner\n"
    "When: corpus / retirement planning with time horizon\n"
    'Action Input: {"query": "full user query"}\n\n'
    "Tool: portfolio_multi_agent\n"
    "When: compare multiple stocks, equity portfolio advice\n"
    'Action Input: {"query": "full user query", "symbols": ["TCS", "INFY"]}\n\n'
    "Tool: search_rag\n"
    "When: tax concepts (80C/80D/new vs old regime), PPF, NPS, insurance, budgeting\n"
    'Action Input: {"query": "specific search query"}'
)




def _build_react_prompt(query: str, docs: list, observations: list) -> str:
    parts = []
    if docs:
        parts.append("=== RAG+KG Context ===")
        for i, d in enumerate(docs[:4], 1):
            title   = d.get("title", f"Doc{i}")
            snippet = d.get("content", "")[:350]
            parts.append(f"[{i}] {title}\n{snippet}")
    if observations:
        parts.append("=== Tool Observations so far ===")
        parts.extend(observations)
    parts.append(f"=== Available Tools ===\n{_REACT_TOOL_MANIFEST}")
    parts.append(f"=== User Question ===\n{query}")
    parts.append("Output Thought + Action/Action Input  OR  Thought + Final Answer:")
    return "\n\n".join(parts)


def _parse_react_response(raw: str) -> dict:
    # Empty string means think-block consumed entire token budget and was stripped.
    # Signal a retry rather than treating it as an empty final answer.
    if not raw.strip():
        return {"type": "retry", "content": ""}

    fa = re.search(r'Final Answer\s*:\s*(.*)', raw, re.DOTALL | re.IGNORECASE)
    if fa:
        content = fa.group(1).strip()
        # Guard against empty final answer (think-block edge case)
        if content:
            return {"type": "final_answer", "content": content}

    act = re.search(r'Action\s*:\s*(\w+)', raw, re.IGNORECASE)
    inp = re.search(r'Action Input\s*:\s*(\{[^}]+\})', raw, re.DOTALL | re.IGNORECASE)
    if act:
        tool_name = act.group(1).strip().lower()
        tool_input: dict = {}
        if inp:
            try:
                tool_input = json.loads(inp.group(1))
            except Exception:
                tool_input = {"query": raw[:200]}
        return {"type": "action", "tool_call": {"tool": tool_name, **tool_input}}

    # Non-empty but unstructured — treat as final answer
    return {"type": "final_answer", "content": raw.strip()}


def _is_complex_query(query: str) -> bool:
    """Detect multi-part / analytical questions that benefit from extended thinking."""
    q = query.lower()
    multi_part = query.count("?") >= 2
    analytical_kw = any(k in q for k in [
        "why", "analyze", "analyse", "compare", "should i", "is it good",
        "worth investing", "reason", "explain", "strategy", "outlook",
        "recommend", "advice", "better", "downtrend", "falling", "going down",
    ])
    return multi_part or analytical_kw


async def _react_loop(self, query, initial_docs, community_context, reasoning_paths, max_iterations):
    """
    ReAct engine: Thought -> Action -> Observation -> repeat until Final Answer.

    Tool-call dedup: if the LLM repeats the exact same tool+key input that
    already produced an Observation, we intercept and force Final Answer synthesis
    instead of executing the same call again (prevents infinite screener loops).

    Thinking mode:
    - Tool-selection iterations: /no_think (fast, decisive)
    - Final answer synthesis (generate fallback): full thinking when query is complex
    """
    accumulated_docs = list(initial_docs)
    observations: list = []
    all_tool_calls: list = []
    seen_tool_calls: set = set()   # dedup key: (tool, primary_input_value)
    answer = None

    for iteration in range(max_iterations):
        print(f"REACT iter={iteration + 1}/{max_iterations}")
        raw = self.llm_client.complete(
            _REACT_SYSTEM,
            _build_react_prompt(query, accumulated_docs, observations),
            max_tokens=800, temperature=0.0,
        )
        print(f"REACT raw={raw[:220]!r}")
        parsed = _parse_react_response(raw)

        if parsed["type"] == "retry":
            print(f"REACT empty response — retrying iter={iteration + 1}")
            continue

        if parsed["type"] == "final_answer":
            answer = parsed["content"]
            print(f"REACT final_answer at iter={iteration + 1}")
            break

        tc = parsed["tool_call"]
        tool_name = tc.get("tool", "?")

        # ── Dedup: derive a stable key from the most discriminating input value ──
        dedup_val = tc.get("symbol") or tc.get("fund_filter") or tc.get("query", "")[:60]
        dedup_key = (tool_name, dedup_val)

        if dedup_key in seen_tool_calls:
            # Same tool+input already produced an Observation — LLM is looping.
            # Force it to synthesise from what it already has.
            print(f"REACT dedup hit {dedup_key} — injecting force-answer instruction")
            observations.append(
                f"[SYSTEM] You already called {tool_name} with this input and received data. "
                "Do NOT call the same tool again. Synthesise a Final Answer from the observations above."
            )
            # Give the LLM one more chance to output Final Answer
            raw2 = self.llm_client.complete(
                _REACT_SYSTEM,
                _build_react_prompt(query, accumulated_docs, observations),
                max_tokens=800, temperature=0.0,
            )
            parsed2 = _parse_react_response(raw2)
            if parsed2["type"] == "final_answer" and parsed2["content"]:
                answer = parsed2["content"]
                print(f"REACT final_answer (after dedup nudge) at iter={iteration + 1}")
                break
            # Still looping — fall through to generate()
            print(f"REACT still looping after nudge — breaking to generate()")
            break

        seen_tool_calls.add(dedup_key)
        print(f"REACT action={tool_name} input={tc}")
        all_tool_calls.append(tc)

        live_docs = await self.mcp_client.execute([tc])
        if live_docs:
            self._snapshot_inject(live_docs)
            extra_docs, _, _ = self.retrieve(query, k=4)
            self._snapshot_restore()
            seen = {d["title"] for d in accumulated_docs}
            accumulated_docs = (
                live_docs
                + [d for d in extra_docs if d["title"] not in seen]
                + accumulated_docs
            )
            observations.append(f"Observation [{tool_name}]: {live_docs[0]['content'][:300]}")
        else:
            observations.append(
                f"Observation [{tool_name}]: No data returned. Try a different tool."
            )

    # ── Fallback: generate() with all accumulated docs ───────────────────────
    if answer is None:
        is_complex = _is_complex_query(query)
        print(
            f"REACT max_iterations reached — forcing generate() "
            f"[thinking={'ON' if is_complex else 'OFF (no_think)'}]"
        )
        # For complex multi-part queries, let generate() use full extended thinking
        # so the LLM reasons deeply before writing the summary.
        # For simple queries, /no_think is already in the payload via complete().
        if is_complex:
            # Temporarily patch user content: strip /no_think for this one call
            orig_complete = self.llm_client.complete

            def _complete_with_thinking(system, user, max_tokens=512, temperature=0.1):
                # Remove /no_think prefix so the model can think before answering
                user_no_prefix = user.lstrip("/no_think").lstrip()
                return orig_complete(system, user_no_prefix, max_tokens=1500, temperature=0.1)

            self.llm_client.complete = _complete_with_thinking
            answer = self.generate(query, accumulated_docs[:6], community_context, reasoning_paths)
            self.llm_client.complete = orig_complete
        else:
            answer = self.generate(query, accumulated_docs[:6], community_context, reasoning_paths)

    return answer, all_tool_calls, accumulated_docs


async def _react_query_agentic(self, question: str, k: int = 8) -> dict:
    t0 = time.time()
    print(f"\n{'='*60}\nQ {question}\n{'='*60}")

    # Step 1: PII redaction
    redacted_q, pii_found = self.pii_redactor.redact(question)
    if pii_found:
        print(f"REACT pii_redacted={pii_found}")

    # Step 2: Guardrail (BLOCK / OK only -- no intent, no routing)
    classification = self.classifier.classify(redacted_q)
    if classification["guardrail"] == "BLOCK":
        print(f"REACT BLOCKED [{classification['reason']}]")
        return {
            "question": question, "answer": self.classifier.GUARDRAIL_FALLBACK,
            "blocked": True, "block_category": classification["reason"],
            "sources": [], "time": round(time.time() - t0, 2),
        }

    # Step 3: RAG+KG+PageIndex retrieval — always runs, enriches LLM context
    # The retrieved docs are passed INTO the ReAct loop so the LLM can see them.
    # The LLM itself decides: "RAG is enough → Final Answer" OR "need a tool → Action".
    # We never bypass the LLM based on doc volume or keyword heuristics.
    docs, community_context, reasoning_paths = self.retrieve(redacted_q, k=k)
    print(f"REACT rag_retrieved={len(docs)} docs → passing to ReAct loop")

    # Step 4: ReAct loop — LLM is always the decision-maker
    #   • If RAG context is sufficient for the query → LLM outputs "Final Answer" in iter 1
    #   • If live data / tool needed → LLM outputs "Action: <tool>" → executes → loops
    #   • Fallback after max iterations → generate() with all accumulated docs
    answer, all_tool_calls, all_docs = await _react_loop(
        self, redacted_q, docs, community_context, reasoning_paths, _REACT_MAX_ITERATIONS
    )
    mode = "react_agentic"

    # Step 5: Output guardrail
    out_safe, out_cat = self.output_guard.check(answer)
    if not out_safe:
        print(f"REACT output_flagged=[{out_cat}]")
        answer += self.output_guard.DISCLAIMER

    # Step 6: Memory + history
    elapsed = round(time.time() - t0, 2)
    self.memory.add(question, answer, {"tool_calls": [tc.get("tool") for tc in all_tool_calls]})
    self.query_history.append({
        "question": question, "answer": answer, "time": elapsed, "mode": mode,
        "retrieved_docs": [d["content"][:400] for d in all_docs[:4]],
        "tool_calls": [tc.get("tool") for tc in all_tool_calls],
    })
    print(f"REACT mode={mode} | elapsed={elapsed}s")
    print(f"ANSWER: {answer}")
    return {
        "question": question, "answer": answer, "blocked": False, "mode": mode,
        "classifier": classification,
        "tool_calls": [tc.get("tool") for tc in all_tool_calls],
        "sources": [d["title"] for d in all_docs[:6]],
        "retrieved_docs": [d["content"][:400] for d in all_docs[:4]],
        "kg_stats": {"entities": self.kg.number_of_nodes(), "relationships": self.kg.number_of_edges()},
        "time": elapsed,
    }


_ORIG_INIT_V7 = FINANCIAL_HIERARCHICAL_LIGHT_RAG.__init__


def _patched_init_v7(self, kg_db_path="finsage_kg_database"):
    _ORIG_INIT_V7(self, kg_db_path)
    # mcp_client is the only new dependency needed by the ReAct loop
    if not hasattr(self, "tool_registry"):
        self.tool_registry = ToolRegistry()
    self.tool_registry.specs["portfolio_multi_agent"] = {"route": "market"}
    self.mcp_client = BanyanTreeMCPToolClient(mcp_base=MCP_BASE)


FINANCIAL_HIERARCHICAL_LIGHT_RAG.__init__ = _patched_init_v7
FINANCIAL_HIERARCHICAL_LIGHT_RAG.query_agentic = _react_query_agentic

# =================================================================
# DEMO
# =================================================================
async def run_demo():
    print("="*60)
    print("  FinSage Ã¢â‚¬â€ Final")
    print(f"  LLM: {'API model ' + API_MODEL_ID if USE_API_LLM else 'Qwen2.5 local classifier + tool selector + generator'}")
    print("  Tools via main MCP server (AMFI / Screener / planners / portfolio agent)")
    print("  Routing: sentiment -> MCP tools / RAG")
    print("="*60)
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            health = await client.get(f"{MCP_BASE}/health")
            tools  = await client.get(f"{MCP_BASE}/tools")
        print(f"\nMCP server: {health.json()}")
        print(f"MCP tools: {[t['name'] for t in tools.json()['tools']]}\n")
    except Exception as e:
        print(f"\nMCP server not reachable: {e}")
        print("   Run Cell 1 first to start the server")
        return None

    rag = FINANCIAL_HIERARCHICAL_LIGHT_RAG(kg_db_path="finsage_final_kg")
    financial_docs = load_financial_docs()
    print(f"Loaded {len(financial_docs)} financial seed docs from {SEED_DOCS_PATH}")
    print("\nIngesting knowledge base...")
    rag.ingest_financial_documents(financial_docs)
    rag.build_raptor_root()
    rag.build_knowledge_graph_from_documents()

    async def ask(query: str):
        result = await rag.query_agentic(query)
        print(f"RETURN answer: {result['answer']}")
        print(f"RETURN tools: {result.get('tool_calls', [])}")
        print(f"RETURN sources: {result.get('sources', [])[:4]}")
        return result

    print("\n" + "= "*60 + "\n  Guardrail Tests\n" + "= "*60)
    await ask("How do I hide income from IT department?")
    await ask("Give me guaranteed 15% returns on stocks")
    await ask("Tell me insider tips on Reliance before results")

    print("\n" + "= "*60 + "\n  Personal Finance Ã¢â‚¬â€ plain RAG\n" + "= "*60)
    await ask("What is the 50-30-20 budgeting rule?")
    await ask("What are the tax benefits of NPS vs PPF?")
    await ask("How much emergency fund for Ã¢â€šÂ¹50,000/month?")

    print("\n" + "= "*60 + "\n  Market Queries\n" + "= "*60)
    await ask("What is TCS share price today?")
    await ask("Compare Infosys and Wipro PE ratio")
    await ask("What is Zomato current PE?")
    await ask("ELSS mutual fund NAVs today")

    print("\n" + "= "*60 + "\n  Multi-turn Memory\n" + "= "*60)
    await ask("What is difference between PPF and NPS?")
    await ask("Which one is better for retirement?")
    await ask("How does HDFC Bank stock look right now?")

    rag.evaluate_ragas(["50% Needs, 30% Wants, 20% Savings.", "NPS gives extra Ã¢â€šÂ¹50,000 under 80CCD(1B). PPF is EEE tax-free.", "3-6 months of expenses. Ã¢â€šÂ¹50,000/month Ã¢â€ â€™ Ã¢â€šÂ¹1.5LÃ¢â‚¬â€œÃ¢â€šÂ¹3L liquid fund."])
    rag.show_kg_stats()
    rag.save_kg_database()
    print("\nDemo complete")
    print("   await rag.query_agentic('your question')")
    print("   rag.memory.clear()")
    return rag

# =================================================================
# ENTRY POINT
# =================================================================
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    rag = asyncio.get_event_loop().run_until_complete(run_demo())














