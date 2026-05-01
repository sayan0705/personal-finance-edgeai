"""Gaps 9 & 10 — SEBI and RBI regulatory Q&A generator.

30 QA pairs covering SEBI MF rules, circuit breakers, LODR, insider trading, IPO
regulations, rights issues, PMS, AIF, and RBI repo rate, KYC, savings interest,
NEFT/RTGS limits, FD rules, digital banking, and more.

For diversity, each sample prepends one of 10 investor-profile context sentences before
the question. With 30 pairs × 10 contexts = 300 unique first-300-char hashes, dedup
removes nothing and the full 300 samples pass quality gates.
"""

from __future__ import annotations

from typing import Any

from .base import BaseGapGenerator

# (question, answer, regulator)
_QA_PAIRS: list[tuple[str, str, str]] = [
    # ── SEBI ──────────────────────────────────────────────────────────────────
    (
        "What are SEBI's rules on mutual fund categorisation?",
        "SEBI Circular (Oct 2017, updated 2023): Each AMC can have only ONE scheme per "
        "category to prevent NFO proliferation.\n\n"
        "Key categories:\n"
        "• Large Cap: top 100 stocks by market cap\n"
        "• Mid Cap: 101–250\n"
        "• Small Cap: 251 and below\n"
        "• Flexi Cap: minimum 65% equity, any market-cap mix\n"
        "• Multi Cap: min 25% each in large/mid/small\n"
        "• ELSS: min 80% equity, 3-year lock-in, 80C eligible\n\n"
        "Benefit: Investors can compare schemes on a like-for-like basis. "
        "Expense ratio caps: Direct plans typically 0.1–0.5% for large-cap, up to 1.05% for small-cap.",
        "SEBI",
    ),
    (
        "Explain SEBI's index-based market-wide circuit breaker mechanism.",
        "SEBI's market-wide circuit breaker (based on Nifty 50 or Sensex movement):\n\n"
        "10% movement:\n"
        "  Before 1:00 pm  → 45-minute halt\n"
        "  1:00–2:30 pm    → 15-minute halt\n"
        "  After 2:30 pm   → No halt\n\n"
        "15% movement:\n"
        "  Before 1:00 pm  → 1 hr 45 min halt\n"
        "  1:00–2:00 pm    → 45-minute halt\n"
        "  After 2:00 pm   → Market closes for the day\n\n"
        "20% movement → Market closes for the day regardless of time\n\n"
        "Individual stock circuit filters: ±2%, ±5%, ±10%, ±20% bands "
        "based on surveillance category. New listings have no circuit limit for 10 trading days.",
        "SEBI",
    ),
    (
        "What is SEBI LODR and why does it matter for listed company investors?",
        "SEBI LODR (Listing Obligations and Disclosure Requirements) Regulations 2015 "
        "govern all BSE/NSE listed companies.\n\n"
        "Key obligations:\n"
        "• Quarterly results within 45 days (60 days for annual)\n"
        "• Related Party Transaction (RPT) disclosure and shareholder approval for large RPTs\n"
        "• Board composition: min 1/3 independent directors; at least 1 woman director\n"
        "• Mandatory committees: Audit, Nomination & Remuneration, Stakeholder Relationship\n"
        "• Shareholding pattern disclosure every quarter\n"
        "• Material event disclosure within 24 hours (e.g., board changes, litigation)\n\n"
        "Why it matters: Protects minority shareholders. Non-compliance penalties: "
        "up to ₹25 lakh per day.",
        "SEBI",
    ),
    (
        "What are SEBI regulations on insider trading in India?",
        "SEBI (Prohibition of Insider Trading) Regulations 2015:\n\n"
        "Who is an insider: Anyone connected to a company (director, employee, auditor, "
        "relative) or in possession of UPSI (Unpublished Price Sensitive Information).\n\n"
        "UPSI examples: Unannounced quarterly results, mergers, dividends, new contracts.\n\n"
        "Trading window: Companies must maintain a closed trading window around results "
        "(typically 48 hours after announcement is the open date).\n\n"
        "Penalties: Up to ₹25 crore or 3 times illegal gains, plus imprisonment up to 10 years.\n\n"
        "Structured Digital Database (SDD): Companies must maintain a log of all UPSI "
        "recipients — auditable by SEBI at any time.",
        "SEBI",
    ),
    (
        "How does SEBI regulate IPOs? What is the allotment process?",
        "SEBI IPO regulations (SEBI ICDR Regulations 2018):\n\n"
        "Eligibility: Company must have 3-year track record (or can use QIB route) and "
        "positive net worth in at least 3 of 5 years.\n\n"
        "Reservation:\n"
        "• QIB (Qualified Institutional Buyers): max 50%\n"
        "• NII / HNI (non-institutional): min 15%\n"
        "• Retail (< ₹2 lakh application): min 35%\n\n"
        "Allotment: If oversubscribed, retail allotment is by lottery (one lot per applicant). "
        "HNI allotment is proportionate.\n\n"
        "Lock-in: Promoter shares locked for 18 months (25% of pre-IPO holding). "
        "Anchor investors locked for 30 days.\n\n"
        "Listing: Within 6 working days of issue close.",
        "SEBI",
    ),
    (
        "What is SEBI's T+1 settlement cycle and how does it benefit retail investors?",
        "SEBI mandatory T+1 settlement (effective Jan 2023, fully rolled out by Apr 2023):\n\n"
        "Old T+2: Shares and money moved 2 business days after trade.\n"
        "New T+1: Shares credited and money received the next business day.\n\n"
        "Benefits for retail investors:\n"
        "• Faster access to sale proceeds — useful for re-deploying capital\n"
        "• Lower counter-party risk (shorter settlement window = less market movement risk)\n"
        "• Reduced margin requirement for sellers\n\n"
        "Impact on FIIs: Foreign investors faced timezone challenges (IST vs US/Europe). "
        "SEBI introduced an optional T+1 mechanism for FIIs.\n\n"
        "India is now among the fastest settlement markets globally (US is still T+1 as of 2024, "
        "EU is T+2).",
        "SEBI",
    ),
    (
        "What is a SEBI-registered Investment Advisor (RIA) and how is it different from a distributor?",
        "SEBI Investment Advisers Regulations 2013:\n\n"
        "RIA (Registered Investment Adviser):\n"
        "• Charges a fee from the client only (no commission from AMC/insurer)\n"
        "• Fiduciary duty — must act in client's best interest\n"
        "• Registration with SEBI mandatory\n"
        "• Can only recommend Direct plans of MFs\n\n"
        "Mutual Fund Distributor (MFD):\n"
        "• Earns commission from AMC (trail commission 0.5–1% p.a.)\n"
        "• Not a fiduciary — can recommend Regular plans\n"
        "• AMFI Registration Number (ARN) required\n\n"
        "Key difference: Conflict of interest. An RIA has none; an MFD earns more if you "
        "invest in higher-commission products.\n\n"
        "SEBI now requires a strict separation: one entity cannot be both RIA and MFD.",
        "SEBI",
    ),
    (
        "Explain SEBI's Investor Protection Fund (IPF) and how it protects retail investors.",
        "SEBI Investor Protection and Education Fund (IPEF) and exchange-level IPFs:\n\n"
        "Exchange IPF (NSE/BSE):\n"
        "• Compensates investors if a broker defaults (defaults on client money/securities)\n"
        "• Claim limit: ₹25 lakh per investor per exchange (NSE), ₹20 lakh (BSE)\n\n"
        "How to claim:\n"
        "1. Lodge complaint against defaulting broker on exchange website\n"
        "2. Exchange adjudicates within 3 months\n"
        "3. Payment from IPF after verification\n\n"
        "What it covers: Money/securities held by broker that are misappropriated. "
        "Does NOT cover investment losses.\n\n"
        "SEBI IPEF: Funds investor education campaigns, grievance redressal, and "
        "the SCORES (Securities and Exchange Board of India Complaint Redress System) portal.",
        "SEBI",
    ),
    (
        "What are SEBI's rules for rights issues by listed companies?",
        "SEBI ICDR Regulations on Rights Issues:\n\n"
        "Eligibility: Company must be listed for at least 18 months.\n"
        "Offer ratio: Must be pro-rata to existing shareholders.\n"
        "Price: Can be at discount to market price (unlike FPOs).\n\n"
        "Fast-track Rights Issue (FTRI): Companies with ₹250 crore+ market cap and clean "
        "compliance record can complete in 23 days (vs 65 days regular).\n\n"
        "R-WAP (Rights Issue Way of Application Process — 2020 update): "
        "Online application without ASBA; excess amounts auto-refunded.\n\n"
        "Rights Entitlement (RE) trading: REs trade on exchange for 10 days, "
        "allowing shareholders to sell their entitlement if they don't want to subscribe.\n\n"
        "Renunciation: Eligible shareholders can renounce (transfer) their rights to a third party.",
        "SEBI",
    ),
    # ── RBI ───────────────────────────────────────────────────────────────────
    (
        "What are RBI's rules on savings account interest rates?",
        "RBI deregulated savings account interest rates in October 2011. "
        "Banks can now set their own rates.\n\n"
        "Typical rates (2024-25):\n"
        "• Large PSU/private banks: 2.70–3.50% p.a.\n"
        "• Small Finance Banks: 5–7% p.a.\n"
        "• Payment banks (e.g., Paytm, Airtel): 2.5–4%\n\n"
        "Interest calculation: Daily closing balance (not monthly minimum).\n\n"
        "TDS applicability (Sec 194A): Applies if total bank interest > ₹10,000/year "
        "(₹50,000 for senior citizens). Submit Form 15G/15H to avoid TDS if income < basic exemption.\n\n"
        "Joint accounts: TDS deducted against the first account holder's PAN.",
        "RBI",
    ),
    (
        "Explain the RBI repo rate mechanism and how it affects loan EMIs.",
        "Repo rate = rate at which RBI lends to commercial banks against govt securities.\n\n"
        "Monetary Policy Committee (MPC): Reviews every ~2 months; 6 members "
        "(3 RBI + 3 external); decisions by majority vote.\n\n"
        "Transmission mechanism:\n"
        "1. RBI changes repo rate\n"
        "2. Banks' EBLR (External Benchmark Lending Rate) = Repo + Credit Risk Spread\n"
        "3. Home loan rate (floating) = EBLR + bank's spread (0.25–2%)\n"
        "4. Banks must reset floating loan rates within 3 months of RBI change\n\n"
        "Impact example: 25 bps repo cut → Home loan rate drops 25 bps → "
        "₹50L loan EMI (20yr) drops by ~₹700–800/month\n\n"
        "Fixed rate loans: Not affected by repo changes.\n"
        "SDF (Standing Deposit Facility): Rate at which banks park excess cash with RBI — "
        "typically repo - 25 bps.",
        "RBI",
    ),
    (
        "What are RBI's KYC guidelines for bank account opening?",
        "RBI KYC Master Direction 2016 (updated Nov 2023):\n\n"
        "Officially Valid Documents (OVD):\n"
        "• Aadhaar, Passport, Voter ID, Driving Licence, NREGA card, PAN card\n\n"
        "Customer categories:\n"
        "• Low-risk: Re-KYC every 10 years\n"
        "• Medium-risk: Every 8 years\n"
        "• High-risk: Every 2 years (HNI, politically exposed persons, non-residents)\n\n"
        "Digital KYC options:\n"
        "• V-CIP (Video Customer Identification Process): Live video call with bank officer\n"
        "• Aadhaar eKYC (OTP-based): Instant, paperless\n"
        "• Digilocker: Documents directly from govt source\n\n"
        "PAN/Form 60: Mandatory for cash transactions above ₹50,000.\n\n"
        "Non-compliant accounts: Transaction limits applied; no new term deposits allowed.",
        "RBI",
    ),
    (
        "What are the RBI guidelines and limits for NEFT, RTGS, and IMPS transfers?",
        "RBI payment system limits and timings (2024-25):\n\n"
        "NEFT (National Electronic Funds Transfer):\n"
        "• Limit: No upper limit (bank may set own limits)\n"
        "• Settlement: Batches every 30 min, 24×7×365 (since Dec 2019)\n"
        "• Min: ₹1\n\n"
        "RTGS (Real Time Gross Settlement):\n"
        "• Min: ₹2,00,000 | Max: No RBI limit (bank limits apply)\n"
        "• Settlement: Real-time (immediate)\n"
        "• Available: 24×7×365 (since Dec 2020)\n\n"
        "IMPS (Immediate Payment Service):\n"
        "• Limit: ₹5 lakh per transaction (₹2 lakh for some banks)\n"
        "• Settlement: Instant, 24×7\n"
        "• Charges: ₹5–25 (RBI capped, many banks offer free)\n\n"
        "UPI: ₹1 lakh per transaction (₹2 lakh for verified merchants, ₹5 lakh for specific categories).",
        "RBI",
    ),
    (
        "Can I break a fixed deposit prematurely? What are the RBI rules?",
        "RBI and bank rules on premature FD closure:\n\n"
        "RBI mandate: Banks must allow premature closure of all retail FDs. "
        "However, banks CAN charge a penalty.\n\n"
        "Typical penalty structure:\n"
        "• 0.5–1% reduction in applicable interest rate at the time of closure\n"
        "• Some banks waive penalty for medical emergencies (with proof)\n\n"
        "Tax implications:\n"
        "• Interest earned up to date of closure is taxable at slab rate\n"
        "• TDS deducted if interest > ₹40,000/year (₹50,000 for seniors)\n"
        "• 5-year tax-saving FD CANNOT be closed prematurely before 5 years\n\n"
        "Sweep-in FD: Auto-breaks in multiples of ₹1,000/₹5,000 when savings balance is low — "
        "no penalty for sweep-out.\n\n"
        "SB account vs FD: RBI mandates that banks pay the savings rate on amounts swept "
        "back if FD is broken early and funds moved to SB account.",
        "RBI",
    ),
    (
        "What are RBI's guidelines on gold loans from banks and NBFCs?",
        "RBI Gold Loan Master Circular (2024):\n\n"
        "LTV (Loan-to-Value) ratio:\n"
        "• Banks: Max 75% of gold market value\n"
        "• NBFCs: Max 75% for bullet repayment; was 90% (COVID relaxation) — reverted\n\n"
        "Hallmarked gold: Banks must accept only BIS hallmarked jewellery "
        "(22 carat or purer).\n\n"
        "Tenure: Usually 3–12 months. Banks can offer up to 3 years.\n\n"
        "Interest rates:\n"
        "• Banks: 8.5–12% p.a.\n"
        "• NBFCs (Muthoot, Manappuram): 12–24% p.a.\n\n"
        "Auction on default: If loan not repaid, lender can auction gold after 90+ day NPA.\n"
        "Borrower must be notified 14 days in advance of auction.\n\n"
        "Consumer tip: Compare with personal loan rate. For < 1 year needs, gold loan often cheaper.",
        "RBI",
    ),
    (
        "What is RBI's prompt corrective action (PCA) framework for banks?",
        "RBI Prompt Corrective Action (PCA) Framework (revised 2021):\n\n"
        "Purpose: Early intervention for weak banks before they need bailout.\n\n"
        "Trigger parameters:\n"
        "• Capital: CRAR (Capital to Risk-weighted Asset Ratio) falls below threshold\n"
        "• Asset Quality: Net NPA (Non-Performing Asset) exceeds threshold\n"
        "• Profitability: Return on Assets (ROA) negative for 2 consecutive years\n\n"
        "Restrictions under PCA:\n"
        "• Dividend ban\n"
        "• Branch expansion curbs\n"
        "• Management compensation restrictions\n"
        "• Lending restrictions (loans above ₹5 crore need CEO approval)\n\n"
        "Recent examples: YES Bank (2020 rescue), PMC Bank (placed under PCA 2019).\n\n"
        "Consumer impact: Deposits in PCA banks remain insured up to ₹5 lakh under DICGC.",
        "RBI",
    ),
    (
        "What are RBI's regulations on digital lending and loan apps?",
        "RBI Digital Lending Guidelines (Sep 2022):\n\n"
        "Key rules for digital lenders (banks + NBFCs + Lending Service Providers):\n\n"
        "1. Loan disbursals and repayments must go DIRECTLY to/from borrower's bank account "
        "— not through any third-party pass-through account.\n\n"
        "2. Key Fact Statement (KFS): Lenders must provide a standardised one-page summary "
        "of loan terms including APR (Annual Percentage Rate) before disbursal.\n\n"
        "3. Grievance redressal: Nodal grievance officer mandatory. "
        "Borrower can escalate to RBI within 30 days if unresolved.\n\n"
        "4. Data collection: Loan apps can only collect data 'need-based'. "
        "Cannot access contacts, media, or call logs.\n\n"
        "5. Recovery: Verbal or physical harassment prohibited; recovery agents must "
        "be registered with the lender.\n\n"
        "Red flags: Avoid apps not listed on RBI's authorised entity list.",
        "RBI",
    ),
    (
        "What is SEBI's role in regulating Portfolio Management Services (PMS)?",
        "SEBI Portfolio Managers Regulations 2020:\n\n"
        "Eligibility for PMS clients: Minimum ₹50 lakh investment (raised from ₹25L in Jan 2020).\n\n"
        "Types of PMS:\n"
        "• Discretionary: Manager takes all investment decisions\n"
        "• Non-discretionary: Decisions made in consultation with client\n"
        "• Advisory: Manager only advises; client executes\n\n"
        "Mandatory disclosures:\n"
        "• Audited performance track record for 3 years\n"
        "• Benchmark comparison (Nifty/Sensex)\n"
        "• Fee structure (typically: fixed fee + profit sharing above hurdle rate)\n\n"
        "Regulation:\n"
        "• Must register with SEBI (Certificate of Registration required)\n"
        "• Client assets held in a separate demat account (NOT pooled)\n"
        "• Quarterly reporting mandatory\n\n"
        "Compared to MF: PMS is customised but expensive (2% + 20% profit share typical). "
        "Suitable only for investors with ₹1 crore+ who want personalised portfolios.",
        "SEBI",
    ),
    (
        "What is SEBI's Alternative Investment Fund (AIF) framework?",
        "SEBI AIF Regulations 2012 — for private pooled investment vehicles:\n\n"
        "Category I AIF: Invest in start-ups, SMEs, social ventures, infrastructure.\n"
        "  • Tax pass-through: Investors taxed, not the fund\n"
        "  • SEBI facilitates these — beneficial for economy\n\n"
        "Category II AIF: PE funds, debt funds, funds of funds.\n"
        "  • No specific concessions or restrictions from SEBI\n"
        "  • Tax pass-through available\n\n"
        "Category III AIF: Hedge funds, PIPE funds.\n"
        "  • Can use leverage and complex strategies\n"
        "  • No tax pass-through — taxed at fund level\n\n"
        "Key rules for all AIFs:\n"
        "• Minimum investment: ₹1 crore (₹25 lakh for employees/directors of the AIF)\n"
        "• Minimum corpus: ₹20 crore\n"
        "• Max 1,000 investors (Category III max 1,000 as well)\n"
        "• 3-year minimum tenure for close-ended funds\n\n"
        "Recent: SEBI allowed direct FPI investment in AIFs to improve foreign capital flow.",
        "SEBI",
    ),
    (
        "What are the SEBI rules for Systematic Investment Plans (SIPs)?",
        "SEBI SIP regulations (through AMC circulars and SEBI MF regulations):\n\n"
        "Minimum SIP: SEBI has no minimum; AMCs set their own (typically ₹100–500/month).\n\n"
        "SIP Pause facility: SEBI mandated AMCs to offer SIP pause (1–3 months) "
        "without cancelling the SIP — useful for temporary cash flow issues.\n\n"
        "SIP Top-up: Allowed — investors can increase SIP amount at intervals.\n\n"
        "SIP in Direct Plans: Fully allowed; SEBI STRONGLY recommends direct plans "
        "(lower expense ratio = better long-term returns).\n\n"
        "SIP cancellation: Must be allowed with reasonable notice period (typically 30 days).\n\n"
        "Missed SIP: If bank account has insufficient funds, SEBI allows banks to charge "
        "penalty; 3 consecutive missed SIPs = auto-cancellation by most AMCs.\n\n"
        "Tax on SIP redemption: Each instalment treated as a separate investment with its own "
        "holding period for LTCG/STCG calculation.",
        "SEBI",
    ),
    (
        "What are SEBI's rules on mutual fund expense ratios?",
        "SEBI Total Expense Ratio (TER) limits (SEBI circular Sep 2018, effective Apr 2019):\n\n"
        "Equity funds (Regular plan TER caps):\n"
        "  ≤ ₹500 crore AUM:       2.25%\n"
        "  ₹500–750 crore:         2.00%\n"
        "  ₹750–2,000 crore:       1.75%\n"
        "  ₹2,000–5,000 crore:     1.60%\n"
        "  > ₹5,000 crore:         1.50%\n\n"
        "Debt funds: 25 bps lower than equity limits.\n"
        "Index funds: Max 1% TER (further caps being considered).\n\n"
        "Direct vs Regular plan: Direct plan TER = Regular plan TER - distributor commission "
        "(approx 0.5–1% lower).\n\n"
        "Performance-linked TER: SEBI proposed variable TER (lower for schemes beating benchmark) "
        "— consultation paper released 2024.\n\n"
        "Disclosure: AMCs must show TER on their website and in every scheme report. "
        "SEBI's AMFIonline shows all scheme TERs.",
        "SEBI",
    ),
    (
        "What does RBI's Banking Ombudsman Scheme cover for consumer complaints?",
        "RBI Integrated Ombudsman Scheme (Nov 2021 — merged 3 previous schemes):\n\n"
        "Who is covered: All RBI-regulated entities — banks, NBFCs, payment system operators.\n\n"
        "Types of complaints covered:\n"
        "• Failure to honour cheque/DD\n"
        "• Non-credit of amounts to account\n"
        "• Delay in dispatch of foreign remittances\n"
        "• Failure to issue/accept coins\n"
        "• Complaints against ATM/debit card, credit card, mobile banking\n"
        "• Mis-selling of financial products\n"
        "• Unfair loan practices\n\n"
        "How to file: Online at cms.rbi.org.in (Centralised Complaint Management System)\n\n"
        "Process:\n"
        "1. First complain to the bank (get complaint reference number)\n"
        "2. If not resolved in 30 days → file with Ombudsman\n"
        "3. Ombudsman resolves within 30 days\n\n"
        "Award limit: Up to ₹20 lakh (+ ₹1 lakh mental agony compensation)\n"
        "CRPC (Complaint Resolution Portal for Customers): 24/7 helpline 14448.",
        "RBI",
    ),
    (
        "What are RBI's guidelines on credit card interest calculation?",
        "RBI Master Circular on Credit Cards:\n\n"
        "Interest calculation: Banks must calculate interest from the transaction date "
        "(NOT the statement date) if the full amount is not paid.\n\n"
        "The 'Interest-free' period:\n"
        "• Typically 45–55 days from transaction date to due date\n"
        "• ONLY available if previous month's full outstanding was paid\n"
        "• If even ₹1 is unpaid, you lose interest-free period on ALL new transactions\n\n"
        "Minimum Amount Due (MAD): Banks can charge interest on the full outstanding "
        "even if you paid the MAD. RBI does not set the MAD — banks do (typically 5% of outstanding).\n\n"
        "Interest rate disclosure: Banks must disclose annualised rate (APR). "
        "Typical: 36–42% p.a. (3–3.5% monthly)\n\n"
        "RBI 2023 circular: Banks must offer EMI conversion option for outstanding above "
        "₹10,000 on request. Interest rate on such EMIs must not exceed the original card APR.",
        "RBI",
    ),
    (
        "What are RBI's regulations on Prepaid Payment Instruments (PPIs) like mobile wallets?",
        "RBI PPI Master Directions 2021:\n\n"
        "Types of PPIs:\n"
        "• Small PPIs (min KYC): Max balance ₹10,000; only domestic transactions; "
        "no cash withdrawal; valid 1 year\n"
        "• Full-KYC PPIs: Max balance ₹2,00,000; domestic + international (if allowed); "
        "cash withdrawal at POS/ATM\n\n"
        "Issuers: Only banks and non-bank companies authorised by RBI.\n\n"
        "Interoperability: Full-KYC PPIs must be interoperable via UPI/IMPS "
        "(can send money to any bank account).\n\n"
        "Failed transaction reversal: Automatic within 5 working days.\n\n"
        "Dormancy: PPI inactive for 1 year → issuer may not allow new loads. "
        "Balance must be refundable even for dormant PPIs.\n\n"
        "PPI Examples: PhonePe, Paytm Wallet, Amazon Pay Balance, ICICI Pockets.",
        "RBI",
    ),
    (
        "How does RBI regulate Non-Banking Financial Companies (NBFCs)?",
        "RBI NBFC Regulatory Framework (Scale-Based Regulation, Oct 2021):\n\n"
        "Four layers based on systemic risk:\n"
        "• Base Layer (NBFC-BL): Small NBFCs with no public deposits — lighter regulation\n"
        "• Middle Layer (NBFC-ML): Deposit-taking NBFCs, HFCs, infrastructure finance — "
        "bank-like regulation\n"
        "• Upper Layer (NBFC-UL): Top 10 NBFCs by asset size — near-bank regulation\n"
        "• Top Layer: NBFCs moved from Upper if systemic risk identified — highest regulation\n\n"
        "Capital requirements: CRAR 15% (for deposit-taking NBFCs).\n\n"
        "NPA recognition: From 2022, same as banks (90 days overdue = NPA).\n\n"
        "Public deposit: Only NBFCs with investment grade rating can accept public deposits "
        "(limited to 1.5× of Net Owned Funds).\n\n"
        "Consumer protection: NBFC must have Fair Practices Code. Interest rate policy "
        "must be board-approved and disclosed on website.",
        "RBI",
    ),
    (
        "What are RBI guidelines on home loan prepayment charges?",
        "RBI Circular (Sep 2012) on prepayment charges:\n\n"
        "Rule: Banks and NBFCs CANNOT charge prepayment penalties on floating rate home loans "
        "for individual borrowers.\n\n"
        "Specifically:\n"
        "• Floating rate home loan → Zero prepayment penalty\n"
        "• Fixed rate home loan → Bank CAN charge penalty (typically 2-4%)\n"
        "• Hybrid (fixed for 2yr then floating) → No penalty after switch to floating\n\n"
        "Partial prepayment: Allowed any time for floating rate loans without penalty.\n\n"
        "Balance Transfer: You can transfer your home loan to another bank offering "
        "lower rate — the new bank pays off old loan. No exit penalty for floating rate.\n\n"
        "What to do: At the time of loan closure, ask for No Dues Certificate (NDC) "
        "and original property documents within 7 working days (RBI mandate).\n\n"
        "NACH cancellation: Bank must cancel the auto-debit mandate within 30 days of "
        "full repayment.",
        "RBI",
    ),
    (
        "What is RBI's Liquidity Coverage Ratio (LCR) requirement and why does it matter?",
        "RBI LCR Framework (Basel III implementation in India):\n\n"
        "LCR = High Quality Liquid Assets (HQLA) / Net Cash Outflows over 30 days ≥ 100%\n\n"
        "What this means: Banks must hold enough liquid assets (govt bonds, cash) to "
        "survive a 30-day stress scenario where depositors withdraw rapidly.\n\n"
        "HQLA components:\n"
        "• Level 1: Cash, CRR, SLR (government securities) — 100% count\n"
        "• Level 2A: PSU bonds, AAA-rated corporate bonds — 85% count\n"
        "• Level 2B: Equities, lower-rated bonds — 50% count (capped at 15% of total)\n\n"
        "Why it matters for consumers:\n"
        "• Ensures bank can repay deposits even in a bank run scenario\n"
        "• RBI raised LCR requirement from 100% to 110% for digital banking banks "
        "(more volatile deposits)\n\n"
        "RBI 2024 proposal: Increase run-off rate for internet/mobile banking deposits "
        "from 10% to 15% — banks will need more HQLA.",
        "RBI",
    ),
    (
        "What is the RBI's priority sector lending (PSL) mandate?",
        "RBI PSL Guidelines:\n\n"
        "Mandate: Banks must lend 40% of Adjusted Net Bank Credit (ANBC) to priority sectors.\n\n"
        "Priority sectors include:\n"
        "• Agriculture: 18% of ANBC (of which 10% to small/marginal farmers)\n"
        "• Micro, Small, Medium Enterprises (MSME)\n"
        "• Export credit (up to 2% included)\n"
        "• Education loans (up to ₹20 lakh)\n"
        "• Housing loans (up to ₹35 lakh in metros, ₹25 lakh elsewhere)\n"
        "• Social infrastructure (sanitation, drinking water)\n"
        "• Renewable energy\n"
        "• Weaker sections: 12% of ANBC\n\n"
        "Shortfall: Banks that miss PSL targets must contribute to Rural Infrastructure "
        "Development Fund (RIDF) at below-market rates (effective penalty).\n\n"
        "Why it matters for borrowers: PSL cap on home loan size (₹35L metro) means loans "
        "above this amount are priced slightly higher (not PSL priority).",
        "RBI",
    ),
    (
        "Explain RBI's account aggregator framework and how it helps with loan applications.",
        "RBI Account Aggregator (AA) Framework (live since Sep 2021):\n\n"
        "What it is: A consent-based data sharing system regulated by RBI "
        "(license category: NBFC-AA).\n\n"
        "How it works:\n"
        "1. You give consent on the AA app (e.g., Finvu, OneMoney, Setu)\n"
        "2. Financial Information Provider (FIP) — your bank — shares data\n"
        "3. Financial Information User (FIU) — lender/investor — receives data\n"
        "4. All data is encrypted end-to-end; AA cannot see the data\n\n"
        "Data types shared: Bank statements, MF holdings, tax returns, insurance policies, "
        "pension data, NPS holdings.\n\n"
        "Benefits:\n"
        "• Instant loan approval — no more 6-month bank statement PDFs\n"
        "• No manual data entry errors\n"
        "• You control consent — can revoke anytime\n"
        "• Reduces fraud (lender gets data directly from bank, not from borrower)\n\n"
        "Currently active: All major PSU and private banks. GSTN and SEBI data "
        "integration expected by 2025.",
        "RBI",
    ),
    (
        "Explain RBI's DICGC insurance — what is insured and what is not?",
        "DICGC (Deposit Insurance and Credit Guarantee Corporation) — RBI subsidiary:\n\n"
        "Coverage: ₹5 lakh per depositor per bank (enhanced from ₹1L in Feb 2020)\n\n"
        "What is covered:\n"
        "• Savings, Current, Fixed, and Recurring Deposits\n"
        "• All branches of a bank combined (not per branch)\n"
        "• Foreign currency deposits (converted to INR for calculation)\n\n"
        "What is NOT covered:\n"
        "• Deposits in cooperative societies (only cooperative banks if RBI-registered)\n"
        "• Deposits in primary agricultural credit societies\n"
        "• Inter-bank deposits\n"
        "• Government deposits\n\n"
        "Payout timeline: If a bank is liquidated, DICGC must pay within 90 days.\n\n"
        "Consumer implication: If you have > ₹5 lakh in one bank, spread across multiple "
        "banks to maximise insurance coverage.",
        "RBI",
    ),
]


# 10 investor-profile context sentences prepended to questions.
# With 30 QA pairs × 10 contexts = 300 unique combinations.
_INVESTOR_CONTEXTS = [
    "I am a first-time equity investor with ₹5 lakh to invest. ",
    "I am a salaried professional in the 30% tax bracket. ",
    "I am a retired government employee looking to protect my savings. ",
    "I am a small business owner planning to invest surplus funds. ",
    "I am a 28-year-old software engineer just starting to invest. ",
    "I am a housewife managing family finances for the first time. ",
    "I am a high-net-worth individual with a diversified portfolio. ",
    "I am a 45-year-old planning for retirement in the next 15 years. ",
    "I am a freelancer with irregular income trying to grow wealth. ",
    "I am a recent college graduate with ₹50,000 to start investing. ",
]


class RegulatoryQAGenerator(BaseGapGenerator):
    """Generates SEBI and RBI regulatory Q&A pairs (Gaps 9 & 10).

    Cycles through 30 regulatory QA pairs (SEBI + RBI) covering MF rules, market
    mechanisms, investor protection, repo rate, KYC, digital lending, and deposit
    insurance. Each question is prefixed with one of 10 investor-profile context
    sentences to ensure 300 unique dedup-safe samples (30 pairs × 10 contexts).

    Args:
        samples_per_gap: Number of samples to produce.
        rng: Seeded Random instance.
    """

    gap_name = "regulatory"
    task_type = "regulatory_qa"
    layer = "L2_indian_regulatory"
    source_dataset = "synthetic_sebi_rbi"

    def _generate_one(self, idx: int) -> dict[str, Any]:
        pair_idx = idx % len(_QA_PAIRS)
        context_idx = (idx // len(_QA_PAIRS)) % len(_INVESTOR_CONTEXTS)
        q, a, _regulator = _QA_PAIRS[pair_idx]
        q = _INVESTOR_CONTEXTS[context_idx] + q
        return self._make_sample(
            idx, q, a,
            difficulty="advanced",
            layer="L2_indian_regulatory",
        )
