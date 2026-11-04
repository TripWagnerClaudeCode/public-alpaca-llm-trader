"""Ticker matching: cashtags, company names, and (carefully) bare uppercase tickers.

Precision over recall. Rules:
- Cashtags ($NVDA, $nvda) always match when the symbol is in the universe.
- Exchange-tagged tickers always match: "(NASDAQ:KIDS)", "NYSE: FIGS", "(XLF:NYSEARCA)".
  A parenthesized ticker "(LIND)", "(KIDS)" matches unless it is a common acronym ("(AI)", "(CEO)").
- Plain bare tickers match only when written in UPPERCASE, >= 3 letters, not in the stoplist, not
  after another exchange prefix ("TSX:ECO"), the text is not mostly upper case (shouting headlines),
  AND the post contains a finance context word (stock, shares, earnings, ETF, ...).
- Company names (normalized: "NVIDIA Corporation Common Stock" -> "NVIDIA") match case-insensitively
  on word boundaries, but the match must start with a capital letter ("Apple", not "apple pie").
  Multi-word names ("Home Depot") and CamelCase names ("CareDx") match on their own.
  Single-word auto-derived names ("Bandwidth", "Manitowoc", "Nvidia") need a finance context word,
  marked with a leading "~". Configured aliases are trusted unless they are everyday words
  ("Apple", "Visa", "Meta": the AMBIGUOUS_NAMES list), which also need context.
  A name equal to the ticker ("FIGS") is dropped: ticker rules apply instead.
- ETFs/funds/trusts get no name matching (tickers/cashtags only).
- A post matching more than `max_symbols` symbols is treated as a list/roundup and dropped.
"""
from __future__ import annotations

import re

# Uppercase words that are common English words or acronyms and must never count as a bare ticker.
# Many are real tickers (ALL, NOW, OPEN, RUN, HOOD, COST, ...); those still match via $cashtag or name.
STOPLIST: frozenset[str] = frozenset("""
A I AI IT ON OR IS AT BE GO SO NO UP US DO AN AS BY IF IN OF TO WE ME MY OK TV PC UK EU
ALL ANY ARE BIG BUY CAN CAR CEO CFO COO CTO CIO CPI DAY DNA EPS ETF EVS FAQ FED FOR FUN GDP GET
HAS HER HIM HIS HOW IPO KEY LOW MAN NEW NOT NOW ONE OUR OUT OWN PER PPI PRO RUN SEC SEE THE TOP
TWO USA USD WAS WAY WHO WHY YES YOU AGO AND API APP ARM ART AWS BOX CAT DOW EAT END ERA FAT FLY
FDA FTC DOJ IRS NYT NBA NFL GPU CPU RAM SSD LLM OSS PDF CEO HR IT LOL OMG WTF IMO TBH FYI ETA
ATH ATL DD YOLO HODL FOMO MOON PUMP DUMP BEAR BULL CALL PUTS PUT LONG SHORT HOLD SELL GAIN LOSS
OPEN HOOD COST LOVE GOOD REAL CASH SAFE TRUE WELL BEST FAST HOME LIFE PLAY NICE WISH BOOM GOLD
BOND RATE FUND NEXT EVER ELSE EDGE CORE PEAK RISK DEAL PAYS BEAT MISS NEWS JUST VERY MUCH MOST
MORE LESS THAN THAT THIS WITH FROM WILL HAVE BEEN WERE THEY THEM WHAT WHEN YOUR INTO OVER ONLY
ALSO SOME SUCH EACH BOTH SAME LIKE MAKE MADE TAKE CAME COME WORK PART PLAN TEAM SHOP CARE CARS
LIVE FREE FIVE FOUR TECH DATA INFO HUGE HELP BACK DOWN NOTE TELL KIDS FIRE WIND SUN AIR SKY
NASA NATO OPEC NYSE AMEX ESG ROI ROE YTD YOY QOQ MOM TTM IRA SPAC USSR CCP NHS WHO EST PST UTC
GAAP SaaS CEOS LLC INC LTD PLC CORP CO EDIT POST TLDR TLDR HTML HTTP JSON REST SQL AMA PSA
""".split())

# Acronyms that often appear in parentheses after their expansion ("artificial intelligence (AI)").
# Other stoplisted words are accepted in parentheses: "OrthoPediatrics Corp. (KIDS)".
PAREN_STOPLIST: frozenset[str] = frozenset("""
AI IT US UK EU EV EVS CEO CFO COO CTO CIO IPO ETF GDP CPI PPI FDA SEC FTC DOJ IRS EPS ESG ROI API LLM
AWS GPU CPU USA FED NYSE HR AGI ML NLP SaaS PE VC IRA ATM ADR ADS
""".split())

# Single-word company names that are also everyday words; they need a finance context word.
AMBIGUOUS_NAMES: frozenset[str] = frozenset("""
apple visa meta amazon target gap block square snap match root oracle shell chase ford general
progressive alphabet discovery dollar general compass upstart opendoor bumble lemonade carvana
coupang unity palantir snowflake affirm toast marathon booking expedia boeing caterpillar
international american united national first southern pioneer frontier liberty sunrise summit
global digital pacific atlantic energy health power gold silver star stars vision focus
signal river ocean nature delta alaska spirit hertz wayfair zillow hims monday docusign
""".split())

CONTEXT_WORDS = re.compile(
    r"\b(stocks?|shares?|shareholders?|earnings|revenue|guidance|nasdaq|nyse|investors?|analysts?|"
    r"price target|ticker|market cap|valuation|quarter(ly)?|q[1-4]|fiscal|dividend|ipo|sec filing|"
    r"acquisition|acquires|merger|traders?|rall(y|ies)|sell-?off|downgrades?|upgrades?|buyback|"
    r"short sellers?|short report|10-k|10-q|8-k|offering|etfs?|fda|outlook|results|wall street|"
    r"premarket|after-hours|options|bullish|bearish|eps)\b",
    re.I,
)
_EXCH = r"(?:NYSE|NASDAQ|NYSEARCA|NYSEAMERICAN|NYSEMKT|AMEX|BATS|CBOE|ARCA)"

_SUFFIX_WORDS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies", "holdings", "holding",
    "group", "ltd", "limited", "plc", "llc", "lp", "nv", "sa", "ag", "se", "the", "and", "&", "of",
    "de", "bancorp", "trust", "international", "intl", "class", "a", "b", "c",
}
_NOISE = re.compile(
    r"\(.*?\)|\b(common stock|common shares|ordinary shares|class [a-c]( common stock| ordinary shares)?|"
    r"american depositary (shares|receipts?)|depositary (shares|receipts?)|sponsored adr|adr|ads|"
    r"series \d+|units?|each representing.*$)\b",
    re.I,
)
_FUND = re.compile(r"\b(etf|etn|fund|trust|spdr|ishares|invesco|proshares|direxion|vanguard)\b", re.I)


_GENERIC_TAIL = {
    "markets", "platforms", "technologies", "technology", "therapeutics", "pharmaceuticals", "pharma",
    "systems", "semiconductor", "semiconductors", "software", "networks", "brands", "entertainment",
    "resources", "solutions", "industries", "labs", "laboratories", "biosciences", "bio", "biotherapeutics",
    "sciences", "financial", "communications", "interactive", "healthcare", "worldwide", "motors",
    "wholesale", "enterprises", "partners", "capital", "acquisition", "energy", "medical", "biopharma",
}


def is_fund_name(raw: str) -> bool:
    return bool(_FUND.search(raw or ""))


def normalize_company_name(raw: str) -> str:
    """'NVIDIA Corporation Common Stock' -> 'NVIDIA'; 'JPMorgan Chase & Co.' -> 'JPMorgan Chase'."""
    if not raw:
        return ""
    s = _NOISE.sub(" ", raw)
    s = re.sub(r"[,]", " ", s)
    words = [w for w in re.split(r"\s+", s.strip()) if w]
    # strip trailing / leading corporate suffixes (compare without dots: 'N.V.' -> 'nv')
    def key(w: str) -> str:
        return re.sub(r"[.\-]", "", w).lower()
    while words and key(words[-1]) in _SUFFIX_WORDS:
        words.pop()
    while words and key(words[0]) == "the":
        words.pop(0)
    name = " ".join(words).strip(" .,-&")
    return name


def _camel(w: str) -> bool:
    return bool(re.search(r"[a-z][A-Z]", w))


def name_variants(raw: str, aliases: list[str] | None = None, symbol: str = "") -> list[str]:
    """Normalized name plus obvious variants and configured aliases.

    'Amazon.com' -> also '~Amazon'; 'Robinhood Markets' -> also '~Robinhood' (generic tail dropped).
    A leading '~' marks a derived variant that only matches with a finance context word nearby.
    """
    out: list[str] = list(aliases or [])  # first: a trusted alias beats the same auto-derived name
    n = normalize_company_name(raw) if raw and not is_fund_name(raw) else ""
    if n and n.upper() == symbol.upper():
        n = ""  # "FIGS, Inc." -> name equals ticker: ticker rules apply
    if n:
        out.append(n if (" " in n or _camel(n)) else "~" + n)
        if n.lower().endswith(".com"):
            out.append("~" + n[:-4])
        words = n.split()
        if len(words) >= 2 and all(w.lower() in _GENERIC_TAIL for w in words[1:]):
            out.append("~" + words[0])
    seen, res = set(), []
    for v in out:
        v = v.strip()
        k = v.lstrip("~").lower()
        if len(v.lstrip("~")) >= 4 and k not in seen:
            seen.add(k)
            res.append(v)
    return res


class TickerMatcher:
    def __init__(
        self,
        symbols: list[str],
        names: dict[str, list[str]] | None = None,
        stoplist: set[str] | frozenset[str] = STOPLIST,
        ambiguous: set[str] | frozenset[str] = AMBIGUOUS_NAMES,
        max_symbols: int = 6,
    ):
        self.symbols = {s.upper() for s in symbols}
        self.stoplist = {w.upper() for w in stoplist}
        self.ambiguous = {w.lower() for w in ambiguous}
        self.max_symbols = max_symbols
        self._cashtag = re.compile(r"(?<![\w$])\$([A-Za-z]{1,5}(?:\.[A-Za-z])?)(?![\w])")
        self._tagged = re.compile(
            rf"(?<![\w]){_EXCH}\s*:\s*([A-Z]{{1,5}})(?![\w])|\(\s*([A-Z]{{1,5}})\s*:\s*{_EXCH}\s*\)")
        self._paren = re.compile(r"\(\s*([A-Z]{1,5})\s*\)")
        self._bare = re.compile(r"(?<![\w$./:-])([A-Z]{3,5})(?![\w/:-])")
        # name -> (regex, symbol, needs_context)
        self._names: list[tuple[re.Pattern, str, bool]] = []
        for sym, variants in (names or {}).items():
            if sym.upper() not in self.symbols:
                continue
            for v in variants:
                derived = v.startswith("~")
                v = v.lstrip("~")
                if len(v) < 4 or v.upper() in self.stoplist:
                    continue
                pat = re.compile(r"(?<![\w])" + re.escape(v).replace(r"\ ", r"\s+") + r"(?!\w)", re.I)
                needs_ctx = derived or ((" " not in v) and v.lower() in self.ambiguous)
                self._names.append((pat, sym.upper(), needs_ctx))

    def match(self, text: str) -> set[str]:
        if not text:
            return set()
        found: set[str] = set()
        for m in self._cashtag.finditer(text):
            sym = m.group(1).upper()
            if sym in self.symbols:
                found.add(sym)
        for m in self._tagged.finditer(text):
            sym = m.group(1) or m.group(2)
            if sym in self.symbols:
                found.add(sym)
        for m in self._paren.finditer(text):
            sym = m.group(1)
            if sym in self.symbols and sym not in PAREN_STOPLIST:
                found.add(sym)
        has_ctx: bool | None = None
        letters = [c for c in text if c.isalpha()]
        shouting = bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.6
        if not shouting:
            for m in self._bare.finditer(text):
                sym = m.group(1)
                if sym in self.symbols and sym not in self.stoplist and sym not in found:
                    if has_ctx is None:
                        has_ctx = bool(CONTEXT_WORDS.search(text))
                    if has_ctx:
                        found.add(sym)
        for pat, sym, needs_ctx in self._names:
            if sym in found:
                continue
            for m in pat.finditer(text):
                if not m.group(0)[0].isupper():
                    continue
                if needs_ctx:
                    if has_ctx is None:
                        has_ctx = bool(CONTEXT_WORDS.search(text))
                    if not has_ctx:
                        break
                found.add(sym)
                break
        if len(found) > self.max_symbols:
            return set()
        return found
