import datetime as dt
import html
import re
from dataclasses import dataclass

import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf


APP_TITLE = "Stock Future Visualizer"
DEFAULT_GROWTH_RATE = 0.10
MAX_BASE_GROWTH = 0.60
MIN_BASE_GROWTH = -0.50
PROJECTION_YEARS = 6

NET_INCOME_LABELS = ["Net Income", "NetIncome", "Net Income Common Stockholders"]
REVENUE_LABELS = ["Total Revenue", "TotalRevenue", "Revenue"]

PROJECTION_ASSUMPTION_TOOLTIPS: dict[str, str] = {
    "Revenue": "Starting annual revenue in billions, from Yahoo Finance TTM data. The base from which all future revenue is projected.",
    "Starting net margin": "Net income as a percentage of revenue at the start of the projection. Reflects current profitability before any assumed improvement or decline.",
    "Selected terminal margin": "The net margin the model targets by the final projection year. Higher terminal margins imply growing operational efficiency over time.",
    "Shares outstanding": "Share count in millions used to calculate EPS. Adjusted each year by the annual share count change rate.",
    "Annual share count change": "Models dilution (positive) or buybacks (negative) per year. Directly affects EPS — fewer shares amplify per-share earnings.",
    "Revenue growth": "Starting annual revenue growth rate applied to projections. May decay toward sustainable levels over time if Growth Normalization is enabled.",
    "P/E range": "Price-to-earnings multiple range applied to projected EPS to estimate a share price band. Derived from current trailing P/E, adjusted per scenario.",
    "Growth normalization": "When on, very high initial growth rates are gradually reduced toward sustainable long-term levels. Prevents unrealistic linear extrapolation of outlier growth.",
}

DATA_QUALITY_TOOLTIPS: dict[str, str] = {
    "Price": "Live market price availability. A missing price prevents all return percentage calculations.",
    "Revenue": "TTM total revenue used as the projection baseline. A placeholder value reduces projection accuracy significantly.",
    "Net income": "Trailing twelve-month net income for margin derivation. Weak data may trigger the quarterly fallback logic.",
    "Analyst target": "Wall Street 12-month consensus price target. Displayed for reference only — not used in model projections.",
    "Shares": "Share count used for EPS calculation. A 1 million share placeholder will produce inaccurate per-share figures.",
    "P/E": "Trailing price-to-earnings ratio. A defaulted value of 15 may not reflect the company's actual valuation multiple.",
    "Quarterly fallback": "Whether the model substituted a recent profitable quarter for unreliable TTM data. Common after one-time charges distort annual figures.",
    "Revenue growth": "Year-over-year revenue growth rate used to seed the projection. Pulled from Yahoo Finance or calculated from quarterly data.",
    "Earnings growth": "Year-over-year earnings growth rate used alongside revenue growth to assess trend momentum.",
    "Growth clamp": "Whether an extreme revenue growth rate was capped at 60%. Applied to prevent outlier growth from producing unrealistic projections.",
}


@dataclass
class StockInputs:
    company_name: str
    current_price: float
    market_cap: float
    shares_outstanding: float
    current_pe: float
    forward_pe: float | None
    total_revenue: float
    net_income: float | None
    trailing_eps: float | None
    target_mean: float | None
    target_low: float | None
    target_high: float | None
    ceo_name: str | None = None
    price_timestamp: str | None = None


@dataclass
class ProjectionSettings:
    scenario: str
    revenue: float
    starting_margin: float
    terminal_margin: float
    shares: float
    share_change_rate: float
    revenue_growth: float
    pe_low: float
    pe_high: float
    use_decay: bool


@dataclass
class DataQuality:
    label: str
    status: str
    detail: str


@dataclass
class EarningsMomentumSlice:
    eps_recent: float
    eps_reference: float
    eps_delta: float
    quarter_recent: str
    quarter_reference: str
    revenue_recent: float | None
    revenue_reference: float | None
    revenue_delta: float | None
    is_improving: bool


@dataclass
class EarningsMomentum:
    qoq: EarningsMomentumSlice | None   # most recent quarter vs prior quarter
    yoy: EarningsMomentumSlice | None   # most recent quarter vs same quarter last year


@dataclass
class HistoricalPeriod:
    label: str
    revenue: float | None
    eps: float | None
    revenue_growth: float | None


@dataclass
class HistoricalContext:
    periods: list
    revenue_trend: str
    eps_trend: str
    revenue_volatility: str
    revenue_insight: str
    eps_insight: str


def configure_page() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

    is_search_active = bool(st.session_state.get("ticker", ""))
    post_search_css = (
        """
        header[data-testid="stHeader"] {
            background-color: transparent !important;
            height: 1.25rem !important;
            min-height: 1.25rem !important;
        }
        .main .block-container {
            padding-top: 0 !important;
        }
        .block-container {
            padding-top: 0 !important;
        }
        [data-testid="collapsedControl"] {
            transform: translateY(0.5rem) !important;
        }
"""
        if is_search_active
        else ""
    )

    st.markdown(
        f"""
        <style>
        .stTextInput, .stButton > button {{
            position: relative !important;
            z-index: 9999 !important;
        }}
        .stTextInput > div > div > input {{
            text-align: center;
        }}
        [data-testid="stAppViewContainer"] {{
            background-color: #0a0b13;
        }}
        [data-testid="stSidebar"] {{
            background-color: #1a1b25;
            border-right: 1px solid #1f1f2e;
            will-change: transform;
            transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), width 0.2s cubic-bezier(0.4, 0, 0.2, 1), margin-left 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }}
        .stButton button {{
            border-radius: 8px;
            font-weight: 500;
        }}
        .stButton button:hover {{
            transform: scale(1.02);
        }}
        [data-testid="collapsedControl"] {{
            position: fixed !important;
            top: 2rem !important;
        }}
        [data-testid="stSidebarCollapseButton"] {{
            margin-top: -2.5rem !important;
        }}
        .sfv-refresh-col-marker {{
            display: none;
        }}
        [data-testid="stColumn"]:has(.sfv-refresh-col-marker) .stButton,
        [data-testid="column"]:has(.sfv-refresh-col-marker) .stButton {{
            margin-top: 1.6rem;
        }}
        [data-testid="stColumn"]:has(.sfv-refresh-col-marker) .stButton button,
        [data-testid="column"]:has(.sfv-refresh-col-marker) .stButton button {{
            padding: 0.15rem 0.45rem !important;
            min-height: 0 !important;
            min-width: 0 !important;
            height: auto !important;
            line-height: 1 !important;
        }}
        {post_search_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


def go_home() -> None:
    st.session_state["ticker"] = ""


def toggle_session_bool(key: str) -> None:
    st.session_state[key] = not st.session_state.get(key, False)


def first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def clamp(value: float, low: float, high: float) -> float:
    return max(min(value, high), low)


def format_market_cap(market_cap: float | None) -> str:
    if not market_cap or market_cap <= 0:
        return "Unavailable"
    if market_cap >= 1e12:
        return f"~${market_cap / 1e12:.2f}T"
    if market_cap >= 1e9:
        return f"~${market_cap / 1e9:.2f}B"
    if market_cap >= 1e6:
        return f"~${market_cap / 1e6:.2f}M"
    return f"~${market_cap:,.0f}"


def format_money(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def calculate_return_percent(predicted_price: float, reference_price: float) -> float:
    return ((predicted_price - reference_price) / reference_price) * 100


def get_return_percent(
    predicted_price: float,
    current_price: float,
    user_cost_basis: float | None = None,
) -> float:
    reference = user_cost_basis if (user_cost_basis and user_cost_basis > 0) else current_price
    return calculate_return_percent(predicted_price, reference)


def format_return_range(
    price_low: float,
    price_high: float,
    current_price: float,
    user_cost_basis: float | None = None,
) -> str:
    using_cost_basis = user_cost_basis is not None and user_cost_basis > 0
    reference = user_cost_basis if using_cost_basis else current_price
    low_pct = calculate_return_percent(price_low, reference)
    high_pct = calculate_return_percent(price_high, reference)
    suffix = " *" if using_cost_basis else ""
    return f"{low_pct:.0f}% to {high_pct:.0f}%{suffix}"


def apply_growth_decay(
    initial_growth: float,
    year_number: int,
    normal_growth_rate: float = 0.15,
    high_growth_threshold: float = 0.30,
) -> float:
    if initial_growth <= high_growth_threshold:
        return initial_growth

    if initial_growth > 2.0:
        floor_rate = 0.25
        decay_factor = 0.50 ** year_number
    elif initial_growth > 1.0:
        floor_rate = 0.22
        decay_factor = 0.60 ** year_number
    elif initial_growth > 0.50:
        floor_rate = 0.20
        decay_factor = 0.70 ** year_number
    else:
        floor_rate = normal_growth_rate
        decay_factor = 0.85 ** year_number

    excess_growth = initial_growth - floor_rate
    decayed_growth = floor_rate + (excess_growth * decay_factor)
    return max(decayed_growth, floor_rate)


def is_unreliable_for_projection(
    net_income: float | None,
    net_growth: float | None,
    current_pe: float | None,
) -> bool:
    if not net_income or net_income <= 0:
        return True
    if not current_pe or current_pe <= 0:
        return True
    return net_growth is not None and net_growth < -0.50


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_quarterly_financials(ticker: str) -> pd.DataFrame:
    try:
        return yf.Ticker(ticker).quarterly_financials
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_annual_financials(ticker: str) -> pd.DataFrame:
    try:
        return yf.Ticker(ticker).financials
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        return yf.Ticker(ticker).history(start=start, end=end).reset_index()
    except Exception:
        return pd.DataFrame()


def get_positive_annualized_quarter(
    quarterly: pd.DataFrame,
    possible_labels: list[str],
) -> tuple[float | None, str | None]:
    if quarterly is None or quarterly.empty:
        return None, None

    row_label = next((label for label in possible_labels if label in quarterly.index), None)
    if row_label is None:
        return None, None

    for column in quarterly.columns:
        value = quarterly.loc[row_label, column]
        if pd.notna(value) and value > 0:
            annualized = float(value) * 4
            quarter_label = pd.Timestamp(column).strftime("%b %Y")
            return annualized, quarter_label

    return None, None


def compute_growth_from_quarterly(
    quarterly: pd.DataFrame,
    metric_label_options: list[str],
) -> float | None:
    if quarterly is None or quarterly.empty:
        return None

    row_label = next((label for label in metric_label_options if label in quarterly.index), None)
    if row_label is None or len(quarterly.columns) < 5:
        return None

    recent = quarterly.loc[row_label, quarterly.columns[0]]
    year_ago = quarterly.loc[row_label, quarterly.columns[4]]

    if pd.isna(recent) or pd.isna(year_ago) or year_ago <= 0:
        return None

    return float((recent - year_ago) / abs(year_ago))


def _build_momentum_slice(
    quarterly: pd.DataFrame,
    ni_label: str,
    rev_label: str | None,
    shares_outstanding: float,
    col_recent,
    col_reference,
) -> EarningsMomentumSlice | None:
    ni_r = quarterly.loc[ni_label, col_recent]
    ni_ref = quarterly.loc[ni_label, col_reference]
    if pd.isna(ni_r) or pd.isna(ni_ref):
        return None
    eps_recent = float(ni_r) / shares_outstanding
    eps_reference = float(ni_ref) / shares_outstanding
    if eps_recent >= 0 or eps_reference >= 0:
        return None
    eps_delta = eps_recent - eps_reference
    revenue_recent = revenue_reference = revenue_delta = None
    if rev_label is not None:
        rev_r = quarterly.loc[rev_label, col_recent]
        rev_ref = quarterly.loc[rev_label, col_reference]
        if pd.notna(rev_r) and pd.notna(rev_ref):
            revenue_recent = float(rev_r)
            revenue_reference = float(rev_ref)
            revenue_delta = revenue_recent - revenue_reference
    return EarningsMomentumSlice(
        eps_recent=eps_recent,
        eps_reference=eps_reference,
        eps_delta=eps_delta,
        quarter_recent=pd.Timestamp(col_recent).strftime("%b %Y"),
        quarter_reference=pd.Timestamp(col_reference).strftime("%b %Y"),
        revenue_recent=revenue_recent,
        revenue_reference=revenue_reference,
        revenue_delta=revenue_delta,
        is_improving=eps_delta > 0,
    )


def compute_earnings_momentum(
    ticker: str,
    shares_outstanding: float,
) -> EarningsMomentum | None:
    if shares_outstanding <= 0:
        return None

    quarterly = fetch_quarterly_financials(ticker)
    if quarterly is None or quarterly.empty or len(quarterly.columns) < 2:
        return None

    ni_label = next((label for label in NET_INCOME_LABELS if label in quarterly.index), None)
    if ni_label is None:
        return None

    rev_label = next((label for label in REVENUE_LABELS if label in quarterly.index), None)
    cols = quarterly.columns

    qoq = _build_momentum_slice(quarterly, ni_label, rev_label, shares_outstanding, cols[0], cols[1])
    yoy = None
    if len(cols) >= 5:
        yoy = _build_momentum_slice(quarterly, ni_label, rev_label, shares_outstanding, cols[0], cols[4])

    if qoq is None and yoy is None:
        return None

    return EarningsMomentum(qoq=qoq, yoy=yoy)


def _classify_revenue_trend(growth_rates: list, period_name: str = "year") -> tuple[str, str]:
    n = len(growth_rates)
    if n < 2:
        return "Insufficient Data", "Not enough periods to assess revenue direction."
    positive = sum(1 for g in growth_rates if g > 0.01)
    negative = sum(1 for g in growth_rates if g < -0.01)
    avg = sum(growth_rates) / n
    alternating = sum(
        1 for i in range(1, n) if (growth_rates[i] > 0) != (growth_rates[i - 1] > 0)
    )
    if negative >= n * 0.70:
        return "Declining", f"Revenue declined in {negative} of {n} measured {period_name}s."
    if positive >= n * 0.75:
        return "Consistent Growth", (
            f"Revenue grew in {positive} of {n} {period_name}s, averaging {avg * 100:.1f}% per {period_name}."
        )
    if alternating >= n * 0.55:
        return "Cyclical", f"Revenue shows alternating growth and contraction across measured {period_name}s."
    if positive > negative:
        return "Generally Increasing", f"Revenue is trending upward overall, though with some {period_name}s of decline."
    return "Flat / Mixed", f"Revenue shows no clear directional trend over the measured {period_name}s."


def _classify_eps_trend(eps_values: list) -> tuple[str, str]:
    n = len(eps_values)
    if n < 2:
        return "Insufficient Data", "Not enough periods to assess EPS direction."
    all_positive = all(e >= 0 for e in eps_values)
    all_negative = all(e < 0 for e in eps_values)
    if all_positive:
        improving = sum(1 for i in range(1, n) if eps_values[i] > eps_values[i - 1])
        declining = sum(1 for i in range(1, n) if eps_values[i] < eps_values[i - 1])
        if improving >= (n - 1) * 0.65:
            return "Profitable Growth", "EPS has grown consistently across measured periods, indicating improving earnings power."
        if declining >= (n - 1) * 0.65:
            return "Declining Profitability", "EPS has been declining despite remaining profitable."
        return "Stable Profitability", "EPS has remained relatively stable in positive territory."
    if all_negative:
        first, last = eps_values[0], eps_values[-1]
        threshold = abs(first) * 0.05
        if last > first + threshold:
            return "Improving Losses", "EPS losses are narrowing over time, indicating improving operational efficiency."
        if last < first - threshold:
            return "Expanding Losses", "EPS losses are widening, indicating deteriorating earnings."
        return "Stable Losses", "EPS losses have remained relatively flat across measured periods."
    half = n // 2
    recent_avg = sum(eps_values[half:]) / len(eps_values[half:])
    early_avg = sum(eps_values[:half]) / len(eps_values[:half]) if half > 0 else recent_avg
    if recent_avg > early_avg + 0.01:
        return "Improving", "EPS has been trending upward and shows signs of improving."
    return "Mixed / Volatile", "EPS fluctuates between profitable and loss periods with no clear sustained trend."


def _compute_revenue_volatility(growth_rates: list) -> str:
    if len(growth_rates) < 3:
        return "N/A"
    s = float(pd.Series(growth_rates).std())
    if s > 0.20:
        return "High"
    if s > 0.08:
        return "Medium"
    return "Low"


def extract_historical_context(ticker: str, shares_outstanding: float) -> HistoricalContext | None:
    annual = fetch_annual_financials(ticker)
    if annual is None or annual.empty:
        return None
    rev_label = next((lbl for lbl in REVENUE_LABELS if lbl in annual.index), None)
    ni_label = next((lbl for lbl in NET_INCOME_LABELS if lbl in annual.index), None)
    if rev_label is None and ni_label is None:
        return None

    cols = list(annual.columns[:8])
    cols.reverse()

    raw_periods = []
    for col in cols:
        year_label = pd.Timestamp(col).strftime("%Y")
        revenue = None
        eps = None
        if rev_label is not None:
            val = annual.loc[rev_label, col]
            if pd.notna(val):
                revenue = float(val)
        if ni_label is not None and shares_outstanding > 0:
            val = annual.loc[ni_label, col]
            if pd.notna(val):
                eps = float(val) / shares_outstanding
        raw_periods.append(HistoricalPeriod(label=year_label, revenue=revenue, eps=eps, revenue_growth=None))

    for i in range(1, len(raw_periods)):
        prev_rev = raw_periods[i - 1].revenue
        curr_rev = raw_periods[i].revenue
        growth = None
        if prev_rev is not None and curr_rev is not None and prev_rev != 0:
            growth = (curr_rev - prev_rev) / abs(prev_rev)
        raw_periods[i] = HistoricalPeriod(
            label=raw_periods[i].label,
            revenue=raw_periods[i].revenue,
            eps=raw_periods[i].eps,
            revenue_growth=growth,
        )

    periods = [p for p in raw_periods if p.revenue is not None or p.eps is not None]
    if len(periods) < 2:
        return None

    growth_rates = [p.revenue_growth for p in raw_periods if p.revenue_growth is not None]
    eps_values = [p.eps for p in periods if p.eps is not None]
    revenue_trend, revenue_insight = _classify_revenue_trend(growth_rates)
    eps_trend, eps_insight = _classify_eps_trend(eps_values)
    revenue_volatility = _compute_revenue_volatility(growth_rates)

    return HistoricalContext(
        periods=periods,
        revenue_trend=revenue_trend,
        eps_trend=eps_trend,
        revenue_volatility=revenue_volatility,
        revenue_insight=revenue_insight,
        eps_insight=eps_insight,
    )


def extract_quarterly_context(ticker: str, shares_outstanding: float) -> HistoricalContext | None:
    quarterly = fetch_quarterly_financials(ticker)
    if quarterly is None or quarterly.empty:
        return None
    rev_label = next((lbl for lbl in REVENUE_LABELS if lbl in quarterly.index), None)
    ni_label = next((lbl for lbl in NET_INCOME_LABELS if lbl in quarterly.index), None)
    if rev_label is None and ni_label is None:
        return None

    cols = sorted(quarterly.columns)[-12:]

    raw_periods = []
    for col in cols:
        q_label = f"Q{pd.Timestamp(col).quarter} {pd.Timestamp(col).year}"
        revenue = None
        eps = None
        if rev_label is not None:
            val = quarterly.loc[rev_label, col]
            if pd.notna(val):
                revenue = float(val)
        if ni_label is not None and shares_outstanding > 0:
            val = quarterly.loc[ni_label, col]
            if pd.notna(val):
                eps = float(val) / shares_outstanding
        raw_periods.append(HistoricalPeriod(label=q_label, revenue=revenue, eps=eps, revenue_growth=None))

    for i in range(1, len(raw_periods)):
        prev_rev = raw_periods[i - 1].revenue
        curr_rev = raw_periods[i].revenue
        growth = None
        if prev_rev is not None and curr_rev is not None and prev_rev != 0:
            growth = (curr_rev - prev_rev) / abs(prev_rev)
        raw_periods[i] = HistoricalPeriod(
            label=raw_periods[i].label,
            revenue=raw_periods[i].revenue,
            eps=raw_periods[i].eps,
            revenue_growth=growth,
        )

    periods = [p for p in raw_periods if p.revenue is not None or p.eps is not None]
    if len(periods) < 2:
        return None

    growth_rates = [p.revenue_growth for p in raw_periods if p.revenue_growth is not None]
    eps_values = [p.eps for p in periods if p.eps is not None]
    revenue_trend, revenue_insight = _classify_revenue_trend(growth_rates, period_name="quarter")
    eps_trend, eps_insight = _classify_eps_trend(eps_values)
    revenue_volatility = _compute_revenue_volatility(growth_rates)

    return HistoricalContext(
        periods=periods,
        revenue_trend=revenue_trend,
        eps_trend=eps_trend,
        revenue_volatility=revenue_volatility,
        revenue_insight=revenue_insight,
        eps_insight=eps_insight,
    )


def get_price_timestamp(info: dict) -> str | None:
    ts = info.get("regularMarketTime")
    if not ts:
        return None
    try:
        t = dt.datetime.fromtimestamp(int(ts))
        hour = t.hour % 12 or 12
        period = "AM" if t.hour < 12 else "PM"
        return f"{hour}:{t.minute:02d} {period}"
    except Exception:
        return None


_CEO_PREFIX_RE = re.compile(r"^(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?|Sir|Lord)\s+", re.IGNORECASE)
_CEO_CREDENTIAL_RE = re.compile(
    r"\s+(?:Ph\.?D\.?|M\.?B\.?A\.?|M\.?D\.?|J\.?D\.?|CPA|CFA|CFP|Esq\.?|Jr\.?|Sr\.?|II|III|IV)\s*$",
    re.IGNORECASE,
)


def extract_ceo_name(info: dict) -> str | None:
    for officer in info.get("companyOfficers", []):
        title = (officer.get("title") or "").lower()
        if "chief executive" in title or "ceo" in title:
            name = (officer.get("name") or "").strip()
            name = name.split(",")[0].strip()
            name = _CEO_PREFIX_RE.sub("", name).strip()
            name = _CEO_CREDENTIAL_RE.sub("", name).strip()
            return name if name else None
    return None


def build_stock_inputs(info: dict) -> tuple[StockInputs, list[str], list[DataQuality]]:
    warnings = []
    quality = []

    current_price = first_not_none(info.get("currentPrice"), info.get("regularMarketPrice"), 0.0)
    market_cap = first_not_none(info.get("marketCap"), 0.0)
    shares_outstanding = first_not_none(info.get("sharesOutstanding"), 1_000_000)
    current_pe = first_not_none(info.get("trailingPE"), 15.0)
    total_revenue = first_not_none(info.get("totalRevenue"), 100_000_000.0)

    quality.append(
        DataQuality(
            "Price",
            "Good" if current_price else "Missing",
            "Live market price found." if current_price else "No live price returned.",
        )
    )
    quality.append(
        DataQuality(
            "Revenue",
            "Good" if info.get("totalRevenue") else "Defaulted",
            "Yahoo Finance revenue found." if info.get("totalRevenue") else "Using a $100 million placeholder.",
        )
    )
    quality.append(
        DataQuality(
            "Net income",
            "Good" if info.get("netIncomeToCommon") else "Weak",
            "TTM net income found." if info.get("netIncomeToCommon") else "May need quarterly fallback.",
        )
    )
    quality.append(
        DataQuality(
            "Analyst target",
            "Good" if info.get("targetMeanPrice") else "Missing",
            "Consensus target found." if info.get("targetMeanPrice") else "No target returned.",
        )
    )

    if info.get("sharesOutstanding") is None:
        warnings.append("Shares outstanding was unavailable, so the app used a 1 million share placeholder.")
        quality.append(DataQuality("Shares", "Defaulted", "Using a 1 million share placeholder."))
    else:
        quality.append(DataQuality("Shares", "Good", "Shares outstanding found."))

    if info.get("trailingPE") is None:
        warnings.append("Trailing P/E was unavailable, so the app used a default P/E of 15.")
        quality.append(DataQuality("P/E", "Defaulted", "Using a default trailing P/E of 15."))
    else:
        quality.append(DataQuality("P/E", "Good", "Trailing P/E found."))

    if info.get("totalRevenue") is None:
        warnings.append("Revenue was unavailable, so the app used a $100 million placeholder.")

    return (
        StockInputs(
            company_name=info.get("longName") or info.get("shortName", "N/A"),
            current_price=float(current_price or 0.0),
            market_cap=float(market_cap or 0.0),
            shares_outstanding=float(shares_outstanding or 1_000_000),
            current_pe=float(current_pe or 15.0),
            forward_pe=info.get("forwardPE"),
            total_revenue=float(total_revenue or 100_000_000.0),
            net_income=info.get("netIncomeToCommon"),
            trailing_eps=info.get("trailingEps"),
            target_mean=info.get("targetMeanPrice"),
            target_low=info.get("targetLowPrice"),
            target_high=info.get("targetHighPrice"),
            ceo_name=extract_ceo_name(info),
            price_timestamp=get_price_timestamp(info),
        ),
        warnings,
        quality,
    )


def apply_quarterly_fallback(
    ticker: str,
    inputs: StockInputs,
    info: dict,
    quality: list[DataQuality],
) -> tuple[StockInputs, dict, str | None, list[DataQuality]]:
    if not is_unreliable_for_projection(inputs.net_income, info.get("earningsGrowth"), inputs.current_pe):
        quality.append(DataQuality("Quarterly fallback", "Not needed", "TTM data looked usable."))
        return inputs, info, None, quality

    quarterly = fetch_quarterly_financials(ticker)
    fallback_income, income_quarter = get_positive_annualized_quarter(quarterly, NET_INCOME_LABELS)
    fallback_revenue, _ = get_positive_annualized_quarter(quarterly, REVENUE_LABELS)

    if fallback_income is None:
        quality.append(DataQuality("Quarterly fallback", "Missing", "No clean profitable quarter found."))
        return inputs, info, None, quality

    updated_inputs = StockInputs(
        company_name=inputs.company_name,
        current_price=inputs.current_price,
        market_cap=inputs.market_cap,
        shares_outstanding=inputs.shares_outstanding,
        current_pe=inputs.current_pe,
        forward_pe=inputs.forward_pe,
        total_revenue=fallback_revenue or inputs.total_revenue,
        net_income=fallback_income,
        trailing_eps=inputs.trailing_eps,
        target_mean=inputs.target_mean,
        target_low=inputs.target_low,
        target_high=inputs.target_high,
        ceo_name=inputs.ceo_name,
        price_timestamp=inputs.price_timestamp,
    )

    updated_info = dict(info)
    quarterly_net_growth = compute_growth_from_quarterly(quarterly, NET_INCOME_LABELS)
    quarterly_rev_growth = compute_growth_from_quarterly(quarterly, REVENUE_LABELS)

    if quarterly_net_growth is not None:
        updated_info["_fallback_net_growth"] = quarterly_net_growth
    if quarterly_rev_growth is not None:
        updated_info["_fallback_rev_growth"] = quarterly_rev_growth

    quality.append(DataQuality("Quarterly fallback", "Used", f"Using {income_quarter} annualized."))
    note = (
        "Most recent TTM data appears unreliable, possibly because of a one-time charge or loss. "
        f"Using the last clean quarter instead: {income_quarter} annualized."
    )
    return updated_inputs, updated_info, note, quality


def render_header() -> str:
    is_search_active = bool(st.session_state.get("ticker", ""))

    if is_search_active:
        st.markdown(
            """
            <style>
            button[kind="primary"],
            [data-testid="baseButton-primary"] {
                background: rgba(255, 255, 255, 0.12) !important;
                border: none !important;
                border-radius: 6px !important;
                padding: 0.2rem 1rem !important;
                color: #ffffff !important;
            }
            button[kind="primary"] p,
            [data-testid="baseButton-primary"] p {
                font-size: 1.05em !important;
                font-weight: 700 !important;
                color: #ffffff !important;
                letter-spacing: 0.02em !important;
            }
            button[kind="primary"]:hover,
            [data-testid="baseButton-primary"]:hover {
                background: rgba(255, 255, 255, 0.2) !important;
                border: none !important;
                transform: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        _, col_title, _ = st.columns([2.75, 1.5, 2.75])
        with col_title:
            st.button(APP_TITLE, on_click=go_home, key="home_title_btn", type="primary", use_container_width=True)
        _, center, _ = st.columns([1.5, 2, 1.5])
        with center:
            return st.text_input(
                "",
                placeholder="Enter a stock ticker (e.g. AAPL, NVDA)",
                key="ticker",
            ).strip().upper()

    st.markdown(
        f"""
        <div style="text-align: center; padding-top: 20px;">
            <h1 style="font-size: 2.5em;">{APP_TITLE}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([1.5, 2, 1.5])
    with center:
        return st.text_input(
            "",
            placeholder="Enter a stock ticker (e.g. AAPL, NVDA)",
            key="ticker",
        ).strip().upper()


def render_home_screen() -> None:
    st.markdown(
        """
        <div style="text-align: center; transform: translateY(-50px); padding: 50px 20px;">
            <h1 style="font-size: 3em;">Welcome to <span style="color:#00bfff;">Stock Future Visualizer</span></h1>
            <p style="font-size: 1.2em; color: gray; max-width: 600px; margin: auto;">
                Explore future stock price projections based on real financial data.
            </p>
            <br>
            <img src="https://images.unsplash.com/photo-1565723858624-12d3614748e8?ixlib=rb-4.1.0&auto=format&fit=crop&q=80&w=687" width="200" style="opacity:0.85;"/>
            <img src="https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?ixlib=rb-4.1.0&auto=format&fit=crop&q=80&w=1170" width="442" style="opacity:0.85;"/>
            <img src="https://images.unsplash.com/photo-1506787497326-c2736dde1bef?ixlib=rb-4.1.0&auto=format&fit=crop&q=80&w=692" width="200" style="opacity:0.85;"/>
            <br><br>
            <p style="color: #666; font-size: 0.8em;">Powered by Yahoo Finance + Streamlit</p>
            <p style="color: #666; font-size: 0.8em;">Programmed by Evan Kulesza</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analyst_targets(inputs: StockInputs) -> None:
    st.markdown("<br>", unsafe_allow_html=True)

    if inputs.target_mean and inputs.current_price > 0:
        low_high = ""
        if inputs.target_low is not None and inputs.target_high is not None:
            low_high = f"(Low: {format_money(inputs.target_low)} - High: {format_money(inputs.target_high)})"

        target_return = (inputs.target_mean / inputs.current_price - 1) * 100
        st.markdown(
            f"""
            <div style="text-align: center; font-family: Helvetica, sans-serif;">
                <h3 style="color: #00bfff; font-weight: 700;">1 Year Analyst Price Target</h3>
                <p style="font-size: 1.1em; color: white;">
                    <b>Consensus:</b> {format_money(inputs.target_mean)}
                    <span style="color: gray;">{low_high}</span><br>
                    <span style="font-size: 0.95em; color: #a8a8a8;">
                        {target_return:.1f}% versus current price ({format_money(inputs.current_price)})
                    </span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <div style="text-align: center; font-family: Helvetica, sans-serif;">
            <h3 style="color: #00bfff; font-weight: 700;">1 Year Analyst Price Target</h3>
            <p style="font-size: 1.05em; color: gray;">No analyst consensus data available for this ticker.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_company_summary(
    ticker: str,
    inputs: StockInputs,
    fallback_note: str | None,
    data_warnings: list[str],
) -> None:
    render_analyst_targets(inputs)

    st.markdown(
        f"""
        <div style="margin: 0.25rem 0 1rem;">
            <div style="color: #a8a8a8; font-size: 0.9rem;">Company</div>
            <div style="font-size: 1.45rem; font-weight: 700; line-height: 1.25; overflow-wrap: anywhere;">
                {inputs.company_name}
            </div>
            <div style="color: #00bfff; font-size: 1rem; font-weight: 700; margin-top: 0.2rem;">
                {ticker.upper()}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_price_area, col_mktcap, col_ceo = st.columns(3)
    ts_label = f" ({inputs.price_timestamp})" if inputs.price_timestamp else ""
    with col_price_area:
        sub_price, sub_refresh, sub_empty = st.columns([1.2, 0.45, 1.35])
        sub_price.metric(f"Current Price{ts_label}", format_money(inputs.current_price))
        sub_refresh.markdown('<div class="sfv-refresh-col-marker"></div>', unsafe_allow_html=True)
        if sub_refresh.button("↻", key="refresh_data_btn"):
            fetch_stock_info.clear()
            fetch_quarterly_financials.clear()
            fetch_annual_financials.clear()
            fetch_history.clear()
            st.rerun()
    col_mktcap.metric("Market Cap", format_market_cap(inputs.market_cap))
    col_ceo.metric("CEO", inputs.ceo_name or "N/A")
    st.markdown(
        """
        <div style="height: 1px; background: #2c2f3f; margin: 0.2rem 0 1.1rem;"></div>
        """,
        unsafe_allow_html=True,
    )

    if fallback_note:
        st.warning(fallback_note)
    for warning in data_warnings:
        st.warning(warning)


def _format_quarterly_revenue(value: float) -> str:
    abs_val = abs(value)
    if abs_val >= 1e9:
        return f"${abs_val / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"${abs_val / 1e6:.2f}M"
    return f"${abs_val:,.0f}"


def _format_eps(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):.2f}"


def _momentum_slice_html(slice: EarningsMomentumSlice, label: str) -> str:
    color = "#2ca02c" if slice.is_improving else "#d95f02"
    trajectory = "Improving" if slice.is_improving else "Worsening"
    delta_sign = "+" if slice.eps_delta >= 0 else ""
    arrow = "&#9650;" if slice.is_improving else "&#9660;"
    rev_html = ""
    if slice.revenue_recent is not None and slice.revenue_reference is not None and slice.revenue_delta is not None:
        rc = "#2ca02c" if slice.revenue_delta >= 0 else "#d95f02"
        rs = "+" if slice.revenue_delta >= 0 else "-"
        rev_html = (
            f'<span style="color:#a8a8a8;font-size:0.8rem;margin-left:1rem;">'
            f'Rev: <span style="color:#fff;">{_format_quarterly_revenue(slice.revenue_reference)}</span>'
            f'<span style="color:{rc};margin:0 0.25rem;">&#8594;</span>'
            f'<span style="color:#fff;">{_format_quarterly_revenue(slice.revenue_recent)}</span>'
            f'<span style="color:{rc};margin-left:0.3rem;">({rs}{_format_quarterly_revenue(abs(slice.revenue_delta))})</span>'
            f'</span>'
        )
    return (
        f'<div style="display:flex;align-items:center;gap:1.2rem;flex-wrap:wrap;'
        f'padding:0.65rem 0;border-top:1px solid #2c2f3f;">'
        f'<span style="font-size:0.74rem;color:#888;background:#1e2030;'
        f'padding:0.15rem 0.5rem;border-radius:4px;border:1px solid #2c2f3f;white-space:nowrap;">{label}</span>'
        f'<div>'
        f'<div style="color:#a8a8a8;font-size:0.75rem;">{slice.quarter_reference}</div>'
        f'<div style="font-size:1.1rem;font-weight:800;color:#fff;">{_format_eps(slice.eps_reference)} EPS</div>'
        f'</div>'
        f'<div style="font-size:1.4rem;color:{color};">{arrow}</div>'
        f'<div>'
        f'<div style="color:#a8a8a8;font-size:0.75rem;">{slice.quarter_recent}</div>'
        f'<div style="font-size:1.1rem;font-weight:800;color:#fff;">{_format_eps(slice.eps_recent)} EPS</div>'
        f'</div>'
        f'<div style="padding-left:0.9rem;border-left:1px solid #2c2f3f;">'
        f'<div style="font-size:0.95rem;font-weight:800;color:{color};">{delta_sign}{slice.eps_delta:.2f} EPS</div>'
        f'<div style="font-size:0.8rem;color:{color};font-weight:600;">{trajectory}</div>'
        f'</div>'
        f'{rev_html}'
        f'</div>'
    )


def render_earnings_momentum_panel(momentum: EarningsMomentum) -> None:
    rows_html = ""
    if momentum.qoq is not None:
        rows_html += _momentum_slice_html(momentum.qoq, "QoQ")
    if momentum.yoy is not None:
        rows_html += _momentum_slice_html(momentum.yoy, "YoY")

    html = (
        f'<div style="background:#11131d;border:1px solid #2c2f3f;border-radius:8px;'
        f'padding:1.25rem 1.5rem;margin:1rem 0;font-family:Helvetica,sans-serif;">'
        f'<div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:0.25rem;">'
        f'<span style="font-size:1.05rem;font-weight:800;color:#00bfff;">Earnings Momentum</span>'
        f'<span style="font-size:0.74rem;color:#888;background:#1e2030;'
        f'padding:0.18rem 0.55rem;border-radius:4px;border:1px solid #2c2f3f;">'
        f'Unprofitable Company</span>'
        f'</div>'
        f'{rows_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def get_growth_assumptions(info: dict) -> tuple[float, float, list[DataQuality], str | None]:
    quality = []
    raw_rev_growth = first_not_none(info.get("_fallback_rev_growth"), info.get("revenueGrowth"))
    raw_net_growth = first_not_none(
        info.get("_fallback_net_growth"),
        info.get("earningsGrowth"),
        info.get("earningsQuarterlyGrowth"),
    )

    if raw_rev_growth is None:
        st.warning("Revenue growth data unavailable, using a 10% default.")
        raw_rev_growth = DEFAULT_GROWTH_RATE
        quality.append(DataQuality("Revenue growth", "Defaulted", "Using 10%."))
    else:
        quality.append(DataQuality("Revenue growth", "Good", f"Raw value: {format_percent(float(raw_rev_growth))}."))

    if raw_net_growth is None:
        quality.append(DataQuality("Earnings growth", "Defaulted", "Using 10%."))
    else:
        quality.append(DataQuality("Earnings growth", "Good", f"Raw value: {format_percent(float(raw_net_growth))}."))

    rev_growth_base = clamp(float(raw_rev_growth), MIN_BASE_GROWTH, MAX_BASE_GROWTH)

    clamp_note = None
    if raw_rev_growth != rev_growth_base:
        clamp_note = f"Revenue growth clamped from {raw_rev_growth * 100:.1f}% to {rev_growth_base * 100:.1f}%."
        quality.append(DataQuality("Growth clamp", "Used", "Revenue growth was outside the allowed range."))

    return rev_growth_base, raw_net_growth if raw_net_growth is not None else DEFAULT_GROWTH_RATE, quality, clamp_note


def scenario_defaults(
    scenario: str,
    current_pe: float,
    base_revenue_growth: float,
    starting_margin: float,
) -> tuple[float, float, float, float, float]:
    if scenario == "Bear":
        return (
            base_revenue_growth / 2,
            max(starting_margin - 0.02, 0.01),
            0.01,
            max(current_pe - 15, 5),
            max(current_pe - 5, 8),
        )
    if scenario == "Bull":
        return (
            min(base_revenue_growth, 0.50),
            min(starting_margin + 0.03, 0.60),
            -0.01,
            max(current_pe + 5, 0),
            max(current_pe + 15, 0),
        )
    return (
        min(base_revenue_growth, 0.50),
        starting_margin,
        0.0,
        max(current_pe - 5, 0),
        max(current_pe + 5, 0),
    )


def render_sidebar_inputs(
    inputs: StockInputs,
    info: dict,
) -> tuple[ProjectionSettings, list[DataQuality], bool, list[str], float | None, str | None]:
    st.sidebar.header("Projection Inputs")
    view_mode = st.sidebar.radio("View Mode", ["Single Scenario", "Compare Bear/Base/Bull"], horizontal=False)
    scenario = st.sidebar.selectbox("Select Scenario", ["Bear", "Base", "Bull", "Custom"], index=1)
    use_decay = st.sidebar.checkbox(
        "Apply Growth Normalization",
        value=True,
        help="Automatically reduces abnormally high growth rates over time to more sustainable levels.",
    )
    st.sidebar.markdown("---")

    revenue = inputs.total_revenue / 1e9
    net_income = float(inputs.net_income or 0.0) / 1e9
    shares = inputs.shares_outstanding / 1e6
    starting_margin = clamp(net_income / revenue, 0.001, 0.80)

    st.sidebar.subheader("Your Position")
    cost_basis_raw = st.sidebar.number_input(
        "Average Cost Basis ($)",
        min_value=0.0,
        value=None,
        step=0.01,
        format="%.2f",
        placeholder="Optional",
        help="Enter your average purchase price to see returns relative to your cost basis.",
    )
    cost_basis = float(cost_basis_raw) if (cost_basis_raw is not None and cost_basis_raw > 0) else None
    st.sidebar.subheader("Operating Metrics")
    revenue = st.sidebar.number_input("Revenue (Billion $)", min_value=0.01, value=float(revenue), step=0.1)
    starting_margin = st.sidebar.number_input(
        "Starting Net Margin %",
        min_value=0.1,
        max_value=80.0,
        value=float(starting_margin * 100),
        step=0.1,
    ) / 100
    shares = st.sidebar.number_input("Shares Outstanding (Millions)", min_value=0.01, value=float(shares), step=1.0)
    st.sidebar.text_input("Current P/E", value=f"{inputs.current_pe:.2f}", disabled=True)
    st.sidebar.markdown("---")

    rev_growth_base, _, growth_quality, clamp_note = get_growth_assumptions(info)

    projection_notes = []
    if use_decay and rev_growth_base > 0.40:
        projection_notes.append(
            f"High revenue growth detected ({rev_growth_base * 100:.1f}%). Growth normalization is active."
        )

    default_revenue_growth, default_terminal_margin, default_share_change, default_pe_low, default_pe_high = scenario_defaults(
        "Base" if scenario == "Custom" else scenario,
        inputs.current_pe,
        rev_growth_base,
        starting_margin,
    )

    if scenario == "Custom":
        revenue_growth = st.sidebar.number_input(
            "Custom Revenue Growth %",
            min_value=-50.0,
            max_value=100.0,
            value=float(default_revenue_growth * 100),
            step=0.1,
        ) / 100
        terminal_margin = st.sidebar.number_input(
            "Terminal Net Margin %",
            min_value=0.1,
            max_value=80.0,
            value=float(default_terminal_margin * 100),
            step=0.1,
        ) / 100
        share_change_rate = st.sidebar.number_input(
            "Annual Share Count Change %",
            min_value=-20.0,
            max_value=50.0,
            value=float(default_share_change * 100),
            step=0.1,
            help="Negative values model buybacks. Positive values model dilution.",
        ) / 100
        pe_low = st.sidebar.number_input("Custom P/E Low", min_value=0.0, value=float(default_pe_low), step=0.1)
        pe_high = st.sidebar.number_input("Custom P/E High", min_value=0.0, value=float(default_pe_high), step=0.1)
    else:
        revenue_growth = default_revenue_growth
        terminal_margin = default_terminal_margin
        share_change_rate = default_share_change
        pe_low = default_pe_low
        pe_high = default_pe_high
        st.sidebar.subheader("Scenario assumptions")
        st.sidebar.text_input("Revenue Growth %", value=f"{revenue_growth * 100:.1f}", disabled=True)
        st.sidebar.text_input("Terminal Net Margin %", value=f"{terminal_margin * 100:.1f}", disabled=True)
        st.sidebar.text_input("Annual Share Change %", value=f"{share_change_rate * 100:.1f}", disabled=True)
        st.sidebar.text_input("P/E Range", value=f"{pe_low:.1f} - {pe_high:.1f}", disabled=True)

    if pe_low > pe_high:
        st.warning("P/E low was higher than P/E high, so the values were swapped.")
        pe_low, pe_high = pe_high, pe_low

    settings = ProjectionSettings(
        scenario=scenario,
        revenue=revenue,
        starting_margin=starting_margin,
        terminal_margin=terminal_margin,
        shares=shares,
        share_change_rate=share_change_rate,
        revenue_growth=revenue_growth,
        pe_low=pe_low,
        pe_high=pe_high,
        use_decay=use_decay,
    )
    compare_mode = view_mode == "Compare Bear/Base/Bull"
    return settings, growth_quality, compare_mode, projection_notes, cost_basis, clamp_note


def build_projection_dataframe(
    current_price: float,
    settings: ProjectionSettings,
) -> pd.DataFrame:
    current_year = dt.date.today().year
    years = list(range(current_year, current_year + PROJECTION_YEARS))
    projected_revenue = settings.revenue
    projected_shares = settings.shares
    data = []

    for idx, year in enumerate(years):
        year_number = idx + 1
        revenue_growth = (
            apply_growth_decay(settings.revenue_growth, year_number)
            if settings.use_decay
            else settings.revenue_growth
        )
        margin_progress = year_number / PROJECTION_YEARS
        net_margin = settings.starting_margin + (
            settings.terminal_margin - settings.starting_margin
        ) * margin_progress

        projected_revenue *= 1 + revenue_growth
        projected_shares *= 1 + settings.share_change_rate
        projected_net_income = projected_revenue * net_margin

        eps = (projected_net_income * 1e9) / (projected_shares * 1e6)
        price_low = eps * settings.pe_low
        price_high = eps * settings.pe_high
        price_avg = (price_low + price_high) / 2
        revenue_growth_pct = ((projected_revenue - settings.revenue) / settings.revenue) * 100

        data.append(
            {
                "Scenario": settings.scenario,
                "Year": year,
                "Rev Growth Rate %": round(revenue_growth * 100, 2),
                "Net Margin %": round(net_margin * 100, 2),
                "Shares (M)": round(projected_shares, 2),
                "Revenue (B)": round(projected_revenue, 2),
                "Net Income (B)": round(projected_net_income, 2),
                "Cumulative Rev Growth %": round(revenue_growth_pct, 2),
                "EPS ($)": round(eps, 2),
                "Share Price Low ($)": round(price_low, 2),
                "Share Price Average ($)": round(price_avg, 2),
                "Share Price High ($)": round(price_high, 2),
            }
        )

    df = pd.DataFrame(data)
    df["% Growth Range"] = df.apply(
        lambda row: (
            f"{((row['Share Price Low ($)'] - current_price) / current_price * 100):.0f}% to "
            f"{((row['Share Price High ($)'] - current_price) / current_price * 100):.0f}%"
        ),
        axis=1,
    )
    return df


def build_scenario_settings(settings: ProjectionSettings, inputs: StockInputs) -> list[ProjectionSettings]:
    scenario_settings = []
    for scenario in ["Bear", "Base", "Bull"]:
        revenue_growth, terminal_margin, share_change, pe_low, pe_high = scenario_defaults(
            scenario,
            inputs.current_pe,
            settings.revenue_growth,
            settings.starting_margin,
        )
        scenario_settings.append(
            ProjectionSettings(
                scenario=scenario,
                revenue=settings.revenue,
                starting_margin=settings.starting_margin,
                terminal_margin=terminal_margin,
                shares=settings.shares,
                share_change_rate=share_change,
                revenue_growth=revenue_growth,
                pe_low=pe_low,
                pe_high=pe_high,
                use_decay=settings.use_decay,
            )
        )
    return scenario_settings


def render_assumptions_panel(settings: ProjectionSettings, compare_mode: bool) -> None:
    with st.expander("Projection Assumptions", expanded=False):
        scope = "Bear, Base, and Bull scenarios" if compare_mode else settings.scenario
        st.write(
            f"{scope} use revenue growth, net margin, share count change, and P/E multiples to estimate future EPS and price ranges."
        )

        assumptions = pd.DataFrame(
            [
                ("Revenue", f"${settings.revenue:.2f}B"),
                ("Starting net margin", f"{settings.starting_margin * 100:.1f}%"),
                ("Selected terminal margin", f"{settings.terminal_margin * 100:.1f}%"),
                ("Shares outstanding", f"{settings.shares:.2f}M"),
                ("Annual share count change", f"{settings.share_change_rate * 100:.1f}%"),
                ("Revenue growth", f"{settings.revenue_growth * 100:.1f}%"),
                ("P/E range", f"{settings.pe_low:.1f} - {settings.pe_high:.1f}"),
                ("Growth normalization", "On" if settings.use_decay else "Off"),
            ],
            columns=["Input", "Value"],
        )
        render_centered_table(assumptions, width_ratio=0.72, data_font_size="1.15rem", tooltips=PROJECTION_ASSUMPTION_TOOLTIPS)


def render_data_quality_panel(quality: list[DataQuality]) -> None:
    with st.expander("Data Quality", expanded=False):
        quality_df = pd.DataFrame([item.__dict__ for item in quality])
        quality_df = quality_df.rename(
            columns={
                "label": "Label",
                "status": "Status",
                "detail": "Detail",
            }
        )
        render_centered_table(quality_df, width_ratio=0.9, data_font_size="1rem", header_font_size="0.9rem", tooltips=DATA_QUALITY_TOOLTIPS)


def render_valuation_context(inputs: StockInputs, ticker: str) -> None:
    annual_financials = fetch_annual_financials(ticker)
    hist_start_year = dt.date.today().year - 6
    history = fetch_history(ticker, f"{hist_start_year}-01-01", f"{dt.date.today().year}-12-31")
    historical_pe = compute_historical_pe(annual_financials, history, inputs.shares_outstanding)

    st.subheader("Valuation Context")
    col1, col2, col3 = st.columns(3)
    col1.metric("Trailing P/E", f"{inputs.current_pe:.2f}")
    col2.metric("Forward P/E", f"{inputs.forward_pe:.2f}" if inputs.forward_pe else "N/A")
    col3.metric("Rough Historical Median P/E", f"{historical_pe:.2f}" if historical_pe else "N/A")

    if historical_pe:
        st.caption(
            "Historical median P/E is a rough estimate using annual net income, current share count, and year-end prices."
        )

    st.markdown(
        """
        <div style="height: 1px; background: #2c2f3f; margin: 1rem 0 1.1rem;"></div>
        """,
        unsafe_allow_html=True,
    )


def compute_historical_pe(
    annual_financials: pd.DataFrame,
    history: pd.DataFrame,
    shares_outstanding: float,
) -> float | None:
    if annual_financials is None or annual_financials.empty or history is None or history.empty:
        return None

    row_label = next((label for label in NET_INCOME_LABELS if label in annual_financials.index), None)
    if row_label is None or shares_outstanding <= 0:
        return None

    history = history.copy()
    history["Date"] = pd.to_datetime(history["Date"]).dt.tz_localize(None)
    pe_values = []

    for column in annual_financials.columns[:6]:
        net_income = annual_financials.loc[row_label, column]
        if pd.isna(net_income) or net_income <= 0:
            continue

        year = pd.Timestamp(column).year
        year_prices = history[history["Date"].dt.year == year]
        if year_prices.empty:
            continue

        year_end_price = year_prices.sort_values("Date").iloc[-1]["Close"]
        eps = float(net_income) / shares_outstanding
        if eps > 0:
            pe_values.append(float(year_end_price) / eps)

    if not pe_values:
        return None

    return float(pd.Series(pe_values).median())


def render_scenario_explanation(settings: ProjectionSettings, compare_mode: bool) -> None:
    if compare_mode:
        st.info(
            "Comparison view uses the same starting revenue, margin, and share count, then applies different bear, base, and bull assumptions."
        )
        return

    direction = "dilution" if settings.share_change_rate > 0 else "buybacks" if settings.share_change_rate < 0 else "flat share count"
    st.info(
        f"{settings.scenario} case assumes {settings.revenue_growth * 100:.1f}% starting revenue growth, "
        f"net margin moving from {settings.starting_margin * 100:.1f}% to {settings.terminal_margin * 100:.1f}%, "
        f"{abs(settings.share_change_rate) * 100:.1f}% annual {direction}, "
        f"and a {settings.pe_low:.1f}x-{settings.pe_high:.1f}x P/E range."
    )


def render_projection_notes(settings: ProjectionSettings, compare_mode: bool, projection_notes: list[str]) -> None:
    for note in projection_notes:
        st.info(note)
    render_scenario_explanation(settings, compare_mode)


def render_centered_table(
    df: pd.DataFrame,
    width_ratio: float = 1.0,
    data_font_size: str = "1.1rem",
    header_font_size: str = "0.95rem",
    tooltips: dict[str, str] | None = None,
) -> None:
    safe_width = clamp(width_ratio, 0.35, 1.0)
    header_html = "".join(f"<th>{html.escape(str(column))}</th>" for column in df.columns)
    rows_html = []

    for _, row in df.iterrows():
        cells = []
        for value in row:
            escaped = html.escape(str(value))
            if tooltips and str(value) in tooltips:
                tip = html.escape(tooltips[str(value)])
                escaped += f'<span class="sfv-tip" data-tip="{tip}">&#9432;</span>'
            cells.append(f"<td>{escaped}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    table_html = f"""
        <div style="overflow-x: auto; width: 100%;">
            <table class="sfv-centered-table">
                <thead>
                    <tr>{header_html}</tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
        <style>
        .sfv-centered-table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            margin: 0.25rem 0 1rem;
            background: #11131d;
            border: 1px solid #2c2f3f;
        }}
        .sfv-centered-table th {{
            text-align: center !important;
            vertical-align: middle !important;
            padding: 0.9rem 0.75rem;
            color: #a8a8a8;
            font-size: {header_font_size};
            font-weight: 800;
            border: 1px solid #2c2f3f;
            overflow-wrap: anywhere;
        }}
        .sfv-centered-table td {{
            text-align: center !important;
            vertical-align: middle !important;
            padding: 1rem 0.75rem;
            color: #ffffff;
            font-size: {data_font_size};
            font-weight: 800;
            border: 1px solid #2c2f3f;
            overflow-wrap: anywhere;
        }}
        .sfv-tip {{
            color: #6b8cba;
            font-size: 0.72rem;
            cursor: help;
            margin-left: 0.3rem;
            position: relative;
            display: inline-block;
            font-weight: 400;
            vertical-align: middle;
        }}
        .sfv-tip::after {{
            content: attr(data-tip);
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background: #151827;
            color: #d0d5e8;
            border: 1px solid #313550;
            border-radius: 6px;
            padding: 0.5rem 0.7rem;
            font-size: 0.78rem;
            font-weight: 400;
            white-space: normal;
            min-width: 200px;
            max-width: 270px;
            z-index: 9999;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.15s ease;
            text-align: left;
            line-height: 1.45;
        }}
        .sfv-tip:hover::after {{
            opacity: 1;
        }}
        </style>
    """

    if safe_width >= 0.99:
        st.markdown(table_html, unsafe_allow_html=True)
        return

    side_width = (1.0 - safe_width) / 2
    _, center, _ = st.columns([side_width, safe_width, side_width])
    with center:
        st.markdown(table_html, unsafe_allow_html=True)


def render_simple_projection_view(
    df: pd.DataFrame,
    compare_mode: bool,
    current_price: float,
    cost_basis: float | None = None,
) -> None:
    using_cost_basis = cost_basis is not None and cost_basis > 0

    if compare_mode:
        final_year = df["Year"].max()
        simple_df = df[df["Year"] == final_year][
            [
                "Scenario",
                "Year",
                "Share Price Low ($)",
                "Share Price Average ($)",
                "Share Price High ($)",
                "% Growth Range",
            ]
        ].copy()
    else:
        simple_df = df[
            [
                "Year",
                "Share Price Low ($)",
                "Share Price Average ($)",
                "Share Price High ($)",
                "% Growth Range",
            ]
        ].copy()

    if using_cost_basis:
        simple_df["% Growth Range"] = simple_df.apply(
            lambda row: format_return_range(
                row["Share Price Low ($)"],
                row["Share Price High ($)"],
                current_price,
                cost_basis,
            ),
            axis=1,
        )

    money_columns = ["Share Price Low ($)", "Share Price Average ($)", "Share Price High ($)"]
    for column in money_columns:
        simple_df[column] = simple_df[column].map(format_money)

    render_centered_table(
        simple_df,
        width_ratio=0.82,
        data_font_size="1.5rem",
        header_font_size="1rem",
    )

    if using_cost_basis:
        st.caption("* Returns calculated using your average cost basis")


def render_projection_table(
    title: str,
    df: pd.DataFrame,
    compare_mode: bool,
    current_price: float,
    cost_basis: float | None = None,
) -> None:
    st.subheader(title)
    table_state_key = f"expanded_projection_table_{title}"
    if table_state_key not in st.session_state:
        st.session_state[table_state_key] = False

    button_label = "Show Simple Table" if st.session_state[table_state_key] else "Show Expanded Table"
    st.button(button_label, on_click=toggle_session_bool, args=(table_state_key,))

    using_cost_basis = cost_basis is not None and cost_basis > 0

    if compare_mode:
        if st.session_state[table_state_key]:
            display_df = df
            render_centered_table(display_df, width_ratio=1.0, data_font_size="0.95rem", header_font_size="0.82rem")
        else:
            render_simple_projection_view(df, compare_mode=True, current_price=current_price, cost_basis=cost_basis)
        return

    if st.session_state[table_state_key]:
        display_columns = [
            "Year",
            "Cumulative Rev Growth %",
            "Net Margin %",
            "EPS ($)",
            "Share Price Low ($)",
            "Share Price Average ($)",
            "Share Price High ($)",
            "% Growth Range",
        ]
        display_df = df[display_columns].copy()
        if using_cost_basis:
            display_df["% Growth Range"] = df.apply(
                lambda row: format_return_range(
                    row["Share Price Low ($)"],
                    row["Share Price High ($)"],
                    current_price,
                    cost_basis,
                ),
                axis=1,
            )
        render_centered_table(display_df, width_ratio=1.0, data_font_size="1rem", header_font_size="0.86rem")
        if using_cost_basis:
            st.caption("* Returns calculated using your average cost basis")
    else:
        render_simple_projection_view(df, compare_mode=False, current_price=current_price, cost_basis=cost_basis)


def render_annual_operating_assumptions(df: pd.DataFrame, compare_mode: bool) -> None:
    if compare_mode:
        return

    with st.expander("View Annual Operating Assumptions"):
        growth_df = df[
            [
                "Year",
                "Rev Growth Rate %",
                "Net Margin %",
                "Shares (M)",
                "Revenue (B)",
                "Net Income (B)",
            ]
        ].copy()
        growth_df["Year"] = growth_df["Year"].astype(int).astype(str)
        render_centered_table(growth_df, width_ratio=0.95, data_font_size="1rem", header_font_size="0.86rem")


def render_final_year_summary(
    df: pd.DataFrame,
    compare_mode: bool,
    current_price: float,
    cost_basis: float | None = None,
) -> None:
    final_year = df["Year"].max()
    final_rows = df[df["Year"] == final_year]

    st.subheader(f"{final_year} Price Prediction (5 Year)")
    if compare_mode:
        cols = st.columns(3)
        for col, scenario in zip(cols, ["Bear", "Base", "Bull"]):
            scenario_row = final_rows[final_rows["Scenario"] == scenario]
            if scenario_row.empty:
                continue
            row = scenario_row.iloc[0]
            return_range = format_return_range(
                row["Share Price Low ($)"],
                row["Share Price High ($)"],
                current_price,
                cost_basis,
            )
            col.metric(
                scenario,
                f"{format_money(row['Share Price Low ($)'])} - {format_money(row['Share Price High ($)'])}",
                return_range,
            )
        return

    row = final_rows.iloc[0]
    return_range = format_return_range(
        row["Share Price Low ($)"],
        row["Share Price High ($)"],
        current_price,
        cost_basis,
    )
    col1, col2, col3, col4 = st.columns([1, 1, 2, 2])
    col1.metric("Low Price", format_money(row["Share Price Low ($)"]))
    col2.metric("High Price", format_money(row["Share Price High ($)"]))
    col3.metric("EPS", format_money(row["EPS ($)"]))
    col4.metric("Return Range", return_range)


def render_projection_chart(
    df: pd.DataFrame,
    current_price: float,
    target_mean: float | None,
    compare_mode: bool,
) -> None:
    current_year = dt.date.today().year
    st.subheader("Price Range Over Time")

    if compare_mode:
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X("Year:O", axis=alt.Axis(title="Year", labelAngle=0)),
            y=alt.Y("Share Price Average ($):Q", title="Projected Average Price ($)"),
            color=alt.Color("Scenario:N", scale=alt.Scale(range=["#d95f02", "#1f77b4", "#2ca02c"])),
            tooltip=[
                "Scenario:N",
                "Year:O",
                alt.Tooltip("Share Price Low ($):Q", format="$.2f"),
                alt.Tooltip("Share Price Average ($):Q", format="$.2f"),
                alt.Tooltip("Share Price High ($):Q", format="$.2f"),
            ],
        )
    else:
        area = alt.Chart(df).mark_area(opacity=0.3, color="#31aacf").encode(
            x=alt.X("Year:O", axis=alt.Axis(title="Year", labelAngle=0)),
            y="Share Price Low ($):Q",
            y2="Share Price High ($):Q",
        )
        low_line = alt.Chart(df).mark_line(color="#b41f1f").encode(x="Year:O", y="Share Price Low ($):Q")
        high_line = alt.Chart(df).mark_line(color="#2da721").encode(x="Year:O", y="Share Price High ($):Q")
        chart = area + low_line + high_line

    current_dot = alt.Chart(pd.DataFrame([{"Year": current_year, "Price": current_price, "Label": "Current price"}])).mark_point(
        color="#FFFFFF",
        size=85,
        filled=True,
    ).encode(
        x="Year:O",
        y=alt.Y("Price:Q", title="Projected Share Price ($)"),
        tooltip=["Label:N", alt.Tooltip("Price:Q", format="$.2f")],
    )

    layers = chart + current_dot
    if target_mean:
        target_df = pd.DataFrame([{"Year": current_year + 1, "Target": target_mean, "Label": "Analyst target"}])
        target_marker = alt.Chart(target_df).mark_point(
            color="#f2c94c",
            size=110,
            filled=True,
            shape="diamond",
        ).encode(
            x="Year:O",
            y="Target:Q",
            tooltip=["Label:N", alt.Tooltip("Target:Q", format="$.2f")],
        )
        layers += target_marker

    st.altair_chart(layers, use_container_width=True)


def render_historical_chart(ticker: str, current_year: int) -> None:
    st.markdown("---")
    with st.expander("Historical Price Performance", expanded=True):
        hist_end_year = current_year
        hist_start_year = current_year - 7
        hist_data = fetch_history(ticker, f"{hist_start_year}-01-01", f"{hist_end_year}-12-31")
        if hist_data.empty:
            st.info("No historical data available for this period.")
            return

        year_ticks = pd.to_datetime([f"{year}-01-01" for year in range(hist_start_year, hist_end_year + 1)])
        hist_chart = alt.Chart(hist_data).mark_line(color="#00bfff").encode(
            x=alt.X(
                "Date:T",
                title="Year",
                axis=alt.Axis(format="%Y", labelAngle=0, values=year_ticks),
                scale=alt.Scale(
                    domain=[
                        pd.Timestamp(f"{hist_start_year}-01-01"),
                        pd.Timestamp(f"{hist_end_year}-12-31"),
                    ]
                ),
            ),
            y=alt.Y("Close:Q", title="Closing Price ($)"),
            tooltip=[alt.Tooltip("Date:T", title="Date"), alt.Tooltip("Close:Q", title="Close ($)", format="$.2f")],
        ).properties(height=400)
        st.altair_chart(hist_chart, use_container_width=True)


def _render_financial_period_charts(periods: list, period_col: str, growth_label: str) -> None:
    has_revenue = any(p.revenue is not None for p in periods)
    has_eps = any(p.eps is not None for p in periods)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Revenue History**")
        if has_revenue:
            rev_df = pd.DataFrame(
                [
                    {
                        period_col: p.label,
                        "Revenue ($B)": round(p.revenue / 1e9, 2),
                        growth_label: f"{p.revenue_growth * 100:.1f}%" if p.revenue_growth is not None else "—",
                    }
                    for p in periods
                    if p.revenue is not None
                ]
            )
            rev_chart = (
                alt.Chart(rev_df)
                .mark_bar(color="#31aacf", opacity=0.85)
                .encode(
                    x=alt.X(f"{period_col}:O", sort=None, axis=alt.Axis(labelAngle=-35, title=None)),
                    y=alt.Y("Revenue ($B):Q", title="Revenue ($B)"),
                    tooltip=[f"{period_col}:O", alt.Tooltip("Revenue ($B):Q", format=".2f"), f"{growth_label}:N"],
                )
                .properties(height=220)
            )
            st.altair_chart(rev_chart, use_container_width=True)
        else:
            st.info("Revenue history unavailable.")

    with col_right:
        st.markdown("**EPS History** *(approx.)*")
        if has_eps:
            eps_df = pd.DataFrame(
                [
                    {
                        period_col: p.label,
                        "EPS ($)": round(p.eps, 2),
                        "Type": "Profitable" if p.eps >= 0 else "Loss",
                    }
                    for p in periods
                    if p.eps is not None
                ]
            )
            eps_chart = (
                alt.Chart(eps_df)
                .mark_bar(opacity=0.85)
                .encode(
                    x=alt.X(f"{period_col}:O", sort=None, axis=alt.Axis(labelAngle=-35, title=None)),
                    y=alt.Y("EPS ($):Q", title="EPS ($)"),
                    color=alt.Color(
                        "Type:N",
                        scale=alt.Scale(domain=["Profitable", "Loss"], range=["#2ca02c", "#d95f02"]),
                        legend=None,
                    ),
                    tooltip=[f"{period_col}:O", alt.Tooltip("EPS ($):Q", format=".2f"), "Type:N"],
                )
                .properties(height=220)
            )
            zero_line = (
                alt.Chart(pd.DataFrame({"y": [0]}))
                .mark_rule(color="#555", strokeDash=[4, 4])
                .encode(y="y:Q")
            )
            st.altair_chart(eps_chart + zero_line, use_container_width=True)
        else:
            st.info("EPS history unavailable.")


def render_historical_performance_context(ticker: str, shares_outstanding: float) -> None:
    st.markdown("---")
    with st.expander("Historical Performance Context", expanded=True):
        context = extract_historical_context(ticker, shares_outstanding)
        quarterly_context = extract_quarterly_context(ticker, shares_outstanding)

        if context is None and quarterly_context is None:
            st.info("Insufficient historical financial data to display performance context.")
            return

        tab_annual, tab_quarterly = st.tabs(["Annual", "Quarterly"])

        with tab_annual:
            if context is None:
                st.info("Insufficient annual financial data.")
            else:
                _render_financial_period_charts(context.periods, period_col="Year", growth_label="YoY Growth")

                metric_cols = st.columns(3)
                metric_cols[0].metric("Revenue Trend", context.revenue_trend)
                metric_cols[1].metric("EPS Trend", context.eps_trend)
                metric_cols[2].metric("Rev Volatility", context.revenue_volatility)

                st.markdown(
                    f"""
                    <div style="background:#11131d;border:1px solid #2c2f3f;border-radius:8px;
                    padding:1rem 1.5rem;margin:0.75rem 0;font-family:Helvetica,sans-serif;">
                        <div style="color:#00bfff;font-size:0.85rem;font-weight:700;margin-bottom:0.6rem;">
                            SUMMARY INSIGHTS
                        </div>
                        <div style="color:#e0e0e0;margin-bottom:0.35rem;font-size:0.95rem;">
                            &#8226;&nbsp;{context.revenue_insight}
                        </div>
                        <div style="color:#e0e0e0;font-size:0.95rem;">
                            &#8226;&nbsp;{context.eps_insight}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("EPS approximated using current share count applied to historical net income. Historical context is descriptive only and does not influence projections.")

        with tab_quarterly:
            if quarterly_context is None:
                st.info("Insufficient quarterly financial data.")
            else:
                _render_financial_period_charts(quarterly_context.periods, period_col="Quarter", growth_label="QoQ Growth")

                metric_cols = st.columns(3)
                metric_cols[0].metric("Revenue Trend", quarterly_context.revenue_trend)
                metric_cols[1].metric("EPS Trend", quarterly_context.eps_trend)
                metric_cols[2].metric("Rev Volatility", quarterly_context.revenue_volatility)

                st.markdown(
                    f"""
                    <div style="background:#11131d;border:1px solid #2c2f3f;border-radius:8px;
                    padding:1rem 1.5rem;margin:0.75rem 0;font-family:Helvetica,sans-serif;">
                        <div style="color:#00bfff;font-size:0.85rem;font-weight:700;margin-bottom:0.6rem;">
                            SUMMARY INSIGHTS
                        </div>
                        <div style="color:#e0e0e0;margin-bottom:0.35rem;font-size:0.95rem;">
                            &#8226;&nbsp;{quarterly_context.revenue_insight}
                        </div>
                        <div style="color:#e0e0e0;font-size:0.95rem;">
                            &#8226;&nbsp;{quarterly_context.eps_insight}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("EPS approximated using current share count applied to quarterly net income. Historical context is descriptive only and does not influence projections.")


def is_valid_ticker_response(info: dict) -> bool:
    if not info:
        return False
    if first_not_none(info.get("regularMarketPrice"), info.get("currentPrice")):
        return True
    return bool(info.get("quoteType") and info.get("symbol"))


def render_projection_app(ticker: str) -> None:
    with st.spinner(f"Fetching data for {ticker}..."):
        info = fetch_stock_info(ticker)

    if not is_valid_ticker_response(info):
        st.error(f"'{ticker}' does not look like a valid ticker, or Yahoo Finance did not return usable data.")
        st.caption("Try a common U.S. stock ticker such as AAPL, MSFT, NVDA, GOOGL, or TSLA.")
        return

    inputs, data_warnings, quality = build_stock_inputs(info)
    inputs, info, fallback_note, quality = apply_quarterly_fallback(ticker, inputs, info, quality)
    show_company_summary(ticker, inputs, fallback_note, data_warnings)

    if not inputs.net_income or inputs.net_income <= 0:
        st.warning("Can't predict share price for an unprofitable company. Use an individual profitable company, not an ETF.")
        momentum = compute_earnings_momentum(ticker, inputs.shares_outstanding)
        if momentum is not None:
            render_earnings_momentum_panel(momentum)
        render_historical_performance_context(ticker, inputs.shares_outstanding)
        render_data_quality_panel(quality)
        return

    settings, growth_quality, compare_mode, projection_notes, cost_basis, clamp_note = render_sidebar_inputs(inputs, info)
    quality.extend(growth_quality)

    if inputs.current_price <= 0:
        st.error("Current price is unavailable, so projections cannot be compared to today's price.")
        render_data_quality_panel(quality)
        return

    render_valuation_context(inputs, ticker)

    if compare_mode:
        projection_frames = [
            build_projection_dataframe(inputs.current_price, scenario_settings)
            for scenario_settings in build_scenario_settings(settings, inputs)
        ]
        df = pd.concat(projection_frames, ignore_index=True)
        table_title = "Scenario Comparison"
    else:
        df = build_projection_dataframe(inputs.current_price, settings)
        table_title = f"{settings.scenario} Projection"

    render_final_year_summary(df, compare_mode, current_price=inputs.current_price, cost_basis=cost_basis)
    render_projection_table(table_title, df, compare_mode, current_price=inputs.current_price, cost_basis=cost_basis)
    if clamp_note:
        st.info(clamp_note)
    render_projection_notes(settings, compare_mode, projection_notes)
    render_projection_chart(df, inputs.current_price, inputs.target_mean, compare_mode)
    render_historical_chart(ticker, dt.date.today().year)
    render_historical_performance_context(ticker, inputs.shares_outstanding)
    render_annual_operating_assumptions(df, compare_mode)
    render_assumptions_panel(settings, compare_mode)
    render_data_quality_panel(quality)
    st.caption("Educational use only. This is not financial advice.")


def main() -> None:
    configure_page()
    ticker = render_header()

    if not ticker:
        render_home_screen()
        return

    render_projection_app(ticker)


if __name__ == "__main__":
    main()
