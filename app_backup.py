import datetime as dt
import html
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


def configure_page() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(
        """
        <style>
        .stTextInput, .stButton > button {
            position: relative !important;
            z-index: 9999 !important;
        }
        .stTextInput > div > div > input {
            text-align: center;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #0a0b13;
        }
        [data-testid="stSidebar"] {
            background-color: #1a1b25;
            border-right: 1px solid #1f1f2e;
        }
        .stButton button {
            border-radius: 8px;
            font-weight: 500;
            transition: transform 0.2s ease;
        }
        .stButton button:hover {
            transform: scale(1.02);
        }
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
            company_name=info.get("longName", "N/A"),
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
    if st.session_state.get("ticker"):
        left, _, _ = st.columns([1, 6, 1])
        with left:
            st.button("Home", on_click=go_home)

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


def show_company_summary(inputs: StockInputs, fallback_note: str | None, data_warnings: list[str]) -> None:
    render_analyst_targets(inputs)

    st.markdown(
        f"""
        <div style="margin: 0.25rem 0 1rem;">
            <div style="color: #a8a8a8; font-size: 0.9rem;">Company</div>
            <div style="font-size: 1.45rem; font-weight: 700; line-height: 1.25; overflow-wrap: anywhere;">
                {inputs.company_name}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    col1.metric("Current Price", format_money(inputs.current_price))
    col2.metric("Market Cap", format_market_cap(inputs.market_cap))
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


def get_growth_assumptions(info: dict) -> tuple[float, float, list[DataQuality]]:
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

    if raw_rev_growth != rev_growth_base:
        st.info(f"Revenue growth clamped from {raw_rev_growth * 100:.1f}% to {rev_growth_base * 100:.1f}%.")
        quality.append(DataQuality("Growth clamp", "Used", "Revenue growth was outside the allowed range."))

    return rev_growth_base, raw_net_growth if raw_net_growth is not None else DEFAULT_GROWTH_RATE, quality


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
) -> tuple[ProjectionSettings, list[DataQuality], bool]:
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

    st.sidebar.text("Data from Yahoo Finance:")
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

    rev_growth_base, _, growth_quality = get_growth_assumptions(info)

    if use_decay and rev_growth_base > 0.40:
        st.info(
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
        st.sidebar.text("Scenario assumptions:")
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
    return settings, growth_quality, compare_mode


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
        midpoint = (price_low + price_high) / 2
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
                "Share Price High ($)": round(price_high, 2),
                "Share Price Mid ($)": round(midpoint, 2),
            }
        )

    df = pd.DataFrame(data)
    df["% Growth Range"] = df.apply(
        lambda row: (
            f"{((row['Share Price Low ($)'] - current_price) / current_price * 100):.0f}% - "
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
        render_centered_table(assumptions, width_ratio=0.72, data_font_size="1.15rem")


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
        render_centered_table(quality_df, width_ratio=0.9, data_font_size="1rem", header_font_size="0.9rem")


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


def render_centered_table(
    df: pd.DataFrame,
    width_ratio: float = 1.0,
    data_font_size: str = "1.1rem",
    header_font_size: str = "0.95rem",
) -> None:
    safe_width = clamp(width_ratio, 0.35, 1.0)
    header_html = "".join(f"<th>{html.escape(str(column))}</th>" for column in df.columns)
    rows_html = []

    for _, row in df.iterrows():
        cells_html = "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        rows_html.append(f"<tr>{cells_html}</tr>")

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
        </style>
    """

    if safe_width >= 0.99:
        st.markdown(table_html, unsafe_allow_html=True)
        return

    side_width = (1.0 - safe_width) / 2
    _, center, _ = st.columns([side_width, safe_width, side_width])
    with center:
        st.markdown(table_html, unsafe_allow_html=True)


def render_simple_projection_view(df: pd.DataFrame, compare_mode: bool) -> None:
    if compare_mode:
        final_year = df["Year"].max()
        simple_df = df[df["Year"] == final_year][
            [
                "Scenario",
                "Year",
                "Share Price Low ($)",
                "Share Price High ($)",
                "% Growth Range",
            ]
        ].copy()
    else:
        simple_df = df[
            [
                "Year",
                "Share Price Low ($)",
                "Share Price High ($)",
                "% Growth Range",
            ]
        ].copy()

    money_columns = ["Share Price Low ($)", "Share Price High ($)"]
    for column in money_columns:
        simple_df[column] = simple_df[column].map(format_money)

    render_centered_table(
        simple_df,
        width_ratio=0.76,
        data_font_size="1.5rem",
        header_font_size="1rem",
    )


def render_projection_table(title: str, df: pd.DataFrame, compare_mode: bool) -> None:
    st.subheader(title)
    table_state_key = f"expanded_projection_table_{title}"
    if table_state_key not in st.session_state:
        st.session_state[table_state_key] = False

    button_label = "Show Simple Table" if st.session_state[table_state_key] else "Show Expanded Table"
    st.button(button_label, on_click=toggle_session_bool, args=(table_state_key,))

    if compare_mode:
        if st.session_state[table_state_key]:
            display_df = df
            render_centered_table(display_df, width_ratio=1.0, data_font_size="0.95rem", header_font_size="0.82rem")
        else:
            render_simple_projection_view(df, compare_mode=True)
        return

    if st.session_state[table_state_key]:
        display_columns = [
            "Year",
            "Cumulative Rev Growth %",
            "Net Margin %",
            "EPS ($)",
            "Share Price Low ($)",
            "Share Price High ($)",
            "% Growth Range",
        ]
        display_df = df[display_columns]
        render_centered_table(display_df, width_ratio=1.0, data_font_size="1rem", header_font_size="0.86rem")
    else:
        render_simple_projection_view(df, compare_mode=False)

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
        ]
        render_centered_table(growth_df, width_ratio=0.95, data_font_size="1rem", header_font_size="0.86rem")


def render_final_year_summary(df: pd.DataFrame, compare_mode: bool) -> None:
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
            col.metric(
                scenario,
                f"{format_money(row['Share Price Low ($)'])} - {format_money(row['Share Price High ($)'])}",
                row["% Growth Range"],
            )
        return

    row = final_rows.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Low Price", format_money(row["Share Price Low ($)"]))
    col2.metric("High Price", format_money(row["Share Price High ($)"]))
    col3.metric("EPS", format_money(row["EPS ($)"]))
    col4.metric("Return Range", row["% Growth Range"])


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
            y=alt.Y("Share Price Mid ($):Q", title="Projected Midpoint Price ($)"),
            color=alt.Color("Scenario:N", scale=alt.Scale(range=["#d95f02", "#1f77b4", "#2ca02c"])),
            tooltip=[
                "Scenario:N",
                "Year:O",
                alt.Tooltip("Share Price Low ($):Q", format="$.2f"),
                alt.Tooltip("Share Price Mid ($):Q", format="$.2f"),
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
    with st.expander("Historical Price Performance", expanded=False):
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
    show_company_summary(inputs, fallback_note, data_warnings)

    if not inputs.net_income or inputs.net_income <= 0:
        st.warning("Can't predict share price for an unprofitable company. Use an individual profitable company, not an ETF.")
        render_data_quality_panel(quality)
        return

    settings, growth_quality, compare_mode = render_sidebar_inputs(inputs, info)
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

    render_final_year_summary(df, compare_mode)
    render_scenario_explanation(settings, compare_mode)
    render_projection_table(table_title, df, compare_mode)
    render_assumptions_panel(settings, compare_mode)
    render_data_quality_panel(quality)
    render_projection_chart(df, inputs.current_price, inputs.target_mean, compare_mode)
    render_historical_chart(ticker, dt.date.today().year)
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
