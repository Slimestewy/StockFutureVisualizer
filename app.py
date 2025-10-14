import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt

st.set_page_config(page_title="Stock Future Visualizer", layout="wide")

# --- Home Button (configurable position) ---
def go_home():
    st.session_state["ticker"] = ""

# Choose button position: "left", "center", "right"
home_position = "left"  # Change this to move button

col_left, col_center, col_right = st.columns([1, 6, 1])
columns_dict = {"left": col_left, "center": col_center, "right": col_right}

if st.session_state.get("ticker"):  # Only show button if ticker exists
    with columns_dict[home_position]:
        st.button("Home", on_click=go_home)

# --- Header with emojis ---
st.markdown("""
    <style>
        /* Center placeholder + typed text */
        .stTextInput > div > div > input {
            text-align: center;
        }
    </style>

    <div style="text-align: center; padding-top: 20px;">
        <h1 style="font-size: 2.5em;">📈 Stock Future Visualizer 💸</h1>
    </div>
""", unsafe_allow_html=True)

def format_market_cap(market_cap):
    if market_cap >= 1e12:
        return f"~${market_cap/1e12:.2f}T"
    elif market_cap >= 1e9:
        return f"~${market_cap/1e9:.2f}B"
    elif market_cap >= 1e6:
        return f"~${market_cap/1e6:.2f}M"
    else:
        return f"~${market_cap:,.0f}"

def apply_growth_decay(initial_growth, year_number, normal_growth_rate=0.15, high_growth_threshold=0.30):
    if initial_growth <= high_growth_threshold:
        return initial_growth
    excess_growth = initial_growth - normal_growth_rate
    decay_factor = 0.85 ** year_number
    decayed_growth = normal_growth_rate + (excess_growth * decay_factor)
    return max(decayed_growth, normal_growth_rate)

# --- Centered search box ---
col1, col2, col3 = st.columns([1.5, 2, 1.5])
with col2:
    ticker = st.text_input(
        "",
        placeholder="Enter a stock ticker (e.g. AAPL, NVDA)",
        key="ticker"
    )

# --- Home Screen ---
if not ticker:
    st.markdown("""
        <div style="text-align: center; padding: 80px 20px;">
            <h1 style="font-size: 3em;">📊 Welcome to <span style="color:#00bfff;">Stock Future Visualizer</span></h1>
            <p style="font-size: 1.2em; color: gray; max-width: 600px; margin: auto;">
                Explore future stock price projections based on real financial data.  
                Enter a stock ticker (like <b>AAPL</b> or <b>NVDA</b>) above to get started.
            </p>
            <br>
            <img src="https://cdn-icons-png.flaticon.com/512/2331/2331946.png" width="180" style="opacity:0.85;"/>
            <br><br>
            <p style="color: #666; font-size: 0.8em;">Powered by Yahoo Finance + Streamlit</p>
            <p style="color: #666; font-size: 0.8em;">Programmed by Evan Kulesza</p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- Main App ---
if ticker:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        company_name = info.get('longName', 'N/A')
        current_price = info.get('currentPrice', 0.0)
        market_cap = info.get('marketCap', 0.0)
        shares_out = info.get('sharesOutstanding') or 1_000_000
        current_pe = info.get('trailingPE') or 15.0
        total_revenue = info.get('totalRevenue') or 100_000_000.0
        net_income_val = info.get('netIncomeToCommon')

        st.markdown(f"**Company:** {company_name}")
        st.markdown(f"**Current Price:** ${current_price:,.2f}")
        st.markdown(f"**Market Cap:** {format_market_cap(market_cap)}")

        if not net_income_val or net_income_val <= 0:
            st.warning("Company does not have profits yet, can't predict share price for an unprofitable company.")
        else:
            st.sidebar.header("Projection Inputs")
            scenario = st.sidebar.selectbox("Select Scenario", ["Base", "Bear", "Bull", "Custom"])
            use_decay = st.sidebar.checkbox("Apply Growth Normalization", value=True, help="Automatically reduces abnormally high growth rates over time to more sustainable levels")
            st.sidebar.markdown("---")

            if scenario != "Custom":
                st.sidebar.text("Data from Yahoo Finance:")
                st.sidebar.text_input("Revenue (Billion $)", value=f"${total_revenue/1e9:.2f}", disabled=True)
                st.sidebar.text_input("Net Income (Billion $)", value=f"${net_income_val/1e9:.2f}", disabled=True)
                st.sidebar.text_input("Shares Outstanding (Millions)", value=f"{shares_out/1e6:.2f}", disabled=True)
                st.sidebar.text_input("Current P/E", value=f"{current_pe:.2f}", disabled=True)
            else:
                revenue = st.sidebar.number_input("Current Year Revenue (Billion $)", min_value=0.0, value=float(total_revenue)/1e9, step=0.1)
                net_income = st.sidebar.number_input("Current Year Net Income (Billion $)", min_value=0.0, value=float(net_income_val)/1e9, step=0.01)
                shares = st.sidebar.number_input("Shares Outstanding (Millions)", min_value=0.0, value=float(shares_out)/1e6, step=1.0)

            if scenario != "Custom":
                revenue = total_revenue / 1e9
                net_income = net_income_val / 1e9
                shares = shares_out / 1e6

            rev_growth_base = info.get('revenueGrowth') or 0.10
            net_growth_base = info.get('earningsQuarterlyGrowth') or 0.10

            if use_decay and (rev_growth_base > 0.40 or net_growth_base > 0.40):
                st.info(f"⚠️ High growth detected (Rev: {rev_growth_base*100:.1f}%, Earnings: {net_growth_base*100:.1f}%) - applying normalization.")

            if scenario == "Bear":
                rev_growth_initial = rev_growth_base / 2
                net_growth_initial = net_growth_base / 2
                pe_low = (current_pe - 5) - 10
                pe_high = (current_pe + 5) - 10
            elif scenario == "Bull":
                rev_growth_initial = rev_growth_base
                net_growth_initial = net_growth_base
                pe_low = (current_pe - 5) + 10
                pe_high = (current_pe + 5) + 10
            elif scenario == "Custom":
                rev_growth_initial = st.sidebar.number_input("Custom Revenue Growth %", min_value=0.0, max_value=100.0, value=float(min(rev_growth_base, 0.5)*100), step=0.1)/100
                net_growth_initial = st.sidebar.number_input("Custom Net Income Growth %", min_value=0.0, max_value=100.0, value=float(min(net_growth_base, 0.5)*100), step=0.1)/100
                pe_low = st.sidebar.number_input("Custom P/E Low", min_value=0.0, value=current_pe-5, step=0.1)
                pe_high = st.sidebar.number_input("Custom P/E High", min_value=0.0, value=current_pe+5, step=0.1)
            else:
                rev_growth_initial = min(rev_growth_base, 0.5)
                net_growth_initial = min(net_growth_base, 0.5)
                pe_low = current_pe - 5
                pe_high = current_pe + 5

            # --- Projection calculations ---
            years = [2025, 2026, 2027, 2028, 2029, 2030]
            proj_revenue = revenue
            proj_net_income = net_income
            shares_million = shares

            data = []
            for idx, year in enumerate(years):
                year_number = idx + 1
                rev_growth = apply_growth_decay(rev_growth_initial, year_number) if use_decay else rev_growth_initial
                net_growth = apply_growth_decay(net_growth_initial, year_number) if use_decay else net_growth_initial

                proj_revenue *= (1 + rev_growth)
                proj_net_income *= (1 + net_growth)
                proj_net_income_actual = proj_net_income * 1e9
                shares_actual = shares_million * 1e6
                eps = proj_net_income_actual / shares_actual
                price_low = eps * pe_low
                price_high = eps * pe_high
                revenue_growth_pct = ((proj_revenue - revenue) / revenue) * 100
                net_income_growth_pct = ((proj_net_income - net_income) / net_income) * 100

                data.append({
                    "Year": year,
                    "Rev Growth Rate %": round(rev_growth * 100, 2),
                    "Earnings Growth Rate %": round(net_growth * 100, 2),
                    "Cumulative Rev Growth %": round(revenue_growth_pct, 2),
                    "Cumulative Earnings Growth %": round(net_income_growth_pct, 2),
                    "EPS ($)": round(eps, 2),
                    "Share Price Low ($)": round(price_low, 2),
                    "Share Price High ($)": round(price_high, 2)
                })

            df = pd.DataFrame(data)

            # --- Projection Table & Growth Rates ---
            st.subheader(f"{scenario} Projection")
            with st.expander("📊 View Annual Growth Rates"):
                growth_df = df[["Year", "Rev Growth Rate %", "Earnings Growth Rate %"]].set_index("Year")
                st.dataframe(growth_df, use_container_width=True)

            display_df = df[["Year", "Cumulative Rev Growth %", "Cumulative Earnings Growth %", "EPS ($)", "Share Price Low ($)", "Share Price High ($)"]].set_index("Year")
            st.dataframe(display_df, use_container_width=True)

            # --- Price Range Chart ---
            st.subheader("Price Range Over Time")
            area = alt.Chart(df).mark_area(opacity=0.3, color="#31aacf").encode(
                x=alt.X('Year:O', axis=alt.Axis(title='Year', labelAngle=0)),
                y='Share Price Low ($):Q',
                y2='Share Price High ($):Q'
            )
            line = alt.Chart(df).mark_line(color="#b41f1f").encode(x='Year:O', y='Share Price Low ($):Q') + \
                   alt.Chart(df).mark_line(color="#2da721").encode(x='Year:O', y='Share Price High ($):Q')
            current_dot = alt.Chart(pd.DataFrame([{'Year': 2025, 'Price': current_price}])).mark_point(
                color='black', size=80, filled=True
            ).encode(x='Year:O', y='Price:Q', tooltip=[alt.Tooltip('Price:Q', title='Current Price')])
            chart = area + line + current_dot
            st.altair_chart(chart, use_container_width=True)

            # --- Historical Data (2019–2024) ---
            st.markdown("---")
            with st.expander("📊 Historical Data", expanded=False):
                st.subheader("📉 Historical Price Performance (2019–2024)")
                hist_data = stock.history(start="2019-01-01", end="2024-12-31").reset_index()
                if not hist_data.empty:
                    year_ticks = pd.to_datetime([f"{y}-01-01" for y in range(2019, 2025)])
                    hist_chart = alt.Chart(hist_data).mark_line(color="#00bfff").encode(
                        x=alt.X("Date:T", title="Year", axis=alt.Axis(format="%Y", labelAngle=0, values=year_ticks),
                                scale=alt.Scale(domain=[pd.Timestamp("2019-01-01"), pd.Timestamp("2024-12-31")])),
                        y=alt.Y("Close:Q", title="Closing Price ($)"),
                        tooltip=[alt.Tooltip('Date:T', title='Date'), alt.Tooltip('Close:Q', title='Close ($)')]
                    ).properties(height=400)
                    st.altair_chart(hist_chart, use_container_width=True)
                else:
                    st.info("No historical data available for this period.")

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
