import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt

st.set_page_config(page_title="Stock Future Visualizer", layout="wide")

# --- Header ---
st.markdown("""
    <style>
        /* Center placeholder + typed text */
        .stTextInput > div > div > input {
            text-align: center;
        }
    </style>

    <div style="text-align: center; padding-top: 20px;">
        <h1 style="font-size: 2.5em;">📈 Stock Future Visualizer</h1>
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
    """
    Applies decay to abnormally high growth rates over time.
    
    Args:
        initial_growth: Starting growth rate (e.g., 0.50 for 50%)
        year_number: Which projection year (1-6)
        normal_growth_rate: Target "normal" growth rate to decay towards (default 15%)
        high_growth_threshold: Growth rate above which decay applies (default 30%)
    
    Returns:
        Adjusted growth rate for that year
    """
    # If growth is already normal, no decay needed
    if initial_growth <= high_growth_threshold:
        return initial_growth
    
    # Calculate how much above normal we are
    excess_growth = initial_growth - normal_growth_rate
    
    # Decay factor: more aggressive in early years, gentler later
    # Year 1: 0.85, Year 2: 0.72, Year 3: 0.61, Year 4: 0.52, Year 5: 0.44, Year 6: 0.37
    decay_factor = 0.85 ** year_number
    
    # Apply decay to the excess growth
    decayed_growth = normal_growth_rate + (excess_growth * decay_factor)
    
    # Never go below the normal growth rate
    return max(decayed_growth, normal_growth_rate)

# --- Centered, smaller search box using columns ---
col1, col2, col3 = st.columns([1.5, 2, 1.5])
with col2:
    ticker = st.text_input(
        "",
        placeholder="Enter a stock ticker (e.g. AAPL, NVDA)",
        key="centered_input"
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
            
            # Add toggle for growth normalization
            use_decay = st.sidebar.checkbox("Apply Growth Normalization", value=True, 
                                           help="Automatically reduces abnormally high growth rates over time to more sustainable levels")
            
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

            # Show warning if growth is abnormally high
            if use_decay and (rev_growth_base > 0.40 or net_growth_base > 0.40):
                st.info(f"⚠️ **High Growth Detected**: Recent quarterly growth rates (Rev: {rev_growth_base*100:.1f}%, Earnings: {net_growth_base*100:.1f}%) are abnormally high. Applying normalization to prevent overestimation.")

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
            else:  # Base
                rev_growth_initial = min(rev_growth_base, 0.5)
                net_growth_initial = min(net_growth_base, 0.5)
                pe_low = current_pe - 5
                pe_high = current_pe + 5

            years = [2025, 2026, 2027, 2028, 2029, 2030]
            proj_revenue = revenue
            proj_net_income = net_income
            shares_million = shares

            data = []
            for idx, year in enumerate(years):
                year_number = idx + 1
                
                # Apply decay if enabled
                if use_decay:
                    rev_growth = apply_growth_decay(rev_growth_initial, year_number)
                    net_growth = apply_growth_decay(net_growth_initial, year_number)
                else:
                    rev_growth = rev_growth_initial
                    net_growth = net_growth_initial
                
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
            st.subheader(f"{scenario} Projection")
            
            # Show growth rates in an expander
            with st.expander("📊 View Annual Growth Rates"):
                growth_df = df[["Year", "Rev Growth Rate %", "Earnings Growth Rate %"]].set_index("Year")
                st.dataframe(growth_df, use_container_width=True)
            
            # Show main projection table
            display_df = df[["Year", "Cumulative Rev Growth %", "Cumulative Earnings Growth %", "EPS ($)", "Share Price Low ($)", "Share Price High ($)"]].set_index("Year")
            st.dataframe(display_df, use_container_width=True)

            st.subheader("Price Range Over Time")
            chart = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X('Year:O', axis=alt.Axis(title='Year', labelAngle=0)),
                y=alt.Y('value:Q', title='Price ($)'),
                color='variable:N'
            ).transform_fold(
                ['Share Price Low ($)', 'Share Price High ($)'], as_=['variable', 'value']
            )
            st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")