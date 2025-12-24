import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import re
from openai import OpenAI
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="GeoPulse | Strategic Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        height: 100%;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    
    .big-stat { font-size: 32px; font-weight: 700; margin-bottom: 0px; color: #2c3e50; }
    .sub-stat { font-size: 14px; font-weight: 600; margin-bottom: 5px; }
    .stat-label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-top: 5px;}
    
    .trend-bad { color: #c0392b; }
    .trend-good { color: #27ae60; }
    .trend-neutral { color: #7f8c8d; }
    
    .news-item { padding: 12px 0; border-bottom: 1px solid #f1f1f1; }
    .news-title { font-weight: 600; font-size: 15px; color: #2c3e50; margin-bottom: 4px; }
    .news-date { font-size: 11px; color: #95a5a6; }
    
    .risk-high { background-color: #ffebee; color: #c62828; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .risk-med { background-color: #fff3e0; color: #ef6c00; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .risk-low { background-color: #e8f5e9; color: #2e7d32; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def get_color(score):
    # 0 (Peace) -> 100 (War)
    if score < 20: return "#27ae60" # Emerald
    if score < 40: return "#f1c40f" # Yellow
    if score < 60: return "#f39c12" # Orange
    if score < 80: return "#e67e22" # Dark Orange
    return "#c0392b" # Red

def create_gauge(current, past):
    # Visualizes Current Score with a 'Reference' bar for the Past Score
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = current,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "TENSION INDEX", 'font': {'size': 18, 'color': '#7f8c8d'}},
        delta = {'reference': past, 'increasing': {'color': "#c0392b"}, 'decreasing': {'color': "#27ae60"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#bdc3c7"},
            'bar': {'color': get_color(current)},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ecf0f1",
            'steps': [
                {'range': [0, 20], 'color': '#e9f7ef'},
                {'range': [20, 100], 'color': '#fff'}],
            'threshold': {'line': {'color': "#c0392b", 'width': 4}, 'thickness': 0.75, 'value': current}
        }))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'family': "Arial"})
    return fig

def clean_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match: return json.loads(match.group(1))
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: return json.loads(match.group(0))
        return None

def fetch_analysis(c1, c2, key):
    if not key: return {"error": "API Key is missing."}
    
    client = OpenAI(api_key=key, base_url="https://api.perplexity.ai")
    
    system_prompt = """
    You are a Strategic Intelligence Algorithm. Return STRICT JSON.
    
    TASK:
    Analyze the relationship between two nations at TWO points in time:
    1. CURRENT (Now)
    2. PAST (Exactly 1 Year Ago)
    
    SCORING (0-100):
    0-20: Alliance | 21-40: Neutral | 41-60: Strained | 61-80: Hostile | 81-100: Conflict
    
    REQUIRED JSON STRUCTURE:
    {
        "c1_flag": "Emoji", "c2_flag": "Emoji",
        "score_current": Integer (0-100),
        "score_past": Integer (0-100),
        "status_label": "String (e.g. Deteriorating, Improving, Stable)",
        "change_reason": "String (Why did the score change from last year? Max 1 sentence)",
        "summary": "String (Executive summary of current situation)",
        "main_driver": "String (Current primary conflict driver)",
        "trade_deficit": "Float OR String (e.g. 15.2 or 'No Data')",
        "trade_context": "String (e.g. 'US deficit with China')",
        "news": [{"date": "YYYY-MM-DD", "title": "Headline", "source": "Source"}]
    }
    """
    
    user_prompt = (
        f"Analyze {c1} vs {c2}. compare TODAY vs 1 YEAR AGO. "
        "Provide specific tension scores for both timeframes. "
        "For trade_deficit, provide a single number in Billions (USD)."
    )
    
    try:
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        data = clean_json(response.choices[0].message.content)
        if not data: return {"error": "Failed to parse AI response."}
        return data
    except Exception as e:
        return {"error": str(e)}

def fetch_global_rankings(key):
    if not key: return None
    client = OpenAI(api_key=key, base_url="https://api.perplexity.ai")
    system_prompt = """
    Return STRICT JSON with 'highest_pressure' and 'lowest_pressure' (10 items each).
    Item: {"pair": "Name vs Name", "score": Int (0-100), "reason": "Context"}
    """
    try:
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": "Global Geopolitical Rankings 2025"}]
        )
        return clean_json(response.choices[0].message.content)
    except: return None

# --- NEW FUNCTION: MARKET WATCHDOG ---
def fetch_market_risk(commodity, key):
    if not key: return {"error": "API Key Missing"}
    client = OpenAI(api_key=key, base_url="https://api.perplexity.ai")
    
    system_prompt = """
    You are a Global Commodity Risk Analyst. Return STRICT JSON.
    
    TASK:
    1. Identify top 5-8 MAJOR producing countries for the requested commodity.
    2. Analyze the CURRENT geopolitical tension/conflict level for each country.
    3. Calculate a "Supply Chain Risk Score" (0-100) for that commodity based on the stability of these key producers.
    4. Predict price impact purely based on geopolitical risk (Bullish=Prices Up/Risk High, Bearish=Prices Down/Oversupply).
    
    REQUIRED JSON STRUCTURE:
    {
        "commodity": "String",
        "global_risk_score": Integer (0-100),
        "price_outlook": "String (e.g. 'Bullish', 'Bearish', 'Volatile')",
        "outlook_reason": "String (Short explanation of price prediction)",
        "top_producers": [
            {
                "country": "String",
                "production_share": "String (e.g. '12%')",
                "tension_index": Integer (0-100),
                "risk_note": "String (Specific conflict impacting supply, e.g. 'Red Sea shipping attacks')"
            }
        ]
    }
    """
    
    try:
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze Supply Chain Risk for: {commodity}"}
            ]
        )
        return clean_json(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/globe--v1.png", width=60)
    st.title("GeoPulse")
    st.caption("Strategic Intelligence")
    st.divider()
    api_key = st.text_input("API Key (Perplexity)", type="password")
    st.divider()
    page = st.radio("Module", ["📡 Regional Monitor", "📊 Global Heatmap", "📈 Market Watchdog"])

# --- PAGE 1: REGIONAL MONITOR ---
if page == "📡 Regional Monitor":
    st.title("📡 Regional Analysis")
    st.markdown("Real-time diplomatic assessment with historical comparison.")
    
    with st.container():
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: country_a = st.text_input("Entity A", "USA")
        with c2: country_b = st.text_input("Entity B", "India")
        with c3: 
            st.write("##")
            btn = st.button("Initialize Scan", type="primary", use_container_width=True)

    if btn and api_key:
        with st.spinner("Retrieving historical cables & current intel..."):
            data = fetch_analysis(country_a, country_b, api_key)
        
        if "error" in data:
            st.error(f"Error: {data['error']}")
        else:
            # Extract Data
            flag1 = data.get('c1_flag', '')
            flag2 = data.get('c2_flag', '')
            curr = data.get('score_current', 0)
            past = data.get('score_past', 0)
            
            # Calculate Change
            diff = curr - past
            if diff > 0:
                trend_str = f"⬆ +{diff} (Worsening)"
                trend_cls = "trend-bad"
            elif diff < 0:
                trend_str = f"⬇ {diff} (Improving)"
                trend_cls = "trend-good"
            else:
                trend_str = "No Change"
                trend_cls = "trend-neutral"

            # 1. Header
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 25px;">
                <h1 style="margin:0; font-size: 2.5rem;">{flag1} {country_a.upper()} <span style="color:#bdc3c7;">&</span> {country_b.upper()} {flag2}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. Main Dashboard
            col_left, col_mid, col_right = st.columns([1.5, 1, 1])
            
            with col_left:
                # Gauge now shows Delta automatically via Plotly
                st.plotly_chart(create_gauge(curr, past), use_container_width=True)
            
            with col_mid:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="stat-label">YoY Change (1 Year)</div>
                    <div class="sub-stat {trend_cls}">{trend_str}</div>
                    <div style="font-size: 13px; color: #555; margin-top: 5px;">
                        <i>"{data.get('change_reason', 'N/A')}"</i>
                    </div>
                    <hr style="margin: 10px 0; opacity: 0.3;">
                    <div class="stat-label">Primary Driver</div>
                    <div style="font-weight: 600; color: #2c3e50;">{data.get('main_driver', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_right:
                # Trade
                def_raw = data.get('trade_deficit', 'N/A')
                if isinstance(def_raw, (int, float)): val = f"${def_raw} B"
                else: val = str(def_raw).replace(" billion", "B")
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="stat-label">Est. Trade Deficit</div>
                    <div class="big-stat">{val}</div>
                    <div class="stat-label" style="margin-top:0;">{data.get('trade_context', '')}</div>
                </div>
                """, unsafe_allow_html=True)

            # 3. Summary & News
            st.divider()
            col_sum, col_news = st.columns([1.2, 1])
            
            with col_sum:
                st.subheader("📝 Strategic Assessment")
                st.info(data.get('summary', ''))
                st.caption(f"Historical comparison baseline: {datetime.now().year - 1}")

            with col_news:
                st.subheader("📰 Intelligence Feed")
                for item in data.get('news', []):
                    st.markdown(f"""
                    <div class="news-item">
                        <div class="news-title">🔹 {item.get('title')}</div>
                        <div class="news-date">{item.get('source')} • {item.get('date')}</div>
                    </div>
                    """, unsafe_allow_html=True)

# --- PAGE 2: GLOBAL HEATMAP ---
elif page == "📊 Global Heatmap":
    st.title("📊 Global Heatmap")
    
    if not api_key:
        st.warning("API Key required.")
    else:
        # Refresh logic
        col_r1, col_r2 = st.columns([4, 1])
        with col_r2:
            if st.button("🔄 Refresh Data"):
                st.session_state.pop('rankings', None)
                st.rerun()
        
        if "rankings" not in st.session_state:
            with st.spinner("Scanning global datasets..."):
                st.session_state.rankings = fetch_global_rankings(api_key)
        
        ranks = st.session_state.rankings
        if ranks:
            tab1, tab2 = st.tabs(["🔥 Flashpoints (High Tension)", "🕊️ Stable Zones"])
            
            with tab1:
                st.dataframe(pd.DataFrame(ranks.get('highest_pressure', [])), 
                             column_config={"score": st.column_config.ProgressColumn("Tension", min_value=0, max_value=100, format="%d")},
                             hide_index=True, use_container_width=True)
            with tab2:
                st.dataframe(pd.DataFrame(ranks.get('lowest_pressure', [])),
                             column_config={"score": st.column_config.ProgressColumn("Tension", min_value=0, max_value=100, format="%d")},
                             hide_index=True, use_container_width=True)

# --- PAGE 3: MARKET WATCHDOG (NEW) ---
elif page == "📈 Market Watchdog":
    st.title("📈 Commodity Risk Watchdog")
    st.markdown("Analyze how geopolitical tension in top producing nations impacts commodity prices.")
    
    if not api_key:
        st.warning("Please enter your API Key in the sidebar.")
    else:
        # 1. Selection Controls
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            commodity_choice = st.selectbox(
                "Select Commodity to Track:", 
                ["Crude Oil", "Gold", "Silver", "Semiconductors (Chips)"]
            )
        with col_btn:
            st.write("##")
            scan_market = st.button("Analyze Risk", type="primary", use_container_width=True)
            
        if scan_market:
            with st.spinner(f"Analyzing supply chains for {commodity_choice}..."):
                market_data = fetch_market_risk(commodity_choice, api_key)
                
            if "error" in market_data:
                st.error(market_data['error'])
            else:
                # 2. Key Metrics Row
                m1, m2, m3 = st.columns(3)
                
                # Risk Score Coloring
                risk_score = market_data.get('global_risk_score', 0)
                risk_color = "green" if risk_score < 40 else "orange" if risk_score < 70 else "red"
                
                # Outlook Coloring
                outlook = market_data.get('price_outlook', 'Neutral')
                outlook_color = "red" if "Bullish" in outlook else "green" # Bullish usually means prices up (bad for buyers)
                
                with m1:
                    st.metric(label="Global Supply Chain Risk", value=f"{risk_score}/100")
                    st.progress(risk_score)
                with m2:
                    st.metric(label="Price Outlook (Geopolitical)", value=outlook)
                with m3:
                    st.caption("Analyst Note:")
                    st.write(f"_{market_data.get('outlook_reason', 'No data')}_")
                
                st.divider()
                
                # 3. Detailed Producers Table
                st.subheader(f"🌍 Top Producers & Tension Levels")
                
                producers = market_data.get('top_producers', [])
                if producers:
                    # Convert to DataFrame for nicer display
                    df_prod = pd.DataFrame(producers)
                    
                    # Apply some styling functions to the dataframe logic
                    # We will iterate and show custom cards instead of a raw table for better UI
                    
                    for p in producers:
                        t_score = p.get('tension_index', 0)
                        
                        # Dynamic Badge
                        if t_score > 75: badge = "🔴 CRITICAL"
                        elif t_score > 50: badge = "🟠 HIGH"
                        else: badge = "🟢 STABLE"
                        
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid {get_color(t_score)}">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <h4 style="margin:0;">{p.get('country')}</h4>
                                <span style="font-size:0.9em; background:#eee; padding:3px 8px; border-radius:5px;">Share: {p.get('production_share')}</span>
                            </div>
                            <div style="margin-top: 8px; font-size: 0.95em;">
                                <b>Tension Score:</b> {t_score} &nbsp;|&nbsp; <b>Status:</b> {badge}
                            </div>
                            <div style="margin-top: 5px; color: #666; font-size: 0.9em;">
                                ⚠️ <i>Risk Factor: {p.get('risk_note')}</i>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No producer data found.")
