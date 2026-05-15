# 🌏 GeoPulse | Strategic Intelligence Dashboard

<p align="center">
  <img src="assets/logo.png" width="250" alt="GeoPulse Logo"/>
</p>

### *AI-Powered Geopolitical Risk Assessment & Supply Chain Intelligence*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

GeoPulse is a professional-grade open-source intelligence (OSINT) dashboard that analyzes geopolitical relations, global trade routes, and commodity risks in real-time. 

Built with **Python** and **Streamlit**, it leverages advanced LLM orchestration (Perplexity, Google, OpenAI, DeepSeek) to synthesize diplomatic reports, economic data, and logistical flows into a structured, highly visual interface for strategic decision-making.

---

## 👨‍💻 Connect with the Developer
**Jayshil Jain**  
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jayshilj)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jayshiljain/)
[![Website](https://img.shields.io/badge/Website-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](https://www.jayshil.com/)

---

## 🚀 Key Modules & Features

### 1. 📡 Regional Monitor
Deep-dive analysis of bilateral relations between any two global entities.
- **Tension Gauge (0-100)**: Real-time visualization of current diplomatic friction.
- **YoY Comparison**: Automated delta calculation between today's tension and the same date last year.
- **Trade Deficit Estimator**: AI-driven estimates of trade imbalances in USD Billions.
- **Intelligence Feed**: Curated live news headlines with source attribution.

### 2. 📊 Global Heatmap
Macro-level assessment of the world's most stable and unstable regions.
- **Real-time Rankings**: Dynamic "Flashpoints" (High Tension) vs. "Stable Zones" (Low Tension) lists.
- **Global Data Synthesis**: Aggregates tension scores across dozens of nations simultaneously.

### 3. 📈 Market Watchdog (NEW)
Analyze how geopolitical instability impacts global markets and commodities.
- **Commodity Risk Assessment**: Supply chain stability tracking for Oil, Gold, Semiconductors, Lithium, and more.
- **Choke Point Monitoring**: Threat analysis for critical nodes (e.g., Strait of Malacca, Suez Canal, Panama Canal).
- **Producer Tension Index**: Tracks the domestic stability of top-producing nations for specific strategic assets.

### 4. 🦢 Black Swan Simulator
Model the cascading effects of global crises using advanced AI simulations.
- **Iterative Relationship Graph**: Force-directed network graph (powered by Pyvis) that allows users to explore 2nd and 3rd order logistical consequences.
- **Oasis Panic Simulation**: Multi-agent role-playing (via CAMEL-AI) between a "Store Manager" and "Consumer" to predict ground-level behavioral economics during a crisis.
- **Micro-Metric Dashboard**: Real-time analysis of ripple effects across Energy, Logistics, and Finance sectors.

---

## 🛠️ Tech Stack
- **Frontend**: Streamlit (Responsive Web UI)
- **Visualization**: Plotly (Interactive Charts), Pyvis (Network Graphs), Matplotlib
- **Multi-Agent Simulation**: CAMEL-AI (Oasis Framework)
- **AI / LLM Orchestration**: 
  - **Perplexity API**: `sonar-pro` (Real-time web search)
  - **Google Gemini**: `gemini-1.5-pro`, `gemini-2.0-flash`
  - **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `o1-mini`
  - **DeepSeek**: `deepseek-chat`, `deepseek-reasoner` (R1)
- **Data Engine**: Pandas, JSON, `yfinance` (Market Data)

---

## 📂 Project Structure

```plaintext
GeoPulseWebApp/
├── app.py                 # Main Streamlit application entry point
├── src/
│   ├── api.py             # LLM orchestration & CAMEL-AI simulation logic
│   ├── graph.py           # Pyvis network visualization engine
│   └── utils.py           # UI styling, gauges, and helper functions
├── docs/
│   ├── ISSUES.md          # Known issues & future roadmap
│   └── medium_article.md  # Detailed write-up on project methodology
├── assets/                # Logos and UI assets
├── requirements.txt       # Project dependencies
├── LICENSE                # MIT License
└── README.md              # Main documentation
```

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- An API Key from one of the supported providers (Perplexity, Google, OpenAI, or DeepSeek).

### Step 1: Clone the Repository
```bash
git clone https://github.com/jayshilj/GeoPulseWebApp.git
cd GeoPulseWebApp
```

### Step 2: Set Up Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

---

## 🖥️ Usage Guide
1. **Configure Model**: Select your preferred LLM provider and enter your API Key in the sidebar.
2. **Select Module**: Choose between Regional Monitor, Global Heatmap, Market Watchdog, or the Black Swan Simulator.
3. **Execute Scan**: Input your geopolitical parameters and click the action button (e.g., "Initialize Scan", "Analyze Risk").
4. **Analyze Impacts**: Use the interactive maps, cascading impact cards, and the iterative graph expansion to assess geopolitical risk.

---

## ⚠️ Disclaimer
This tool is for informational purposes only. Data is generated by LLMs researching the internet in real-time. While highly effective for strategic intelligence, AI can occasionally misinterpret nuanced diplomatic events or hallucinate specific figures. Always verify critical data with official government sources (e.g., World Bank, IMF, Ministry of External Affairs).

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change. All contributions are subject to the MIT License.
