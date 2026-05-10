# Building GeoPulse: How I Built an AI-Powered OSINT Dashboard for Geopolitical & Supply Chain Analysis

<p align="center">
  <img src="../assets/logo.png" width="300" alt="GeoPulse Logo"/>
</p>

We live in an era where a single blocked canal or a localized geopolitical skirmish can send shockwaves through global supply chains within hours. Understanding these cascading impacts requires more than just reading the news—it requires synthesizing massive amounts of unstructured diplomatic, economic, and logistical data in real-time.

That’s why I built **GeoPulse**—an AI-powered Open-Source Intelligence (OSINT) dashboard designed to analyze geopolitical relations, trade route vulnerabilities, and commodity risks.

In this article, I want to walk you through what GeoPulse does, the architecture behind it, and how integrating Multi-Agent AI simulations is changing the way we can model "Black Swan" events.

---

## 🌍 What is GeoPulse?

[GeoPulse](https://github.com/jayshilj/GeoPulseWebApp) is an open-source, interactive web application built with Python and Streamlit. It acts as a strategic intelligence hub, allowing analysts, researchers, or enthusiasts to query real-time geopolitical tensions and model the economic fallout of global crises. 

Instead of relying on a single AI model, GeoPulse is **model-agnostic**. It seamlessly integrates with Perplexity, Google Gemini, OpenAI, and DeepSeek, allowing users to leverage the specific strengths of different Large Language Models (LLMs) depending on the task—whether that’s deep web-scraping via Perplexity or advanced reasoning via DeepSeek.

### Core Modules

To make the data digestible, I split the application into distinct, highly visual modules:

**1. The Regional Monitor**  
If you want to understand the exact diplomatic standing between any two entities (e.g., the US and China, or the EU and Russia), this module calculates a precise "Tension Gauge" (0-100). It compares today’s score with the score from exactly one year ago, estimates trade deficits in billions, and pulls the latest intelligence headlines.

**2. The Market Watchdog**  
Supply chains run on commodities. This module takes a commodity (like Crude Oil, Semiconductors, or Lithium) and identifies its top global producers. It calculates a "Supply Chain Risk Score" based on the geopolitical stability of those producers and highlights critical logistical **Choke Points** (like the Strait of Hormuz or the Panama Canal) that could threaten global transport.

**3. The Global Heatmap**  
A macro-level view of the world, dynamically ranking the highest-pressure flashpoints and the most stable zones globally.

---

## 🦢 Simulating Chaos: The Black Swan Simulator

The feature I am most proud of is the **Black Swan Simulator**. 

A "Black Swan" is an unpredictable event with massive consequences. What happens if a major maritime choke point is blocked for weeks? What happens if an unprecedented embargo is placed on critical minerals? 

To answer this, GeoPulse employs two advanced AI techniques:

### Multi-Agent Panic Simulation (Powered by CAMEL-AI)
While traditional LLM prompts give you a high-level summary, human economics is driven by *behavior*. I integrated [CAMEL-AI](https://github.com/camel-ai/camel) to instantiate a multi-agent role-playing environment. 

When a Black Swan event is triggered, the system spins up two AI agents—for example, a **Local Retail Store Manager** and an **Anxious Consumer**. These agents converse with each other in real-time, allowing the user to read a simulated transcript of how ground-level panic buying, stock shortages, and price gouging might actually play out in local markets.

### Iterative Cascading Reaction Graphs
Supply chains are complex networks. When an event occurs, GeoPulse uses an LLM to generate a complex JSON structure of immediate reactions across Logistics, Industry, Retail, and Consumers. 

Using **Pyvis** and **Plotly**, this JSON is rendered into an interactive, force-directed network graph. But it doesn't stop there. I built an **Iterative Graph Expansion** engine. Users can click "Expand Consequences", prompting the LLM to read the current graph, identify the "leaf nodes" (the edges of the reaction), and generate the 2nd and 3rd-order consequences—dynamically growing the network right before the user's eyes.

---

## 🛠️ The Tech Stack & Architecture

Building GeoPulse required stringing together several powerful Python libraries:

- **Frontend**: [Streamlit](https://streamlit.io/) handles the entire UI, allowing for rapid deployment of interactive data widgets without writing raw HTML/React.
- **Visualization**: **Plotly** powers the dynamic tension gauges, while **Pyvis** handles the physics-based network graphs.
- **AI Backend**: The system uses a centralized API router to normalize responses from OpenAI, Gemini, Perplexity, and DeepSeek, enforcing strict JSON output for the frontend to render.

### Professionalizing the Codebase

As the project grew, a monolithic `app.py` script became difficult to manage. I recently undertook a major refactoring effort to professionalize the repository. The logic is now cleanly separated into a `src/` directory containing dedicated modules for `api.py` (LLM networking), `graph.py` (Pyvis generation), and `utils.py` (JSON sanitization and error handling). 

The project is also fully open-sourced under the **MIT License**, meaning anyone is free to fork it, modify it, and use it for their own strategic intelligence needs.

---

## 🚀 Try It Yourself

Whether you are a supply chain professional, a geopolitical analyst, or just a developer interested in how LLMs can be used for complex systems modeling, I encourage you to check out the project.

🔗 **GitHub Repository:** [GeoPulseWebApp](https://github.com/jayshilj/GeoPulseWebApp)

If you find it interesting, feel free to drop a ⭐ on GitHub, open a pull request, or reach out to discuss the future of AI in open-source intelligence!

*Built by Jayshil Jain.*
