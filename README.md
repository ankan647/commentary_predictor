# FIFA World Cup 2026: Pressure Index Analytics ⚽📈

> **Can the language of football commentary predict when a goal is about to happen?**

This project explores the relationship between live text commentary and goal events during the FIFA World Cup 2026. Instead of traditional xG or passing network data, this analysis treats play-by-play commentary as a **sensor stream**—using NLP and sentiment analysis to extract a rolling "Pressure Index" to see if mounting pressure reliably precedes goals.

## 🚀 Key Findings

After scraping and analyzing 25 matches (2,397 events, 78 goals), the data reveals that commentary-derived pressure alone doesn't universally predict goals, but segmentation by **goal type** uncovers distinct tactical patterns:

- **Counter-Attacks (+1.89 Uplift):** The opposing team was often generating high pressure right before a rapid transition goal. A spike in the opponent's pressure might be a stronger warning than your own.
- **Set Pieces (p=0.15):** Goals trend toward *lower* pre-goal pressure, aligning with the pause-and-restart nature of dead-ball situations.
- **Open Play (60% of goals):** Near-zero net pressure uplift, indicating that most open-play goals emerge from moments of chance or individual brilliance rather than sustained, grinding pressure.

## 📁 Project Structure

- **`scrape_commentary_try.py`**: Selenium-based web scraper built to extract minute-by-minute text commentary from LiveScore, bypassing dynamic JavaScript rendering.
- **`build_features.py`**: The NLP engine. It uses Regex pattern matching to classify events into 14+ tactical categories and applies VADER sentiment analysis to assign urgency weights to every sentence.
- **`validate_pressure.py`**: The statistical validation suite. Uses pandas, SciPy, and Matplotlib/Seaborn to generate pressure windows, run t-tests, and output validation plots to the `visuals/` directory.
- **`dashboard.html`**: A fully immersive 3D interactive dashboard built with Vanilla JS, **Three.js** (for the 3D stadium, pitch, and rotating football), and **Plotly** (for data visualization).
- **`master_commentary.csv` & `pressure_index.csv`**: The datasets containing raw text events and the computed rolling 5-minute pressure index per team.

## 🛠️ Tech Stack

- **Data Collection:** Python, Selenium WebDriver
- **NLP & Processing:** VADER Sentiment Analysis (NLTK), Regex, Pandas, NumPy
- **Statistics & Plots:** SciPy, Matplotlib, Seaborn
- **Frontend Dashboard:** HTML, CSS, JavaScript, Three.js (3D WebGL), Plotly.js

## 🏃 How to Run

1. **Install Dependencies:**
   Ensure you have Python installed, then install the required libraries:
   ```bash
   pip install pandas numpy nltk selenium matplotlib seaborn scipy
   ```
   *(Note: You'll also need the appropriate WebDriver for Selenium if you plan to re-scrape the data).*

2. **Generate the Analysis:**
   If you want to re-run the NLP classification or statistical validation:
   ```bash
   python build_features.py
   python validate_pressure.py
   ```
   This will output new plots to the `visuals/` folder and log statistical summaries in the console.

3. **View the 3D Dashboard:**
   Start a local HTTP server to avoid CORS issues with the 3D canvas and Plotly scripts:
   ```bash
   python -m http.server 8888
   ```
   Then open `http://localhost:8888/dashboard.html` in your web browser.

## 📊 Dashboard Preview

The web dashboard is an immersive 3D experience. As you scroll, the camera orbits a realistic 3D pitch complete with floodlights and a rotating ball, revealing the methodology, statistical scoreboards, and Plotly charts that detail the pressure uplifts and goal distributions.

---
*Created as an exploration into football analytics using natural language processing.*
