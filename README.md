## ⭐ Overview

This project brings **real-time sentiment analysis** to life using a clean, intuitive **Streamlit dashboard**.  
It simulates how social platforms like Instagram, Twitter, and Facebook deliver comments in real time — and visualizes how sentiment shifts second-by-second.

It was designed with **storytelling, UI elegance, and data-driven insight** in mind, making it perfect for recruiters, hiring managers, and real-world product demonstrations.

---

## 🎯 What This Project Does

### ✔ Real-time NLP sentiment scoring using VADER  
### ✔ Beautiful KPI cards for instant insight  
### ✔ Platform-level sentiment comparisons  
### ✔ Interactive filtering + CSV export  
### ✔ Rolling sentiment trend charts  
### ✔ Comment drilldown table  
### ✔ Advanced live-stream simulator  
### ✔ Clean, professional UI with custom CSS  

The dashboard feels like a **mini SaaS analytics platform**.

---

## 🧠 Why I Built This

I wanted a project that shows I can:

- Build a complete data product  
- Think like a product designer  
- Use NLP to turn noisy comments into insight  
- Make real-time systems with Python  
- Develop clean interfaces with Streamlit  
- Deploy to the cloud and make it public  

This project demonstrates all of that in one place.

---

## 🏗 Architecture Diagram  
A simple visual representation of how everything works:

      ┌──────────────────────────┐
      │  sentiment_analysis.csv   │
      └─────────────┬────────────┘
                    ▼
           ┌─────────────────┐
           │ Preprocessing   │
           │ (cleaning, NLP) │
           └─────────────────┘
                    ▼
           ┌──────────────────┐
           │ VADER Sentiment  │
           │ Scoring Engine   │
           └──────────────────┘
                    ▼
      ┌────────────────────────────────┐
      │ Streamlit Dashboard            │
      │ - KPIs                         │
      │ - Filters                      │
      │ - Trend Charts                 │
      │ - Platform Insights            │
      │ - Drilldown Table              │
      │ - Real-Time Simulator          │
      └────────────────────────────────┘

---

## 🧩 Features Breakdown

### 🔹 1. **KPI Dashboard**
Summaries for:
- Total comments  
- Positive / Negative / Neutral %  
- Platform counts  

### 🔹 2. **Sentiment Trend Analysis**
Plotly line charts with:
- Rolling average  
- Color-coded trends  
- Time-aware layout  

### 🔹 3. **Platform Sentiment Visualization**
- Grouped bar charts  
- Pie chart for volume  
- Heatmap for time-of-day trends  

### 🔹 4. **Drilldown Table**
See raw comments with metadata:
- Platform  
- Cleaned text  
- Sentiment class  
- Timestamp  

### 🔹 5. **Advanced Simulated Stream**
A smooth, non-blocking, real-time flow:
- LIVE indicator  
- Auto-scrolling  
- Dynamic line chart  
- Toggle start/stop  

Looks and feels like real incoming comments.

---

## 🧠 NLP Model Explanation  
This project uses **VADER (Valence Aware Dictionary and sEntiment Reasoner)**:

- Optimized for social media text  
- Understands emojis, slang, punctuation  
- Outputs a **compound score** between -1 and +1  
- Fast enough for real-time updates  
- Zero training required  

Sentiment classes are derived like:
compound >=  0.05  → Positive
compound <= -0.05  → Negative
otherwise          → Neutral


<img width="2537" height="1018" alt="Screenshot 2025-11-21 002955" src="https://github.com/user-attachments/assets/0086968e-9e50-42fa-90f6-7a45adcfba6f" />

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/real-time-social-sentiment.git

cd real-time-social-sentiment

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py




