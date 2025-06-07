#%%
import requests
import sqlite3
import time

def fetch_papers(start_date, end_date, cursor=0):
    # Bouw de URL om alleen bioinformatica-artikelen op te halen
    url = f"https://api.biorxiv.org/details/biorxiv/{start_date}/{end_date}/{cursor}?category=neuroscience"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error fetching data:", response.status_code)
        return None

# Maak of open de SQLite-database en maak de tabel aan als deze nog niet bestaat
conn = sqlite3.connect("bioinformatics_papers.db")
cur = conn.cursor()

cur.execute('''
    CREATE TABLE IF NOT EXISTS papers (
        doi TEXT PRIMARY KEY,
        title TEXT,
        authors TEXT,
        date TEXT,
        version TEXT,
        type TEXT,
        license TEXT,
        category TEXT,
        jatsxml TEXT,
        abstract TEXT,
        published TEXT,
        server TEXT
    )
''')
conn.commit()

# Stel het datuminterval in en initialiseer de cursor voor paginering
start_date = "2020-05-01"
end_date = "2025-06-05"  # Pas dit aan naar je gewenste interval
cursor_val = 0

while True:
    data = fetch_papers(start_date, end_date, cursor_val)
    if data is None:
        break

    messages = data.get("messages", [])
    if not messages:
        break
    count = int(messages[0].get("count", 0))
    
    articles = data.get("collection", [])
    if not articles:
        break

    # Print de laatste datum van de huidige API-call
    last_date = articles[-1].get("date", "unknown")
    print(f"Laatste datum in deze call: {last_date}, cursor: {cursor_val}, aantal artikelen: {count}", end="\r")

    # Verwerk elk artikel en sla het op in de database
    for article in articles:
        doi = article.get("doi")
        title = article.get("title")
        authors = article.get("authors")
        date_paper = article.get("date")
        version = article.get("version")
        type_paper = article.get("type")
        paper_license = article.get("license")
        category = article.get("category")
        jatsxml = article.get("jatsxml")
        abstract = article.get("abstract")
        published = article.get("published")
        server = article.get("server")
        
        try:
            cur.execute('''
                INSERT INTO papers 
                (doi, title, authors, date, version, type, license, category, jatsxml, abstract, published, server)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (doi, title, authors, date_paper, version, type_paper, paper_license, category, jatsxml, abstract, published, server))
        except sqlite3.IntegrityError:
            # Indien de DOI al bestaat, overslaan
            continue

    conn.commit()
    
    # Als er minder dan 100 records zijn, is dit waarschijnlijk de laatste pagina
    if count < 100:
        break

    # Update de cursor voor de volgende batch
    cursor_val += count
    time.sleep(.1)  # Even pauzeren om de server niet te overbelasten

cur.execute('''
            update papers
            set category = rtrim(lower(category))
            ''')
conn.commit()
conn.close()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.colors as mcolors # Added for custom colormap

# --- Configuration ---
process_all_categories = True 
make_category_plots = False

defined_categories = ["pediatrics"] 

# Nieuwe filterparameters voor de heatmap categorieën
MIN_MODELS_PER_YEAR_HEATMAP = 20 # Aantal geclassificeerde modellen (MM, ML, of Hybrid) per jaar
MIN_TOTAL_PAPERS_PER_YEAR_HEATMAP = 20 # Totaal aantal papers (ongeacht classificatie) per jaar
FILTER_START_YEAR_HEATMAP = 2020 
FILTER_END_YEAR_HEATMAP = 2025 

# --- Helper Functions for SQL Query Building ---
def build_like_conditions(terms, field):
    if not terms or (len(terms) == 1 and terms[0].strip() == ""):
        return "1=1" 
    return " OR ".join([f"lower({field}) LIKE '%{term.lower()}%'" for term in terms])

def build_combined_like_conditions(terms):
    title_conditions = build_like_conditions(terms, 'title')
    abstract_conditions = build_like_conditions(terms, 'abstract')
    
    if title_conditions == "1=1" and abstract_conditions == "1=1":
        return "1=1"  
    elif title_conditions == "1=1":
        return abstract_conditions
    elif abstract_conditions == "1=1":
        return title_conditions
    else:
        return f"({title_conditions}) OR ({abstract_conditions})"

# --- Define Search Term Lists ---
mm_terms = ["mechanistic", "kinetic", "differential equation", "simulation", "first principles",
            "physics-based model","computational model", "systems biology", "action kinetics",
            "biochemical network", "dynamic model", "mathematical model", "stochastic model",
            "agent-based model", "cellular automata", "flux balance analysis"]

ml_terms = ["machine learning", "deep learning", "neural network", "artificial intelligence",
            "predictive model", "data-driven", "supervised learning","unsupervised learning",
            "reinforcement learning","ensemble learning","transfer learning","feature extraction",
            "pattern recognition","support vector machine", "random forest","gradient boosting"]

explicit_hybrid = ["hybrid model", "mechanistic learning"]

shared_terms = [""] 

# --- Database Interaction Functions ---
def get_filtered_categories_from_db(min_models_per_year, min_total_papers_per_year, 
                                     filter_start_year, filter_end_year, specific_categories=None):
    """
    Fetches unique categories from the database that meet the specified criteria:
    - At least `min_models_per_year` (MM, ML, or Hybrid combined) for EVERY year from `filter_start_year` to `filter_end_year`.
    - At least `min_total_papers_per_year` (all papers) for EVERY year from `filter_start_year` to `filter_end_year`.
    If `specific_categories` is provided, filters only within those categories.
    """
    conn = sqlite3.connect("bioinformatics_papers.db")
    
    # Fetch all papers with category, date, title, and abstract within the relevant year range
    query = f"""
    SELECT doi, date, title, abstract, lower(category) as category
    FROM papers
    WHERE category IS NOT NULL AND category != ''
      AND STRFTIME('%Y', date) >= '{filter_start_year}'
      AND STRFTIME('%Y', date) <= '{filter_end_year}';
    """
    df_all_papers = pd.read_sql_query(query, conn)
    conn.close()

    if df_all_papers.empty:
        return []

    df_all_papers['date'] = pd.to_datetime(df_all_papers['date'])
    df_all_papers['year'] = df_all_papers['date'].dt.year

    # Filter for specific categories if provided
    if specific_categories:
        df_all_papers = df_all_papers[df_all_papers['category'].isin([c.lower() for c in specific_categories])]
        if df_all_papers.empty:
            return []

    eligible_categories = []
    
    # Check each unique category
    for category_name in df_all_papers['category'].unique():
        df_cat = df_all_papers[df_all_papers['category'] == category_name].copy()
        
        # Classify papers for this category to get model counts
        df_classified_counts = classify_papers(df_cat)
        df_classified_counts['year'] = df_classified_counts['date'].dt.year
        
        # Aggregate counts per year for models and total papers
        yearly_model_counts = df_classified_counts.groupby('year')[['count_mm', 'count_ml', 'count_hybrid']].sum().reset_index()
        yearly_model_counts['total_models_classified'] = yearly_model_counts['count_mm'] + yearly_model_counts['count_ml'] + yearly_model_counts['count_hybrid']
        
        yearly_total_papers = df_cat.groupby('year').size().reset_index(name='total_papers')
        
        # Combine model and total paper counts
        yearly_combined_counts = pd.merge(yearly_total_papers, yearly_model_counts, on='year', how='left').fillna(0)
        
        # Check if all required years are present and meet the minimum thresholds
        meets_criteria = True
        required_years = set(range(filter_start_year, filter_end_year + 1))
        years_present_in_data = set(yearly_combined_counts['year'])

        if not required_years.issubset(years_present_in_data):
            meets_criteria = False
        else:
            for year in required_years:
                year_data = yearly_combined_counts[yearly_combined_counts['year'] == year]
                if year_data.empty: # Should not happen if required_years.issubset was true, but good for robustness
                    meets_criteria = False
                    break
                
                total_models_in_year = year_data['total_models_classified'].iloc[0]
                total_papers_in_year = year_data['total_papers'].iloc[0]
                
                if total_models_in_year < min_models_per_year or total_papers_in_year < min_total_papers_per_year:
                    meets_criteria = False
                    break
        
        if meets_criteria:
            eligible_categories.append(category_name)
            
    return eligible_categories


def fetch_papers_for_category(category_name):
    """
    Fetches all papers for a given category from the database, applying shared terms filter.
    """
    conn = sqlite3.connect("bioinformatics_papers.db")
    
    shared_condition = build_combined_like_conditions(shared_terms)
    category_condition = f"lower(category) = '{category_name.lower()}'"

    query = f"""
    SELECT doi, date, title, abstract
    FROM papers
    WHERE {shared_condition} AND {category_condition};
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# --- Classification Logic (in Pandas) ---
def classify_papers(df):
    """
    Classifies papers into MM, ML, and Hybrid categories based on defined terms.
    Handles explicit hybrid terms and mutual exclusivity.
    """
    df_classified = df.copy()

    df_classified['is_mm'] = df_classified.apply(
        lambda row: any(term.lower() in row['title'].lower() or term.lower() in row['abstract'].lower() for term in mm_terms), axis=1
    )
    df_classified['is_ml'] = df_classified.apply(
        lambda row: any(term.lower() in row['title'].lower() or term.lower() in row['abstract'].lower() for term in ml_terms), axis=1
    )
    df_classified['is_explicit_hybrid'] = df_classified.apply(
        lambda row: any(term.lower() in row['title'].lower() or term.lower() in row['abstract'].lower() for term in explicit_hybrid), axis=1
    )

    df_classified['count_mm'] = 0
    df_classified['count_ml'] = 0
    df_classified['count_hybrid'] = 0

    for index, row in df_classified.iterrows():
        is_mm_match = row['is_mm']
        is_ml_match = row['is_ml']
        is_explicit_hybrid_match = row['is_explicit_hybrid']

        if is_explicit_hybrid_match:
            df_classified.loc[index, 'count_hybrid'] = 1
        elif is_mm_match and is_ml_match:
            df_classified.loc[index, 'count_hybrid'] = 1
        elif is_mm_match and not is_ml_match: 
            df_classified.loc[index, 'count_mm'] = 1
        elif is_ml_match and not is_mm_match: 
            df_classified.loc[index, 'count_ml'] = 1
    
    return df_classified[['date', 'count_mm', 'count_ml', 'count_hybrid']]

# --- Slope Calculation Function ---
def calculate_slopes_for_category(df_monthly, startdate="2020-01-01"):
    """
    Calculates the yearly slopes for MM, ML, and Hybrid models.
    Does NOT generate a plot.
    """
    slopes = {'mm': np.nan, 'ml': np.nan, 'hybrid': np.nan} 

    if df_monthly.empty:
        return slopes

    start_date = pd.to_datetime(startdate)
    df_recent = df_monthly[df_monthly.index >= start_date].copy() 

    if df_recent.empty:
        return slopes

    dates_recent = df_recent.index
    dates_num = mdates.date2num(dates_recent)

    def get_trend_and_r2_values(y_values):
        if len(dates_num) < 2 or len(np.unique(dates_num)) < 2:
            return None, np.nan, None, None 
        
        if np.std(y_values) == 0:
            return np.full_like(y_values, y_values[0]), 0.0, 1.0, 0.0

        coeffs = np.polyfit(dates_num, y_values, 1)
        slope_yearly = coeffs[0] * 365.25
        return None, slope_yearly, None, None 

    _, slope_mm, _, _ = get_trend_and_r2_values(df_recent["pct_mm"].values)
    _, slope_ml, _, _ = get_trend_and_r2_values(df_recent["pct_ml"].values)
    _, slope_hybrid, _, _ = get_trend_and_r2_values(df_recent["pct_hybrid"].values)

    slopes['mm'] = slope_mm
    slopes['ml'] = slope_ml
    slopes['hybrid'] = slope_hybrid
    
    return slopes 


# --- Plotting Functions ---
def plot_monthly_distribution(df_monthly, category_name):
    if df_monthly.empty:
        print(f"\nNo data to plot for monthly distribution in category: {category_name}")
        return

    dates = df_monthly.index
    pct_mm = df_monthly["pct_mm"].values
    pct_ml = df_monthly["pct_ml"].values
    pct_hybrid = df_monthly["pct_hybrid"].values

    plt.figure(figsize=(12, 6))
    plt.stackplot(dates, pct_mm, pct_ml, pct_hybrid,
                  labels=["Mechanistic Models", "Machine Learning Models", "Hybrid Models"])
    plt.xlabel("Month")
    plt.ylabel("Percentage")
    plt.title(f"Normalized Monthly Distribution of MM, ML and Hybrid Modeling Papers in {category_name}")
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def plot_correlation(df_monthly, category_name, startdate="2020-01-01"):
    if df_monthly.empty:
        print(f"\nNo data to plot for correlation in category: {category_name}")
        return

    start_date = pd.to_datetime(startdate)
    df_recent = df_monthly[df_monthly.index >= start_date].copy() 

    if df_recent.empty:
        print(f"\nNo data available from {startdate} for correlation plot in category: {category_name}")
        return

    dates_recent = df_recent.index
    dates_num = mdates.date2num(dates_recent)

    def get_trend_slope_r2(y_values):
        if len(dates_num) < 2 or len(np.unique(dates_num)) < 2:
            return None, np.nan, np.nan
        
        if np.std(y_values) == 0:
            return np.full_like(y_values, y_values[0]), 0.0, 1.0

        coeffs = np.polyfit(dates_num, y_values, 1)
        trend_func = np.poly1d(coeffs)
        trend_line = trend_func(dates_num)
        
        ss_res = np.sum((y_values - trend_line) ** 2)
        ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        
        slope_yearly = coeffs[0] * 365.25
        return trend_line, slope_yearly, r2

    trend_line_mm, slope_mm, R2_mm = get_trend_slope_r2(df_recent["pct_mm"].values)
    trend_line_ml, slope_ml, R2_ml = get_trend_slope_r2(df_recent["pct_ml"].values)
    trend_line_hybrid, slope_hybrid, R2_hybrid = get_trend_slope_r2(df_recent["pct_hybrid"].values)

    plt.figure(figsize=(12, 6))

    plt.scatter(dates_recent, df_recent["pct_mm"].values, label="Mechanistic models (Data)", alpha=0.6)
    plt.scatter(dates_recent, df_recent["pct_ml"].values, label="Machine learning models (Data)", alpha=0.6)
    plt.scatter(dates_recent, df_recent["pct_hybrid"].values, label="Hybrid models (Data)", alpha=0.6)

    if trend_line_mm is not None:
        plt.plot(dates_recent, trend_line_mm, linewidth=2,
                 label=f"Mechanistic models (slope: {slope_mm:.4f}/yr, R²: {R2_mm:.4f})", color='C0')
    if trend_line_ml is not None:
        plt.plot(dates_recent, trend_line_ml, linewidth=2,
                 label=f"Machine learning models (slope: {slope_ml:.4f}/yr, R²: {R2_ml:.4f})", color='C1')
    if trend_line_hybrid is not None:
        plt.plot(dates_recent, trend_line_hybrid, linewidth=2,
                 label=f"Hybrid models (slope: {slope_hybrid:.4f}/yr, R²: {R2_hybrid:.4f})", color='C2')
    
    plt.xlabel("Date")
    plt.ylabel("Relative Distribution (Percentage)")
    plt.title(f"Correlation Lines of Relative Distributions ({startdate}–Present) in {category_name}")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.show()

def plot_cumulative_distribution(df_monthly, category_name):
    if df_monthly.empty:
        print(f"\nNo data to plot for cumulative distribution in category: {category_name}")
        return

    df_monthly_cum = df_monthly[['count_mm', 'count_ml', 'count_hybrid']].cumsum()
    
    dates = df_monthly_cum.index
    mm_cum = df_monthly_cum["count_mm"]
    ml_cum = df_monthly_cum["count_ml"]
    hybrid_cum = df_monthly_cum["count_hybrid"]

    plt.figure(figsize=(12, 6))
    plt.stackplot(dates, mm_cum, ml_cum, hybrid_cum,
                  labels=["Mechanistic Models", "Machine Learning Models", "Hybrid Models"])
    plt.xlabel("Date")
    plt.ylabel("Cumulative Article Count")
    plt.title(f"Cumulative Distribution of MM, ML and Hybrid Modeling Papers in {category_name}")
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_slope_heatmap(slopes_df):
    if slopes_df.empty:
        print("\nNo slope data to plot heatmap.")
        return

    if 'Mechanistic Models' in slopes_df.columns:
        slopes_df = slopes_df.sort_values(by='Mechanistic Models', ascending=False)
    else:
        print("Warning: 'Mechanistic Models' column not found for sorting. Heatmap will not be sorted.")

    category_height_factor = 0.4 
    model_width_factor = 2.0
    
    heatmap_height = max(5, len(slopes_df.index) * category_height_factor) 
    heatmap_width = max(len(slopes_df.columns) * model_width_factor, 8) 

    cmap = mcolors.LinearSegmentedColormap.from_list('red_green', ['red', 'white', 'green'])

    plt.figure(figsize=(heatmap_width, heatmap_height))
    sns.heatmap(slopes_df, annot=False, fmt=".3f", cmap=cmap, center=0, linewidths=.5, cbar_kws={'label': 'Yearly Slope'})
    plt.title("Yearly Trend Slopes of Modeling Paper Percentages per Category (from 2020)")
    plt.xlabel("Model Type") 
    plt.ylabel("Category")   
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0) 
    plt.tight_layout()
    plt.show()


# --- Main Execution Logic ---
all_category_slopes = {} 

if process_all_categories:
    categories_to_process = get_filtered_categories_from_db(
        min_models_per_year=MIN_MODELS_PER_YEAR_HEATMAP, 
        min_total_papers_per_year=MIN_TOTAL_PAPERS_PER_YEAR_HEATMAP,
        filter_start_year=FILTER_START_YEAR_HEATMAP, 
        filter_end_year=FILTER_END_YEAR_HEATMAP
    )
    if not categories_to_process:
        print(f"\nNo categories found in the database matching the criteria (at least {MIN_MODELS_PER_YEAR_HEATMAP} models AND {MIN_TOTAL_PAPERS_PER_YEAR_HEATMAP} total papers per year from {FILTER_START_YEAR_HEATMAP}-{FILTER_END_YEAR_HEATMAP}).")
        print("Please check your database or adjust filtering criteria.")
else:
    categories_to_process = get_filtered_categories_from_db(
        min_models_per_year=MIN_MODELS_PER_YEAR_HEATMAP, 
        min_total_papers_per_year=MIN_TOTAL_PAPERS_PER_YEAR_HEATMAP,
        filter_start_year=FILTER_START_YEAR_HEATMAP, 
        filter_end_year=FILTER_END_YEAR_HEATMAP,
        specific_categories=defined_categories
    )
    if not categories_to_process:
        print(f"\nNone of the defined categories match the filtering criteria (at least {MIN_MODELS_PER_YEAR_HEATMAP} models AND {MIN_TOTAL_PAPERS_PER_YEAR_HEATMAP} total papers per year from {FILTER_START_YEAR_HEATMAP}-{FILTER_END_YEAR_HEATMAP}).")
        print("Please check your 'defined_categories' list or adjust filtering criteria.")

total_categories = len(categories_to_process)
if total_categories > 0:
    print(f"\nStarting processing for {total_categories} categories...")

for i, category in enumerate(categories_to_process):
    print(f"--- Processing category {i + 1}/{total_categories}: {category} ---                        ", end="\r")
    
    df_raw = fetch_papers_for_category(category)
    
    if df_raw.empty:
        print(f"\rNo papers found for category: {category}. Skipping plots.                               ") 
        continue

    df_raw["date"] = pd.to_datetime(df_raw["date"])

    df_classified_counts = classify_papers(df_raw)

    min_date = df_classified_counts['date'].min()
    max_date = df_classified_counts['date'].max()

    if pd.isna(min_date) or pd.isna(max_date):
        print(f"\rInvalid date range for category: {category}. Skipping plots.                               ") 
        continue

    df_daily_counts = df_classified_counts.groupby('date').sum().reindex(
        pd.date_range(start=min_date, end=max_date, freq='D'),
        fill_value=0
    )
    df_daily_counts.index.name = "date"

    df_monthly = df_daily_counts.resample("M").sum()

    df_monthly["total"] = df_monthly["count_mm"] + df_monthly["count_ml"] + df_monthly["count_hybrid"]
    df_monthly["total"].replace(0, 1, inplace=True) 
    df_monthly["pct_mm"] = df_monthly["count_mm"] / df_monthly["total"]
    df_monthly["pct_ml"] = df_monthly["count_ml"] / df_monthly["total"]
    df_monthly["pct_hybrid"] = df_monthly["count_hybrid"] / df_monthly["total"]
    
    category_slopes = calculate_slopes_for_category(df_monthly, startdate=f"{FILTER_START_YEAR_HEATMAP}-01-01")
    all_category_slopes[category] = category_slopes

    if make_category_plots:
        print("\r" + " " * 100, end="\r") 
        plot_monthly_distribution(df_monthly, category)
        plot_correlation(df_monthly, category, startdate=f"{FILTER_START_YEAR_HEATMAP}-01-01") 
        plot_cumulative_distribution(df_monthly, category)
        if i + 1 < total_categories:
             print(f"--- Processing category {i + 1}/{total_categories}: {category} ---", end="\r")


print("\r" + " " * 100 + "\r") 
print("--- All categories processed. Generating heatmap ---")

slopes_df = pd.DataFrame.from_dict(all_category_slopes, orient='index')
slopes_df.index.name = 'Category'
slopes_df.columns = ['Mechanistic Models', 'Machine Learning Models', 'Hybrid Models']

if not slopes_df.empty:
    plot_slope_heatmap(slopes_df)
else:
    print("\nNo valid categories found to generate the heatmap. Please check your data and filtering criteria.")

# %%
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.colors as mcolors
import threading
import queue
import os # To check for database file existence

# --- Helper Functions for SQL Query Building ---
def build_like_conditions(terms, field):
    if not terms or (len(terms) == 1 and terms[0].strip() == ""):
        return "1=1"
    return " OR ".join([f"lower({field}) LIKE '%{term.lower()}%'" for term in terms])

def build_combined_like_conditions(terms):
    title_conditions = build_like_conditions(terms, 'title')
    abstract_conditions = build_like_conditions(terms, 'abstract')

    if title_conditions == "1=1" and abstract_conditions == "1=1":
        return "1=1"
    elif title_conditions == "1=1":
        return abstract_conditions
    elif abstract_conditions == "1=1":
        return title_conditions
    else:
        return f"({title_conditions}) OR ({abstract_conditions})"

# --- Define Search Term Lists (Fixed terms, shared_terms will be user input) ---
mm_terms = ["mechanistic", "kinetic", "differential equation", "simulation", "first principles",
            "physics-based model","computational model", "systems biology", "action kinetics",
            "biochemical network", "dynamic model", "mathematical model", "stochastic model",
            "agent-based model", "cellular automata", "flux balance analysis"]

ml_terms = ["machine learning", "deep learning", "neural network", "artificial intelligence",
            "predictive model", "data-driven", "supervised learning","unsupervised learning",
            "reinforcement learning","ensemble learning","transfer learning","feature extraction",
            "pattern recognition","support vector machine", "random forest","gradient boosting"]

explicit_hybrid = ["hybrid model", "mechanistic learning"]


# --- Database Interaction Functions ---
def get_all_categories_from_db(db_path):
    """Fetches all unique categories from the database."""
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    # Corrected column selection
    query = "SELECT DISTINCT lower(category) FROM papers WHERE category IS NOT NULL AND category != '' ORDER BY lower(category);"
    # The column name from the query will be 'lower(category)'
    categories = pd.read_sql_query(query, conn)["lower(category)"].tolist()
    conn.close()
    return categories

def get_filtered_categories_from_db(db_path, min_models_per_year, min_total_papers_per_year,
                                     filter_start_year, filter_end_year, specific_categories=None, output_queue=None):
    if output_queue:
        output_queue.put(f"Fetching papers for category filtering from {filter_start_year} to {filter_end_year}...")
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT doi, date, title, abstract, lower(category) as category
    FROM papers
    WHERE category IS NOT NULL AND category != ''
      AND STRFTIME('%Y', date) >= '{filter_start_year}'
      AND STRFTIME('%Y', date) <= '{filter_end_year}';
    """
    df_all_papers = pd.read_sql_query(query, conn)
    conn.close()

    if df_all_papers.empty:
        if output_queue:
            output_queue.put("No papers found in the specified date range for category filtering.")
        return []

    df_all_papers['date'] = pd.to_datetime(df_all_papers['date'])
    df_all_papers['year'] = df_all_papers['date'].dt.year

    if specific_categories:
        if output_queue:
            output_queue.put(f"Filtering for specific categories: {', '.join(specific_categories)}...")
        df_all_papers = df_all_papers[df_all_papers['category'].isin([c.lower() for c in specific_categories])]
        if df_all_papers.empty:
            if output_queue:
                output_queue.put("No papers found for the selected specific categories within the date range.")
            return []

    eligible_categories = []
    unique_categories_to_check = df_all_papers['category'].unique()
    if output_queue:
        output_queue.put(f"Checking {len(unique_categories_to_check)} unique categories against filtering criteria...")

    for category_name in unique_categories_to_check:
        if output_queue:
            output_queue.put(f"  Checking category '{category_name}'...")
        df_cat = df_all_papers[df_all_papers['category'] == category_name].copy()
        df_classified_counts = classify_papers(df_cat, output_queue=output_queue) # Pass queue
        df_classified_counts['year'] = df_classified_counts['date'].dt.year

        yearly_model_counts = df_classified_counts.groupby('year')[['count_mm', 'count_ml', 'count_hybrid']].sum().reset_index()
        yearly_model_counts['total_models_classified'] = yearly_model_counts['count_mm'] + yearly_model_counts['count_ml'] + yearly_model_counts['count_hybrid']

        yearly_total_papers = df_cat.groupby('year').size().reset_index(name='total_papers')

        yearly_combined_counts = pd.merge(yearly_total_papers, yearly_model_counts, on='year', how='left').fillna(0)

        meets_criteria = True
        required_years = set(range(filter_start_year, filter_end_year + 1))
        years_present_in_data = set(yearly_combined_counts['year'])

        if not required_years.issubset(years_present_in_data):
            meets_criteria = False
        else:
            for year in required_years:
                year_data = yearly_combined_counts[yearly_combined_counts['year'] == year]
                if year_data.empty:
                    meets_criteria = False
                    break

                total_models_in_year = year_data['total_models_classified'].iloc[0]
                total_papers_in_year = year_data['total_papers'].iloc[0]

                if total_models_in_year < min_models_per_year or total_papers_in_year < min_total_papers_per_year:
                    meets_criteria = False
                    break
        if meets_criteria:
            eligible_categories.append(category_name)
            if output_queue:
                output_queue.put(f"    Category '{category_name}' meets all criteria and is eligible.")
        else:
            if output_queue:
                output_queue.put(f"    Category '{category_name}' does NOT meet all criteria.")

    if output_queue:
        output_queue.put(f"Finished filtering categories. Found {len(eligible_categories)} eligible categories.")
    return eligible_categories


def fetch_papers_for_category(db_path, category_name, current_shared_terms, output_queue=None):
    if output_queue:
        output_queue.put(f"  Fetching papers for category '{category_name}' with shared terms: '{', '.join(current_shared_terms)}'...")
    conn = sqlite3.connect(db_path)
    shared_condition = build_combined_like_conditions(current_shared_terms)
    category_condition = f"lower(category) = '{category_name.lower()}'"

    query = f"""
    SELECT doi, date, title, abstract
    FROM papers
    WHERE {shared_condition} AND {category_condition};
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    if output_queue:
        output_queue.put(f"  Fetched {len(df)} papers for category '{category_name}'.")
    return df

# --- Classification Logic ---
def classify_papers(df, output_queue=None):
    if output_queue:
        output_queue.put(f"  Classifying {len(df)} papers...")
    df_classified = df.copy()

    df_classified['is_mm'] = df_classified.apply(
        lambda row: any(term.lower() in row['title'].lower() or term.lower() in row['abstract'].lower() for term in mm_terms), axis=1
    )
    df_classified['is_ml'] = df_classified.apply(
        lambda row: any(term.lower() in row['title'].lower() or term.lower() in row['abstract'].lower() for term in ml_terms), axis=1
    )
    df_classified['is_explicit_hybrid'] = df_classified.apply(
        lambda row: any(term.lower() in row['title'].lower() or term.lower() in row['abstract'].lower() for term in explicit_hybrid), axis=1
    )

    df_classified['count_mm'] = 0
    df_classified['count_ml'] = 0
    df_classified['count_hybrid'] = 0

    for index, row in df_classified.iterrows():
        is_mm_match = row['is_mm']
        is_ml_match = row['is_ml']
        is_explicit_hybrid_match = row['is_explicit_hybrid']

        if is_explicit_hybrid_match:
            df_classified.loc[index, 'count_hybrid'] = 1
        elif is_mm_match and is_ml_match:
            df_classified.loc[index, 'count_hybrid'] = 1
        elif is_mm_match and not is_ml_match:
            df_classified.loc[index, 'count_mm'] = 1
        elif is_ml_match and not is_mm_match:
            df_classified.loc[index, 'count_ml'] = 1
    if output_queue:
        output_queue.put("  Paper classification complete.")
    return df_classified[['date', 'count_mm', 'count_ml', 'count_hybrid']]

# --- Slope Calculation Function ---
def calculate_slopes_for_category(df_monthly, startdate, output_queue=None):
    if output_queue:
        output_queue.put(f"  Calculating slopes from {startdate}...")
    slopes = {'mm': np.nan, 'ml': np.nan, 'hybrid': np.nan}

    if df_monthly.empty:
        if output_queue: output_queue.put("  No monthly data for slope calculation.")
        return slopes

    start_date = pd.to_datetime(startdate)
    df_recent = df_monthly[df_monthly.index >= start_date].copy()

    if df_recent.empty:
        if output_queue: output_queue.put(f"  No recent data from {startdate} for slope calculation.")
        return slopes

    dates_recent = df_recent.index
    dates_num = mdates.date2num(dates_recent)

    def get_trend_and_r2_values(y_values):
        if len(dates_num) < 2 or len(np.unique(dates_num)) < 2:
            return None, np.nan, None, None
        if np.std(y_values) == 0:
            return np.full_like(y_values, y_values[0]), 0.0, 1.0, 0.0

        coeffs = np.polyfit(dates_num, y_values, 1)
        slope_yearly = coeffs[0] * 365.25
        return None, slope_yearly, None, None

    _, slope_mm, _, _ = get_trend_and_r2_values(df_recent["pct_mm"].values)
    _, slope_ml, _, _ = get_trend_and_r2_values(df_recent["pct_ml"].values)
    _, slope_hybrid, _, _ = get_trend_and_r2_values(df_recent["pct_hybrid"].values)

    slopes['mm'] = slope_mm
    slopes['ml'] = slope_ml
    slopes['hybrid'] = slope_hybrid
    if output_queue:
        output_queue.put("  Slopes calculated.")
    return slopes

# --- Plotting Functions ---
def plot_monthly_distribution(df_monthly, category_name):
    if df_monthly.empty:
        print(f"\nNo data to plot for monthly distribution in category: {category_name}")
        return

    dates = df_monthly.index
    pct_mm = df_monthly["pct_mm"].values
    pct_ml = df_monthly["pct_ml"].values
    pct_hybrid = df_monthly["pct_hybrid"].values

    plt.figure(figsize=(12, 6))
    plt.stackplot(dates, pct_mm, pct_ml, pct_hybrid,
                  labels=["Mechanistic Models", "Machine Learning Models", "Hybrid Models"])
    plt.xlabel("Month")
    plt.ylabel("Percentage")
    plt.title(f"Normalized Monthly Distribution of MM, ML and Hybrid Modeling Papers in {category_name}")
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def plot_correlation(df_monthly, category_name, startdate):
    if df_monthly.empty:
        print(f"\nNo data to plot for correlation in category: {category_name}")
        return

    start_date = pd.to_datetime(startdate)
    df_recent = df_monthly[df_monthly.index >= start_date].copy()

    if df_recent.empty:
        print(f"\nNo data available from {startdate} for correlation plot in category: {category_name}")
        return

    dates_recent = df_recent.index
    dates_num = mdates.date2num(dates_recent)

    def get_trend_slope_r2(y_values):
        if len(dates_num) < 2 or len(np.unique(dates_num)) < 2:
            return None, np.nan, np.nan

        if np.std(y_values) == 0:
            return np.full_like(y_values, y_values[0]), 0.0, 1.0

        coeffs = np.polyfit(dates_num, y_values, 1)
        trend_func = np.poly1d(coeffs)
        trend_line = trend_func(dates_num)

        ss_res = np.sum((y_values - trend_line) ** 2)
        ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        slope_yearly = coeffs[0] * 365.25
        return trend_line, slope_yearly, r2

    trend_line_mm, slope_mm, R2_mm = get_trend_slope_r2(df_recent["pct_mm"].values)
    trend_line_ml, slope_ml, R2_ml = get_trend_slope_r2(df_recent["pct_ml"].values)
    trend_line_hybrid, slope_hybrid, R2_hybrid = get_trend_slope_r2(df_recent["pct_hybrid"].values)

    plt.figure(figsize=(12, 6))

    plt.scatter(dates_recent, df_recent["pct_mm"].values, label="Mechanistic models (Data)", alpha=0.6)
    plt.scatter(dates_recent, df_recent["pct_ml"].values, label="Machine learning models (Data)", alpha=0.6)
    plt.scatter(dates_recent, df_recent["pct_hybrid"].values, label="Hybrid models (Data)", alpha=0.6)

    if trend_line_mm is not None:
        plt.plot(dates_recent, trend_line_mm, linewidth=2,
                 label=f"Mechanistic models (slope: {slope_mm:.4f}/yr, R²: {R2_mm:.4f})", color='C0')
    if trend_line_ml is not None:
        plt.plot(dates_recent, trend_line_ml, linewidth=2,
                 label=f"Machine learning models (slope: {slope_ml:.4f}/yr, R²: {R2_ml:.4f})", color='C1')
    if trend_line_hybrid is not None:
        plt.plot(dates_recent, trend_line_hybrid, linewidth=2,
                 label=f"Hybrid models (slope: {slope_hybrid:.4f}/yr, R²: {R2_hybrid:.4f})", color='C2')

    plt.xlabel("Date")
    plt.ylabel("Relative Distribution (Percentage)")
    plt.title(f"Correlation Lines of Relative Distributions ({startdate}–Present) in {category_name}")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.show()

def plot_cumulative_distribution(df_monthly, category_name):
    if df_monthly.empty:
        print(f"\nNo data to plot for cumulative distribution in category: {category_name}")
        return

    df_monthly_cum = df_monthly[['count_mm', 'count_ml', 'count_hybrid']].cumsum()

    dates = df_monthly_cum.index
    mm_cum = df_monthly_cum["count_mm"]
    ml_cum = df_monthly_cum["count_ml"]
    hybrid_cum = df_monthly_cum["count_hybrid"]

    plt.figure(figsize=(12, 6))
    plt.stackplot(dates, mm_cum, ml_cum, hybrid_cum,
                  labels=["Mechanistic Models", "Machine Learning Models", "Hybrid Models"])
    plt.xlabel("Date")
    plt.ylabel("Cumulative Article Count")
    plt.title(f"Cumulative Distribution of MM, ML and Hybrid Modeling Papers in {category_name}")
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_slope_heatmap(slopes_df):
    if slopes_df.empty:
        print("\nNo slope data to plot heatmap.")
        return

    if 'Mechanistic Models' in slopes_df.columns:
        slopes_df = slopes_df.sort_values(by='Mechanistic Models', ascending=False)
    else:
        print("Warning: 'Mechanistic Models' column not found for sorting. Heatmap will not be sorted.")

    category_height_factor = 0.4
    model_width_factor = 2.0

    heatmap_height = max(5, len(slopes_df.index) * category_height_factor)
    heatmap_width = max(len(slopes_df.columns) * model_width_factor, 8)

    cmap = mcolors.LinearSegmentedColormap.from_list('red_green', ['red', 'white', 'green'])

    plt.figure(figsize=(heatmap_width, heatmap_height))
    sns.heatmap(slopes_df, annot=True, fmt=".3f", cmap=cmap, center=0, linewidths=.5, cbar_kws={'label': 'Yearly Slope'})
    plt.title("Yearly Trend Slopes of Modeling Paper Percentages per Category (from 2020)")
    plt.xlabel("Model Type")
    plt.ylabel("Category")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bioinformatics Paper Analysis")
        self.geometry("900x700")

        # Set default path
        self.db_path = tk.StringVar(value="C:/Users/herme/Desktop/writing_assignment/bioinformatics_papers.db")
        self.process_all_categories = tk.BooleanVar(value=True)
        self.make_category_plots = tk.BooleanVar(value=False)
        self.shared_terms_str = tk.StringVar(value="")

        self.min_models_per_year_heatmap = tk.IntVar(value=20)
        self.min_total_papers_per_year_heatmap = tk.IntVar(value=20)
        self.filter_start_year_heatmap = tk.IntVar(value=2020)
        self.filter_end_year_heatmap = tk.IntVar(value=2025)

        self.categories_in_db = []

        self.create_widgets()
        self.output_queue = queue.Queue()
        self.after(100, self.process_queue)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initial load of categories after GUI is set up
        self.after(50, self.load_categories_from_db)

    def create_widgets(self):
        # Database selection
        db_frame = ttk.LabelFrame(self, text="Database Settings", padding="10")
        db_frame.pack(pady=10, padx=10, fill="x")

        ttk.Label(db_frame, text="Database Path:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.db_entry = ttk.Entry(db_frame, textvariable=self.db_path, width=50)
        self.db_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(db_frame, text="Browse", command=self.browse_db_path).grid(row=0, column=2, padx=5, pady=5)
        db_frame.grid_columnconfigure(1, weight=1)

        # Shared Terms Input
        shared_terms_frame = ttk.LabelFrame(self, text="Global Search Terms (Shared Terms)", padding="10")
        shared_terms_frame.pack(pady=10, padx=10, fill="x")
        ttk.Label(shared_terms_frame, text="Comma-separated keywords (e.g., 'model, analysis'):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.shared_terms_entry = ttk.Entry(shared_terms_frame, textvariable=self.shared_terms_str, width=80)
        self.shared_terms_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        shared_terms_frame.grid_columnconfigure(1, weight=1)

        # Category processing options
        category_frame = ttk.LabelFrame(self, text="Category Selection", padding="10")
        category_frame.pack(pady=10, padx=10, fill="both", expand=True)

        ttk.Radiobutton(category_frame, text="Process All Eligible Categories", variable=self.process_all_categories, value=True, command=self.toggle_category_selection).grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(category_frame, text="Select Specific Categories from List (Ctrl+Click for multiple)", variable=self.process_all_categories, value=False, command=self.toggle_category_selection).grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        ttk.Label(category_frame, text="Available Categories:").grid(row=2, column=0, padx=5, pady=5, sticky="nw")

        self.category_listbox = tk.Listbox(category_frame, selectmode="multiple", height=8)
        self.category_listbox.grid(row=2, column=1, padx=5, pady=5, sticky="nsew")
        self.category_listbox_scrollbar = ttk.Scrollbar(category_frame, orient="vertical", command=self.category_listbox.yview)
        self.category_listbox_scrollbar.grid(row=2, column=2, sticky="ns")
        self.category_listbox.config(yscrollcommand=self.category_listbox_scrollbar.set)

        category_frame.grid_columnconfigure(1, weight=1)
        category_frame.grid_rowconfigure(2, weight=1)

        self.toggle_category_selection() # Set initial state for listbox (disabled if process all)

        # Filtering parameters for heatmap
        filter_frame = ttk.LabelFrame(self, text="Heatmap Filtering Criteria", padding="10")
        filter_frame.pack(pady=10, padx=10, fill="x")

        ttk.Label(filter_frame, text="Min Models Per Year:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(filter_frame, from_=0, to=1000, textvariable=self.min_models_per_year_heatmap, width=10).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(filter_frame, text="Min Total Papers Per Year:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(filter_frame, from_=0, to=1000, textvariable=self.min_total_papers_per_year_heatmap, width=10).grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(filter_frame, text="Filter Start Year:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        ttk.Spinbox(filter_frame, from_=1990, to=2030, textvariable=self.filter_start_year_heatmap, width=10).grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        ttk.Label(filter_frame, text="Filter End Year:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        ttk.Spinbox(filter_frame, from_=1990, to=2030, textvariable=self.filter_end_year_heatmap, width=10).grid(row=1, column=3, padx=5, pady=5, sticky="ew")

        # Plotting options
        plot_options_frame = ttk.LabelFrame(self, text="Plotting Options", padding="10")
        plot_options_frame.pack(pady=10, padx=10, fill="x")

        ttk.Checkbutton(plot_options_frame, text="Generate Individual Category Plots", variable=self.make_category_plots).pack(padx=5, pady=5, anchor="w")

        # Action buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Generate Plots", command=self.start_processing).pack(side="left", padx=10)
        ttk.Button(button_frame, text="Reset Settings", command=self.reset_settings).pack(side="left", padx=10)
        ttk.Button(button_frame, text="Refresh Categories", command=self.load_categories_from_db).pack(side="left", padx=10)


        # Status output
        self.status_label = ttk.Label(self, text="Status: Ready", wraplength=800)
        self.status_label.pack(pady=5)

        self.output_text = tk.Text(self, height=8, state="disabled", wrap="word")
        self.output_text.pack(pady=10, padx=10, fill="both", expand=True)

    def browse_db_path(self):
        file_path = filedialog.askopenfilename(
            title="Select Database File",
            filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")]
        )
        if file_path:
            self.db_path.set(file_path)
            self.load_categories_from_db() # Reload categories if DB path changes

    def toggle_category_selection(self):
        # Enable listbox first to allow updates, then disable if "Process All" is checked
        if self.process_all_categories.get():
            self.category_listbox.config(state="disabled")
        else:
            self.category_listbox.config(state="normal")

    def load_categories_from_db(self):
        db_path = self.db_path.get()
        self.output_queue.put(f"Attempting to load categories from database: {db_path}...")
        self.status_label.config(text="Status: Loading categories...")
        try:
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database file not found at: {db_path}")

            # Temporarily enable listbox to allow insertion of categories
            self.category_listbox.config(state="normal")
            self.category_listbox.delete(0, tk.END) # Clear existing listbox items

            categories = get_all_categories_from_db(db_path)
            self.categories_in_db = sorted(categories)

            if self.categories_in_db:
                for cat in self.categories_in_db:
                    self.category_listbox.insert(tk.END, cat)
                self.output_queue.put(f"Successfully loaded {len(self.categories_in_db)} categories into the list.")
            else:
                self.output_queue.put("No categories found in the database or categories table is empty.")
            self.status_label.config(text="Status: Categories loaded.")
        except Exception as e_captured:
            self.output_queue.put(f"Error loading categories: {e_captured}")
            self.after(0, lambda err=e_captured: messagebox.showerror("Database Error", f"Could not load categories: {err}\nCheck database path and file."))
            self.status_label.config(text="Status: Error loading categories.")
        finally:
            # Re-apply the desired state (disabled if 'Process All' is checked)
            self.toggle_category_selection()

    def reset_settings(self):
        # Reset to the new default path
        self.db_path.set("C:/Users/herme/Desktop/writing_assignment/bioinformatics_papers.db")
        self.process_all_categories.set(True)
        self.make_category_plots.set(False)
        self.shared_terms_str.set("")
        self.min_models_per_year_heatmap.set(20)
        self.min_total_papers_per_year_heatmap.set(20)
        self.filter_start_year_heatmap.set(2020)
        self.filter_end_year_heatmap.set(2025)
        self.toggle_category_selection()
        self.update_output("Settings reset to defaults.")
        self.load_categories_from_db() # Reload categories after reset to reflect new DB path

    def update_output(self, message):
        self.output_text.config(state="normal")
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.output_text.config(state="disabled")

    def process_queue(self):
        try:
            while True:
                message = self.output_queue.get_nowait()
                self.update_output(message)
        except queue.Empty:
            pass
        if self.winfo_exists():
             self.after(100, self.process_queue)

    def start_processing(self):
        self.output_text.config(state="normal")
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state="disabled")
        self.status_label.config(text="Status: Initializing processing...")

        db_path = self.db_path.get()
        process_all = self.process_all_categories.get()
        make_plots = self.make_category_plots.get()
        shared_terms_raw = self.shared_terms_str.get()

        min_models = self.min_models_per_year_heatmap.get()
        min_papers = self.min_total_papers_per_year_heatmap.get( )
        start_year = self.filter_start_year_heatmap.get()
        end_year = self.filter_end_year_heatmap.get()

        if not db_path:
            messagebox.showerror("Input Error", "Please specify a database path.")
            self.status_label.config(text="Status: Error")
            return
        if not os.path.exists(db_path):
            messagebox.showerror("File Error", f"Database file not found at: {db_path}")
            self.status_label.config(text="Status: Error")
            return

        selected_categories = []
        if not process_all:
            selected_indices = self.category_listbox.curselection()
            if not selected_indices:
                messagebox.showerror("Input Error", "Please select at least one category from the list, or choose 'Process All Eligible Categories'.")
                self.status_label.config(text="Status: Error")
                return
            for i in selected_indices:
                selected_categories.append(self.category_listbox.get(i))
            if not selected_categories:
                messagebox.showerror("Input Error", "Selected categories list is empty. Please select valid categories.")
                self.status_label.config(text="Status: Error")
                return

        if start_year > end_year:
            messagebox.showerror("Input Error", "Filter Start Year cannot be greater than Filter End Year.")
            self.status_label.config(text="Status: Error")
            return

        current_shared_terms = [term.strip() for term in shared_terms_raw.split(',') if term.strip()]
        if not current_shared_terms:
            current_shared_terms = [""]

        # Run the main processing logic in a separate thread
        self.status_label.config(text="Status: Starting analysis thread...")
        threading.Thread(target=self._run_analysis, args=(
            db_path, process_all, make_plots, selected_categories,
            min_models, min_papers, start_year, end_year, current_shared_terms
        )).start()

    def _run_analysis(self, db_path, process_all_categories, make_category_plots,
                      selected_categories_from_list, min_models_per_year_heatmap,
                      min_total_papers_per_year_heatmap, filter_start_year_heatmap,
                      filter_end_year_heatmap, current_shared_terms):
        try:
            self.output_queue.put("\n--- Starting full data analysis process ---")
            self.status_label.config(text="Status: Performing data analysis...")
            all_category_slopes = {}

            categories_to_process = []
            if process_all_categories:
                self.output_queue.put("Determining eligible categories (all mode)...")
                self.status_label.config(text="Status: Filtering all categories...")
                categories_to_process = get_filtered_categories_from_db(
                    db_path,
                    min_models_per_year=min_models_per_year_heatmap,
                    min_total_papers_per_year=min_total_papers_per_year_heatmap,
                    filter_start_year=filter_start_year_heatmap,
                    filter_end_year=filter_end_year_heatmap,
                    output_queue=self.output_queue
                )
                if not categories_to_process:
                    self.output_queue.put(f"No categories found in the database matching the criteria (at least {min_models_per_year_heatmap} models AND {min_total_papers_per_year_heatmap} total papers per year from {filter_start_year_heatmap}-{filter_end_year_heatmap}).")
                    self.output_queue.put("Please check your database or adjust filtering criteria.")
                    self.after(0, lambda: self.status_label.config(text="Status: Finished with warnings."))
                    return
            else:
                self.output_queue.put(f"Determining eligible categories (selected mode): {', '.join(selected_categories_from_list)}...")
                self.status_label.config(text="Status: Filtering selected categories...")
                categories_to_process = get_filtered_categories_from_db(
                    db_path,
                    min_models_per_year=min_models_per_year_heatmap,
                    min_total_papers_per_year=min_total_papers_per_year_heatmap,
                    filter_start_year=filter_start_year_heatmap,
                    filter_end_year=filter_end_year_heatmap,
                    specific_categories=selected_categories_from_list,
                    output_queue=self.output_queue
                )
                if not categories_to_process:
                    self.output_queue.put(f"None of the selected categories match the filtering criteria (at least {min_models_per_year_heatmap} models AND {min_total_papers_per_year_heatmap} total papers per year from {filter_start_year_heatmap}-{filter_end_year_heatmap}).")
                    self.output_queue.put("Please check your selection or adjust filtering criteria.")
                    self.after(0, lambda: self.status_label.config(text="Status: Finished with warnings."))
                    return

            total_categories = len(categories_to_process)
            if total_categories > 0:
                self.output_queue.put(f"\nProceeding with {total_categories} eligible categories...")

            for i, category in enumerate(categories_to_process):
                self.output_queue.put(f"\n--- Processing category {i + 1}/{total_categories}: {category} ---")
                self.status_label.config(text=f"Status: Processing category {i + 1}/{total_categories} - {category}...")

                df_raw = fetch_papers_for_category(db_path, category, current_shared_terms, output_queue=self.output_queue)

                if df_raw.empty:
                    self.output_queue.put(f"No papers found for category: {category} with the given shared terms. Skipping further analysis for this category.")
                    continue

                df_raw["date"] = pd.to_datetime(df_raw["date"])
                df_classified_counts = classify_papers(df_raw, output_queue=self.output_queue) # Pass queue

                min_date = df_classified_counts['date'].min()
                max_date = df_classified_counts['date'].max()

                if pd.isna(min_date) or pd.isna(max_date):
                    self.output_queue.put(f"  Invalid date range for category: {category}. Skipping plots for this category.")
                    continue

                self.output_queue.put("  Resampling data to monthly frequency...")
                df_daily_counts = df_classified_counts.groupby('date').sum().reindex(
                    pd.date_range(start=min_date, end=max_date, freq='D'),
                    fill_value=0
                )
                df_daily_counts.index.name = "date"

                df_monthly = df_daily_counts.resample("M").sum()
                df_monthly["total"] = df_monthly["count_mm"] + df_monthly["count_ml"] + df_monthly["count_hybrid"]
                df_monthly["total"].replace(0, 1, inplace=True)
                df_monthly["pct_mm"] = df_monthly["count_mm"] / df_monthly["total"]
                df_monthly["pct_ml"] = df_monthly["count_ml"] / df_monthly["total"]
                df_monthly["pct_hybrid"] = df_monthly["count_hybrid"] / df_monthly["total"]
                self.output_queue.put("  Monthly data preparation complete.")


                category_slopes = calculate_slopes_for_category(df_monthly, startdate=f"{filter_start_year_heatmap}-01-01", output_queue=self.output_queue) # Pass queue
                all_category_slopes[category] = category_slopes

                if make_category_plots:
                    self.output_queue.put(f"  Generating individual plots for {category}...")
                    self.after(0, lambda df=df_monthly, cat=category, sy=f"{filter_start_year_heatmap}-01-01":
                               (plot_monthly_distribution(df, cat),
                                plot_correlation(df, cat, sy),
                                plot_cumulative_distribution(df, cat),
                                plt.close('all')))
                    self.output_queue.put(f"  Finished individual plots for {category}.")

            self.output_queue.put("\n--- All categories processed. Generating overall heatmap ---")
            self.status_label.config(text="Status: Generating final heatmap...")

            slopes_df = pd.DataFrame.from_dict(all_category_slopes, orient='index')
            slopes_df.index.name = 'Category'
            slopes_df.columns = ['Mechanistic Models', 'Machine Learning Models', 'Hybrid Models']

            if not slopes_df.empty:
                self.after(0, lambda df=slopes_df: (plot_slope_heatmap(df), plt.close('all')))
                self.output_queue.put("Overall heatmap generated.")
            else:
                self.output_queue.put("\nNo valid categories found to generate the heatmap. Please check your data and filtering criteria.")

            self.output_queue.put("\n--- Analysis complete! ---")
            self.after(0, lambda: self.status_label.config(text="Status: Done"))

        except Exception as e_captured:
            self.output_queue.put(f"An error occurred during analysis: {e_captured}")
            self.after(0, lambda: self.status_label.config(text="Status: Error"))
            self.after(0, lambda err=e_captured: messagebox.showerror("Error", f"An unexpected error occurred during analysis: {err}"))

    def on_closing(self):
        print("Closing application...")
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
# %%
