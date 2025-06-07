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
