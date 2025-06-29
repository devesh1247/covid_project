# IMPORTS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# CONFIGURATION
DATA_PATH    = 'data/COVID clinical trials.csv'  
OUTPUT_DIR   = 'outputs/figures'                 
CLEANED_PATH = 'outputs/cleaned_trials.csv'     
RANDOM_SEED  = 123                               

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

# LOAD & INITIAL INSPECTION

df = pd.read_csv(DATA_PATH)
print("1) Data loaded:")
print("  • rows:", df.shape[0], "columns:", df.shape[1])
print(df.info(), "\n")

# Show a glimpse
print("First 5 rows:")
print(df.head(), "\n")



# DROP OVERLY-SPARSE COLUMNS
# Columns with >80% missing values are dropped
missing_pct = df.isnull().mean()
to_drop = missing_pct[missing_pct > 0.80].index.tolist()
print("2) Dropping columns with >80% missing:", to_drop)
df.drop(columns=to_drop, inplace=True)



# HANDLE MODERATELY MISSING CATEGORICALS

# Fill remaining NA in these string fields with 'Unknown'
for col in ['Acronym', 'Phases', 'Interventions']:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')
        print(f"  • Imputed '{col}' missing → 'Unknown'")

# DATE PARSING & FEATURE EXTRACTION

# Convert date strings to datetime, then extract year/month
for date_col in ['Start Date', 'Primary Completion Date', 'Completion Date']:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[f'{date_col} Year'] = df[date_col].dt.year
        df[f'{date_col} Month'] = df[date_col].dt.month
        print(f"4) Parsed and extracted year/month from '{date_col}'")


# NUMERIC COLUMN CLEANUP

if 'Enrollment' in df.columns:
    df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
    
    df.loc[df['Enrollment'] < 0, 'Enrollment'] = np.nan
    enroll_median = int(df['Enrollment'].median())
    df['Enrollment'] = df['Enrollment'].fillna(enroll_median)
    print(f"5) Cleaned 'Enrollment', imputed missing with median = {enroll_median}")



# UNIVARIATE ANALYSIS (VALUE COUNTS & BARCHARTS)
# Helper to save and close figures
def save_plot(name):
    path = os.path.join(OUTPUT_DIR, name + '.png')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"   • Saved plot: {path}")

# Study Status distribution
plt.figure(figsize=(6,4))
df['Status'].value_counts().plot(kind='bar')
plt.title('Study Status Distribution')
save_plot('univariate_status')

# Phases distribution
if 'Phases' in df.columns:
    plt.figure(figsize=(6,4))
    df['Phases'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Clinical Trial Phases')
    save_plot('univariate_phases')

#  Trial Design types
if 'Study Designs' in df.columns:
    plt.figure(figsize=(6,4))
    df['Study Designs'].value_counts().nlargest(10).plot(kind='barh')
    plt.title('Top 10 Study Designs')
    save_plot('univariate_designs')



# BIVARIATE ANALYSIS
# Status vs Phase
if {'Status','Phases'}.issubset(df.columns):
    cross = pd.crosstab(df['Phases'], df['Status'], normalize='index')
    cross.plot(kind='bar', stacked=True, figsize=(7,5))
    plt.title('Study Phases vs Status (proportion)')
    save_plot('bivariate_phase_status')



# TIME-SERIES TREND
# Monthly count of trials started
if 'Start Date Year' in df.columns:
    monthly = df.groupby(['Start Date Year','Start Date Month']).size().reset_index(name='Count')
    monthly = monthly.rename(columns={'Start Date Year': 'year', 'Start Date Month': 'month'})
    monthly['day'] = 1
    monthly['Date'] = pd.to_datetime(monthly[['year', 'month', 'day']])
    plt.figure(figsize=(8,4))
    plt.plot(monthly['Date'], monthly['Count'], marker='o')
    plt.title('Monthly Number of COVID-19 Trial Starts')
    plt.xlabel('Date')
    plt.ylabel('Number of Trials')
    save_plot('timeseries_monthly_trials')



# FEATURE ENGINEERING (COUNTRY EXTRACTION)
# If 'Locations' column exists, extract first country seen
if 'Locations' in df.columns:
    def extract_country(loc):
        return loc.split(';')[0].split(',')[-1].strip()
    df['Country'] = df['Locations'].fillna('').apply(extract_country)
    top_countries = df['Country'].value_counts().nlargest(10)
    plt.figure(figsize=(6,4))
    top_countries.plot(kind='bar')
    plt.title('Top 10 Countries by Trial Count')
    save_plot('feature_country_counts')



#  TOP 10 INTERVENTIONS
if 'Interventions' in df.columns:
    top_interventions = df['Interventions'].str.split('|').explode().value_counts().nlargest(10)
    plt.figure(figsize=(8,4))
    top_interventions.plot(kind='bar', color='teal')
    plt.title('Top 10 Interventions in COVID-19 Trials')
    plt.ylabel('Number of Trials')
    save_plot('top_interventions')

# SAVE CLEANED DATA
df.to_csv(CLEANED_PATH, index=False)
print(f"10) Cleaned dataset saved to {CLEANED_PATH}")

print("11) EDA complete — all figures are in", OUTPUT_DIR)
print("    You can now proceed to deeper analysis or model building.")
