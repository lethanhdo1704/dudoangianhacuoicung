"""
Extract locations and choices from processed housing data
Creates locations.json and choices.json for the web app
"""

import os
import json
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

FILE_PATH = "data/housing_data_processed.csv"
LOCATIONS_FILE = "locations.json"
CHOICES_FILE = "choices.json"

print("\n" + "="*80)
print("📍 EXTRACTING LOCATIONS & CHOICES")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

try:
    df = pd.read_csv(FILE_PATH)
    print(f"✓ Loaded data: {len(df)} records")
    print(f"✓ Columns: {', '.join(df.columns.tolist())}")
except FileNotFoundError:
    print(f"❌ Error: File not found at {FILE_PATH}")
    exit(1)

# ============================================================================
# CHECK REQUIRED COLUMNS
# ============================================================================

required_cols = ['City', 'District', 'Legal status', 'Furniture state']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"❌ Missing columns: {missing_cols}")
    print("Available columns:", df.columns.tolist())
    exit(1)

# ============================================================================
# EXTRACT LOCATIONS (PROVINCE -> DISTRICT -> CITY)
# ============================================================================

def normalize_province(city_value):
    """
    Determine province from City column value
    Common patterns: "Hồ Chí Minh", "Hà Nội", etc.
    """
    if pd.isnull(city_value):
        return "Khác"
    
    city_str = str(city_value).strip()
    
    # Map variations to standard province names
    if any(x in city_str for x in ["Hồ Chí Minh", "TPHCM", "HCM", "TP.HCM", "Sài Gòn"]):
        return "Hồ Chí Minh"
    elif any(x in city_str for x in ["Hà Nội", "HN", "Ha Noi"]):
        return "Hà Nội"
    else:
        return "Khác"

# Add Province column
df['Province'] = df['City'].apply(normalize_province)

print(f"\n📊 Province distribution:")
print(df['Province'].value_counts())

# Filter to only major cities (Hà Nội & Hồ Chí Minh)
df_filtered = df[df['Province'].isin(['Hà Nội', 'Hồ Chí Minh'])].copy()

print(f"\n✓ Filtered to Hà Nội & Hồ Chí Minh: {len(df_filtered)} records")
print(f"  - Hà Nội: {len(df_filtered[df_filtered['Province'] == 'Hà Nội'])}")
print(f"  - Hồ Chí Minh: {len(df_filtered[df_filtered['Province'] == 'Hồ Chí Minh'])}")

# ============================================================================
# BUILD LOCATIONS DICTIONARY
# ============================================================================

def clean_district_name(district):
    """Clean and validate district names"""
    if pd.isnull(district):
        return None
    
    district_str = str(district).strip()
    
    # Filter out invalid entries
    invalid_keywords = [
        'bán nhà', 'giá', 'tỷ', 'phòng công chứng', 
        'đường số', 'mặt tiền', 'hẻm', 'cần bán'
    ]
    
    if any(keyword.lower() in district_str.lower() for keyword in invalid_keywords):
        return None
    
    # Filter out very short or very long names
    if len(district_str) < 2 or len(district_str) > 50:
        return None
    
    return district_str

locations = {}

for province in ['Hà Nội', 'Hồ Chí Minh']:
    # Get all districts for this province
    province_data = df_filtered[df_filtered['Province'] == province]
    
    # Extract unique districts
    districts = province_data['District'].dropna().unique().tolist()
    
    # Clean district names
    clean_districts = [clean_district_name(d) for d in districts]
    clean_districts = [d for d in clean_districts if d is not None]
    
    # Remove duplicates and sort
    clean_districts = sorted(list(set(clean_districts)))
    
    # Add "N/A" option at the beginning
    if clean_districts:
        clean_districts.insert(0, "N/A")
    else:
        clean_districts = ["N/A"]
    
    locations[province] = clean_districts
    print(f"\n✓ {province}: {len(clean_districts)} districts")
    print(f"  Examples: {', '.join(clean_districts[:5])}")

# ============================================================================
# SAVE LOCATIONS.JSON
# ============================================================================

with open(LOCATIONS_FILE, 'w', encoding='utf-8') as f:
    json.dump(locations, f, ensure_ascii=False, indent=4)

print(f"\n✓ Saved {LOCATIONS_FILE}")

# ============================================================================
# EXTRACT CHOICES (LEGAL STATUS & FURNITURE STATE)
# ============================================================================

def clean_choices(values):
    """Clean and prepare choice values"""
    # Remove null and empty values
    values = [v for v in values if pd.notna(v) and str(v).strip() not in ['', 'nan', 'None']]
    
    # Remove duplicates and sort
    values = sorted(list(set(values)))
    
    # Add "N/A" option at the beginning
    if values:
        if "N/A" not in values:
            values.insert(0, "N/A")
    else:
        values = ["N/A"]
    
    return values

choices = {}

# Legal Status
legal_statuses = df_filtered['Legal status'].dropna().unique().tolist()
choices['legal_statuses'] = clean_choices(legal_statuses)
print(f"\n✓ Legal statuses: {len(choices['legal_statuses'])}")
print(f"  Values: {', '.join(choices['legal_statuses'])}")

# Furniture State
furniture_states = df_filtered['Furniture state'].dropna().unique().tolist()
choices['furniture_states'] = clean_choices(furniture_states)
print(f"\n✓ Furniture states: {len(choices['furniture_states'])}")
print(f"  Values: {', '.join(choices['furniture_states'])}")

# ============================================================================
# SAVE CHOICES.JSON
# ============================================================================

with open(CHOICES_FILE, 'w', encoding='utf-8') as f:
    json.dump(choices, f, ensure_ascii=False, indent=4)

print(f"\n✓ Saved {CHOICES_FILE}")

# ============================================================================
# SAVE FILTERED DATA (OPTIONAL)
# ============================================================================

# Save the filtered data for use in training
filtered_output = "data/housing_data_filtered.csv"
df_filtered.to_csv(filtered_output, index=False)
print(f"\n✓ Saved filtered data: {filtered_output}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ EXTRACTION COMPLETE")
print("="*80)
print(f"📊 Summary:")
print(f"  - Total records: {len(df)}")
print(f"  - Filtered records: {len(df_filtered)}")
print(f"  - Provinces: {len(locations)}")
print(f"  - Total districts: {sum(len(v) for v in locations.values())}")
print(f"  - Legal statuses: {len(choices['legal_statuses'])}")
print(f"  - Furniture states: {len(choices['furniture_states'])}")
print("="*80 + "\n")

# ============================================================================
# PREVIEW FILES
# ============================================================================

print("📄 Preview of locations.json:")
for province, districts in locations.items():
    print(f"  {province}: {districts[:3]}... ({len(districts)} total)")

print("\n📄 Preview of choices.json:")
print(f"  Legal statuses: {choices['legal_statuses']}")
print(f"  Furniture states: {choices['furniture_states']}")
print()