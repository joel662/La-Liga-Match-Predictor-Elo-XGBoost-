import pandas as pd
import glob
import os

# === CONFIG ===
data_path = os.path.join(os.path.dirname(__file__), "Match Data")
output_file = "laliga_merged_clean.csv"

# === 1️⃣ FIND FILES ===
all_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
print(f"📂 Found {len(all_files)} files")

df_list = []
for i, file in enumerate(all_files, 1):
    print(f"[{i}/{len(all_files)}] Reading {os.path.basename(file)} ...")
    try:
        # Read CSV safely with on_bad_lines='skip'
        df = pd.read_csv(file, skip_blank_lines=True, skipinitialspace=True, on_bad_lines="skip")

        # Drop completely empty columns (e.g. "Unnamed: 44" etc.)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # Trim spaces from headers
        df.columns = df.columns.str.strip()

        # Keep only relevant columns (if they exist)
        keep_cols = [
            "Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
            "B365H", "B365D", "B365A", "BbMxH", "BbMxD", "BbMxA",
            "BbAvH", "BbAvD", "BbAvA", "PSH", "PSD", "PSA",
            "HTHG", "HTAG", "HTR"
        ]
        df = df[[c for c in keep_cols if c in df.columns]]

        df_list.append(df)
        print(f"✅ Loaded {len(df)} rows")

    except Exception as e:
        print(f"⚠️ Error reading {file}: {e}")

# === 2️⃣ CONCATENATE ===
print("\n🔄 Merging files...")
merged = pd.concat(df_list, ignore_index=True)
print(f"✅ Merged shape: {merged.shape}")

# === 3️⃣ CLEAN DATES ===
print("📅 Converting date column...")
merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce", dayfirst=True)
merged = merged.dropna(subset=["Date"])
merged = merged.sort_values("Date").reset_index(drop=True)

# === 4️⃣ FILTER TO LA LIGA (SP1) ===
if "Div" in merged.columns:
    merged = merged[merged["Div"].astype(str).str.upper() == "SP1"]

# === 5️⃣ REMOVE DUPLICATES & RESET INDEX ===
merged = merged.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
merged = merged.reset_index(drop=True)

# === 6️⃣ SAVE FINAL CLEAN FILE ===
merged.to_csv(output_file, index=False)
print(f"\n✅ Cleaned and merged La Liga dataset saved as '{output_file}'")
print(f"   Total rows: {len(merged)}")
print(f"   Date range: {merged['Date'].min():%Y-%m-%d} → {merged['Date'].max():%Y-%m-%d}")
print(f"   Columns: {list(merged.columns)}")
