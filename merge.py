import pandas as pd
import glob
import os

DATA_DIR = "Match Data"
OUTPUT_FILE = "laliga_merged_clean.csv"


def load_and_clean_csv(path):
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except:
        return pd.DataFrame()

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    if "Date" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"])

    needed_cols = [
        "Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "B365H", "B365D", "B365A", "BbMxH", "BbMxD", "BbMxA",
        "BbAvH", "BbAvD", "BbAvA", "HTHG", "HTAG", "HTR",
        "PSH", "PSD", "PSA",
    ]
    df = df[[c for c in needed_cols if c in df.columns]]

    return df


def main():
    print("🔄 Merging CSV files...")

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    frames = []

    for f in csv_files:
        print(f" → {os.path.basename(f)}")
        df = load_and_clean_csv(f)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("❌ No valid CSVs found.")
        return

    merged = pd.concat(frames, ignore_index=True)

    # Apply SP1 filter
    merged = merged[merged["Div"].astype(str).str.upper() == "SP1"]

    # Drop duplicates
    merged = merged.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last")

    # Sort by date
    merged = merged.sort_values("Date").reset_index(drop=True)

    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Saved '{OUTPUT_FILE}'")
    print(f"Total matches: {len(merged)}")
    print(f"Date range: {merged['Date'].min().date()} → {merged['Date'].max().date()}")


if __name__ == "__main__":
    main()
