import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration for peak conditions based on each lot's specifics
lot_conditions = {
    1: {"peak_days": list(range(1, 8)), "peak_hours": (8, 21), "Th_peak": 1.25, "Th_off": 1, "current_occupancy": 0, "average_occupancy": 3},
    2: {"peak_days": list(range(1, 8)), "peak_hours": (18, 22), "Th_peak": 1.5, "Th_off": 1, "current_occupancy": 0, "average_occupancy": 3},
    3: {"peak_days": list(range(1, 6)), "peak_hours": (8, 17), "Th_peak": 1.5, "Th_off": 1, "current_occupancy": 0, "average_occupancy": 2},
    4: {"peak_days": [3], "peak_hours": (17, 22), "Th_peak": 1.5, "Th_off": 1, "current_occupancy": 0, "average_occupancy": 6},
    5: {"peak_days": [6, 7], "peak_hours": (10, 21), "Th_peak": 1.5, "Th_off": 1, "current_occupancy": 0, "average_occupancy": 4}
}

Pbase_values = {
    1: 200,
    2: 180,
    3: 160,
    4: 150,
    5: 180
}

def _ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _duration_hours(row):
    base = (row["End Hour"] - row["Start Hour"])
    over = 0 if pd.isna(row.get("Overstay Hours", np.nan)) else row["Overstay Hours"]
    # guard against negative or zero (can happen with malformed rows)
    dur = max(float(base + over), 0.0)
    return dur

def _plot_lines_per_lot(series_dict, lots, title, ylabel, outfile):
    """
    series_dict: dict {label: pd.Series indexed by Lot ID}
    lots: ordered list of lot IDs to plot on x-axis
    """
    plt.figure(figsize=(10, 6))
    for label, s in series_dict.items():
        s = s.reindex(lots).fillna(0)
        plt.plot(lots, s.values, marker="o", label=label)
    plt.xlabel("Lot")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(lots)
    plt.legend(title="Model", frameon=False)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.show()
    print(f"Saved: {outfile}")

def generate_all_lot_lineplots(csv_path="gen_bookings_with_price.csv"):
    """
    Creates six line plots (x = Lot ID):
      1) Total revenue per model (Reserved=1)
      2) Revenue/booking per model (Reserved=1)
      3) Number of bookings per lot (two lines: Reserved=1, Reserved=0)
      4) Revenue/hour per model (Reserved=1)
      5) Median final price per model (Reserved=1)
      6) IQR(final price) per model (Reserved=1)
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}")

    df = pd.read_csv(path)

    # Expected columns
    model_cols = ["Pfinal_Model1", "Pfinal_Model2", "Pfinal_Model3", "Pfinal_Model4"]
    required = ["Lot ID", "Reserved", "Start Hour", "End Hour", "Overstay Hours", *model_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    # Clean types
    df = _ensure_numeric(df, ["Lot ID", "Reserved", "Start Hour", "End Hour", "Overstay Hours", *model_cols])

    # Duration (hrs) per booking (used for revenue/hour)
    df["duration_hours"] = df.apply(_duration_hours, axis=1)

    # Reserved filter
    df_paid = df[df["Reserved"] == 1].copy()

    # Lot ordering
    lots = sorted(df["Lot ID"].dropna().astype(int).unique().tolist())

    # Friendly model labels
    name_map = {
        "Pfinal_Model1": "M1: Time-based Dynamic",
        "Pfinal_Model2": "M2: Flat-time Dynamic",
        "Pfinal_Model3": "M3: Static",
        "Pfinal_Model4": "M4: Time-based Static",
    }

    # -------------------------
    # 1) Total revenue per model (Reserved=1)
    # -------------------------
    total_rev = {}
    for col in model_cols:
        s = df_paid.groupby("Lot ID")[col].sum()
        total_rev[name_map[col]] = s
    _plot_lines_per_lot(
        total_rev, lots,
        title="Total Revenue per Lot by Model ",
        ylabel="Total revenue",
        outfile="plot_total_revenue_by_model_per_lot.png"
    )

    # -------------------------
    # 2) Revenue / booking per model (Reserved=1)
    #    (= mean final price for those bookings)
    # -------------------------
    rev_per_booking = {}
    for col in model_cols:
        g = df_paid.groupby("Lot ID")[col]
        # sum / count handles empty groups (avoid divide-by-zero)
        s = g.sum() / g.count().replace(0, np.nan)
        rev_per_booking[name_map[col]] = s
    _plot_lines_per_lot(
        rev_per_booking, lots,
        title="Revenue per Booking by Lot and Model ",
        ylabel="Revenue / booking",
        outfile="plot_revenue_per_booking_by_model_per_lot.png"
    )

    # -------------------------
    # 3) Number of bookings per lot (two lines: Reserved=1, Reserved=0)
    #    (counts don’t depend on model)
    # -------------------------
    counts_reserved = df[df["Reserved"] == 1].groupby("Lot ID")["Reserved"].count()
    counts_unreserved = df[df["Reserved"] == 0].groupby("Lot ID")["Reserved"].count()
    _plot_lines_per_lot(
        {
            "Bookings ": counts_reserved
        },
        lots,
        title="Number of Bookings per Lot by Reservation Status",
        ylabel="# bookings",
        outfile="plot_booking_counts_by_lot_reserved_vs_not.png"
    )

    # -------------------------
    # 4) Revenue / hour per model (Reserved=1)
    #    = sum(final price) / sum(duration hours) by lot
    # -------------------------
    rev_per_hour = {}
    # Protect against zero total durations in a lot: replace 0 with NaN
    dur_by_lot = df_paid.groupby("Lot ID")["duration_hours"].sum().replace(0, np.nan)
    for col in model_cols:
        rev_by_lot = df_paid.groupby("Lot ID")[col].sum()
        s = rev_by_lot / dur_by_lot
        rev_per_hour[name_map[col]] = s
    _plot_lines_per_lot(
        rev_per_hour, lots,
        title="Revenue per Hour by Lot and Model",
        ylabel="Revenue / hour",
        outfile="plot_revenue_per_hour_by_model_per_lot.png"
    )

    # -------------------------
    # 5) Median final price per model (Reserved=1)
    # -------------------------
    med_price = {}
    for col in model_cols:
        s = df_paid.groupby("Lot ID")[col].median()
        med_price[name_map[col]] = s
    _plot_lines_per_lot(
        med_price, lots,
        title="Median Final Price by Lot and Model ",
        ylabel="Median final price",
        outfile="plot_median_final_price_by_model_per_lot.png"
    )

    # -------------------------
    # 6) IQR(final price) per model (Reserved=1)
    #    IQR = Q3 - Q1
    # -------------------------
    iqr_price = {}
    for col in model_cols:
        q1 = df_paid.groupby("Lot ID")[col].quantile(0.25)
        q3 = df_paid.groupby("Lot ID")[col].quantile(0.75)
        s = q3 - q1
        iqr_price[name_map[col]] = s
    _plot_lines_per_lot(
        iqr_price, lots,
        title="Interquartile Range of Final Price by Lot and Model",
        ylabel="IQR(final price)",
        outfile="plot_iqr_final_price_by_model_per_lot.png"
    )

    # Optional: dump the underlying tables for your appendix
    pd.DataFrame(total_rev).reindex(lots).to_csv("tbl_total_revenue_by_model_per_lot.csv")
    pd.DataFrame(rev_per_booking).reindex(lots).to_csv("tbl_rev_per_booking_by_model_per_lot.csv")
    pd.DataFrame({"Reserved=1": counts_reserved.reindex(lots).fillna(0).astype(int),
                  "Reserved=0": counts_unreserved.reindex(lots).fillna(0).astype(int)}).to_csv("tbl_booking_counts_by_lot.csv")
    pd.DataFrame(rev_per_hour).reindex(lots).to_csv("tbl_rev_per_hour_by_model_per_lot.csv")
    pd.DataFrame(med_price).reindex(lots).to_csv("tbl_median_final_price_by_model_per_lot.csv")
    pd.DataFrame(iqr_price).reindex(lots).to_csv("tbl_iqr_final_price_by_model_per_lot.csv")
    print("Saved CSV tables for all six plots.")

# ---- helper: is a given (lot, day, hour) peak? ----
def _is_peak(lot_id: int, day: int, hour: int, lot_conditions: dict) -> bool:
    lc = lot_conditions[int(lot_id)]
    in_day  = day in lc["peak_days"]
    start, end = lc["peak_hours"]
    in_hour = (start <= hour <= end)
    return bool(in_day and in_hour)

def plot_peak_offpeak_rev_per_hour(
    lot_conditions: dict,
    csv_path: str = "gen_bookings_with_price.csv",
    reserved_only: bool = True,
    outfile_png: str = "plot_rev_per_hour_peak_offpeak_by_lot.png",
    summary_csv: str = "tbl_rev_per_hour_peak_offpeak_by_lot.csv"
):
    """
    For each lot and model (M1..M4), compute average revenue/hour
    separately for *peak* and *off-peak* hours, then plot them as lines.

    Notes:
    - Revenue/hour is computed by evenly distributing each booking's final
      price over its occupied hours (including overstay hours), then
      bucketing each occupied hour as peak/off-peak per lot_conditions.
    - If reserved_only=True, uses only bookings with Reserved==1.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}")

    df = pd.read_csv(path)

    model_cols = ["Pfinal_Model1", "Pfinal_Model2", "Pfinal_Model3", "Pfinal_Model4"]
    required = ["Lot ID", "Day", "Start Hour", "End Hour", "Overstay Hours", "Reserved", *model_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    # Cast types
    num_cols = ["Lot ID","Day","Start Hour","End Hour","Overstay Hours","Reserved",*model_cols]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    if reserved_only:
        df = df[df["Reserved"] == 1].copy()

    # Prepare accumulators: sum of per-hour revenue and count of hours per (lot, bucket, model)
    lots = sorted(df["Lot ID"].dropna().astype(int).unique().tolist())
    buckets = ["peak","offpeak"]
    name_map = {
        "Pfinal_Model1":"M1: Time-based Dynamic",
        "Pfinal_Model2":"M2: Flat-time Dynamic",
        "Pfinal_Model3":"M3: Static",
        "Pfinal_Model4":"M4: Time-based Static",
    }

    # dict[(lot, bucket, model_label)] -> {"rev": float, "hours": int}
    acc = {}

    for _, r in df.iterrows():
        lot = int(r["Lot ID"]); day = int(r["Day"])
        start = int(r["Start Hour"]); end = int(r["End Hour"])
        overstay = 0 if pd.isna(r["Overstay Hours"]) else int(r["Overstay Hours"])
        total_hours = max((end - start) + overstay, 0)
        if total_hours <= 0:
            continue

        # Evenly allocate final price across each occupied hour
        per_hour = {m: (float(r[m]) / total_hours) for m in model_cols}

        for h in range(start, end + overstay):
            bucket = "peak" if _is_peak(lot, day, h, lot_conditions) else "offpeak"
            for m in model_cols:
                key = (lot, bucket, name_map[m])
                if key not in acc:
                    acc[key] = {"rev": 0.0, "hours": 0}
                acc[key]["rev"]   += per_hour[m]
                acc[key]["hours"] += 1

    # Build a tidy frame of averages
    rows = []
    for lot in lots:
        for model_label in name_map.values():
            for bucket in buckets:
                key = (lot, bucket, model_label)
                rev = acc.get(key, {"rev":np.nan,"hours":0})["rev"]
                hrs = acc.get(key, {"rev":0.0,"hours":0})["hours"]
                avg_rev_per_hr = (rev / hrs) if hrs else np.nan
                rows.append({"Lot": lot, "Model": model_label, "Bucket": bucket, "Revenue_per_hour": avg_rev_per_hr})
    avg_df = pd.DataFrame(rows)

    # Also compute peak uplift (peak/offpeak) per (lot, model)
    uplifts = []
    for lot in lots:
        for model_label in name_map.values():
            peak = avg_df[(avg_df.Lot==lot)&(avg_df.Model==model_label)&(avg_df.Bucket=="peak")]["Revenue_per_hour"].values
            offp = avg_df[(avg_df.Lot==lot)&(avg_df.Model==model_label)&(avg_df.Bucket=="offpeak")]["Revenue_per_hour"].values
            peak = peak[0] if len(peak) else np.nan
            offp = offp[0] if len(offp) else np.nan
            uplifts.append({"Lot":lot,"Model":model_label,"Peak_offpeak_ratio": (peak/offp) if (offp and not np.isnan(offp)) else np.nan})
    uplift_df = pd.DataFrame(uplifts)

    # Save summary table (wide: separate columns for Peak/Off-peak and ratio)
    wide = avg_df.pivot_table(index=["Lot","Model"], columns="Bucket", values="Revenue_per_hour")
    out_table = wide.join(uplift_df.set_index(["Lot","Model"]))
    out_table.to_csv(summary_csv)
    print(f"Saved: {summary_csv}")

    # Plot: one figure, 8 lines (Peak/Off-peak for each model)
    plt.figure(figsize=(12,7))
    for model_label in name_map.values():
        s_peak = avg_df[(avg_df["Model"]==model_label)&(avg_df["Bucket"]=="peak")].set_index("Lot")["Revenue_per_hour"].reindex(lots)
        s_off  = avg_df[(avg_df["Model"]==model_label)&(avg_df["Bucket"]=="offpeak")].set_index("Lot")["Revenue_per_hour"].reindex(lots)
        plt.plot(lots, s_peak.values, marker="o", linestyle="-", label=f"{model_label} — Peak")
        plt.plot(lots, s_off.values,  marker="o", linestyle="--", label=f"{model_label} — Off-peak")

    plt.xlabel("Lot")
    plt.ylabel("Revenue per hour")
    plt.title("Revenue per Hour by Lot and Peak vs Off-peak (Reserved only)" if reserved_only else
              "Revenue per Hour by Lot and Peak vs Off-peak (All bookings)")
    plt.xticks(lots)
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(outfile_png, dpi=150)
    plt.show()
    print(f"Saved: {outfile_png}")

def _is_weekend(day: int) -> bool:
    # Your Day encoding: 1=Mon ... 7=Sun
    return int(day) in (6, 7)

def plot_weekday_weekend_rev_per_hour(
    csv_path: str = "gen_bookings_with_price.csv",
    reserved_only: bool = True,
    outfile_png: str = "plot_rev_per_hour_weekday_weekend_by_lot.png",
    summary_csv: str = "tbl_rev_per_hour_weekday_weekend_by_lot.csv"
):
    """
    For each lot and model (M1..M4), compute average revenue/hour
    separately for WEEKDAY (Mon–Fri) and WEEKEND (Sat–Sun), then plot them.

    Notes:
    - Revenue/hour is computed by evenly distributing each booking’s final price
      across its occupied hours (incl. overstay hours).
    - If reserved_only=True, uses only bookings with Reserved==1.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}")

    df = pd.read_csv(path)

    model_cols = ["Pfinal_Model1", "Pfinal_Model2", "Pfinal_Model3", "Pfinal_Model4"]
    required = ["Lot ID","Day","Start Hour","End Hour","Overstay Hours","Reserved", *model_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    # types
    num_cols = ["Lot ID","Day","Start Hour","End Hour","Overstay Hours","Reserved", *model_cols]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    if reserved_only:
        df = df[df["Reserved"] == 1].copy()

    lots = sorted(df["Lot ID"].dropna().astype(int).unique().tolist())
    buckets = ["weekday","weekend"]
    name_map = {
        "Pfinal_Model1":"M1: Time-based Dynamic",
        "Pfinal_Model2":"M2: Flat-time Dynamic",
        "Pfinal_Model3":"M3: Static",
        "Pfinal_Model4":"M4: Time-based Static",
    }

    # accumulator: (lot, bucket, model) -> {rev, hours}
    acc = {}

    for _, r in df.iterrows():
        lot = int(r["Lot ID"]); day = int(r["Day"])
        start = int(r["Start Hour"]); end = int(r["End Hour"])
        overstay = 0 if pd.isna(r["Overstay Hours"]) else int(r["Overstay Hours"])
        total_hours = max((end - start) + overstay, 0)
        if total_hours <= 0:
            continue

        # evenly allocate to hours
        per_hour = {m: (float(r[m]) / total_hours) for m in model_cols}

        bucket = "weekend" if _is_weekend(day) else "weekday"
        # add each occupied hour (keeps “per-hour” denominator correct)
        for _h in range(start, end + overstay):
            for m in model_cols:
                key = (lot, bucket, name_map[m])
                if key not in acc:
                    acc[key] = {"rev": 0.0, "hours": 0}
                acc[key]["rev"]   += per_hour[m]
                acc[key]["hours"] += 1

    # tidy averages
    rows = []
    for lot in lots:
        for model_label in name_map.values():
            for bucket in buckets:
                key = (lot, bucket, model_label)
                rev = acc.get(key, {"rev":np.nan,"hours":0})["rev"]
                hrs = acc.get(key, {"rev":0.0,"hours":0})["hours"]
                avg = (rev / hrs) if hrs else np.nan
                rows.append({"Lot": lot, "Model": model_label, "Bucket": bucket, "Revenue_per_hour": avg})
    avg_df = pd.DataFrame(rows)

    # Weekend uplift = weekend / weekday
    uplifts = []
    for lot in lots:
        for model_label in name_map.values():
            wknd = avg_df[(avg_df.Lot==lot)&(avg_df.Model==model_label)&(avg_df.Bucket=="weekend")]["Revenue_per_hour"].values
            wkdy = avg_df[(avg_df.Lot==lot)&(avg_df.Model==model_label)&(avg_df.Bucket=="weekday")]["Revenue_per_hour"].values
            wknd = wknd[0] if len(wknd) else np.nan
            wkdy = wkdy[0] if len(wkdy) else np.nan
            uplifts.append({
                "Lot": lot,
                "Model": model_label,
                "Weekend_Weekday_ratio": (wknd / wkdy) if (wkdy and not np.isnan(wkdy)) else np.nan
            })
    uplift_df = pd.DataFrame(uplifts)

    # Save summary (wide format + ratio)
    wide = avg_df.pivot_table(index=["Lot","Model"], columns="Bucket", values="Revenue_per_hour")
    out_table = wide.join(uplift_df.set_index(["Lot","Model"]))
    out_table.to_csv(summary_csv)
    print(f"Saved: {summary_csv}")

    # Plot lines: Weekday (solid) and Weekend (dashed) per model
    plt.figure(figsize=(12,7))
    for model_label in name_map.values():
        s_wkdy = avg_df[(avg_df["Model"]==model_label)&(avg_df["Bucket"]=="weekday")].set_index("Lot")["Revenue_per_hour"].reindex(lots)
        s_wknd = avg_df[(avg_df["Model"]==model_label)&(avg_df["Bucket"]=="weekend")].set_index("Lot")["Revenue_per_hour"].reindex(lots)
        plt.plot(lots, s_wkdy.values, marker="o", linestyle="-",  label=f"{model_label} — Weekday")
        plt.plot(lots, s_wknd.values, marker="o", linestyle="--", label=f"{model_label} — Weekend")

    plt.xlabel("Lot")
    plt.ylabel("Revenue per hour")
    plt.title("Revenue per Hour by Lot: Weekday vs Weekend (Reserved only)" if reserved_only else
              "Revenue per Hour by Lot: Weekday vs Weekend (All bookings)")
    plt.xticks(lots)
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(outfile_png, dpi=150)
    plt.show()
    print(f"Saved: {outfile_png}")

# ---------- Helper: build capacity map (capacity = Average Occupancy in your CSV) ----------
def _capacity_by_lot(bookings_df: pd.DataFrame) -> dict:
    caps = (
        bookings_df
        .groupby("Lot ID")["Average Occupancy"]
        .max()  # treat as capacity
        .fillna(0)
        .astype(int)
        .to_dict()
    )
    # fallback if empty
    if not caps:
        raise ValueError("Could not infer capacities; check 'Average Occupancy' column.")
    return caps

# ---------- 1) Availability per lot (A_l) ----------
def plot_availability_per_lot(
    csv_path: str = "gen_bookings_with_price.csv",
    reserved_only: bool = True,
    outfile_png: str = "plot_availability_per_lot.png",
    summary_csv: str = "tbl_availability_per_lot.csv"
):
    """
    Compute availability A_l = share of simulated hours (168) where occupancy < capacity,
    using bookings (Reserved==1 by default) + overstays to mark occupied hours.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}")

    df = pd.read_csv(path)

    needed = ["Lot ID","Day","Start Hour","End Hour","Overstay Hours","Reserved","Average Occupancy"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    # types
    cast_cols = ["Lot ID","Day","Start Hour","End Hour","Overstay Hours","Reserved","Average Occupancy"]
    df[cast_cols] = df[cast_cols].apply(pd.to_numeric, errors="coerce")

    if reserved_only:
        df = df[df["Reserved"] == 1].copy()

    lots = sorted(df["Lot ID"].dropna().astype(int).unique().tolist())
    caps = _capacity_by_lot(df)

    # 168 hours = 7 days * 24 hours
    T = 7 * 24
    occ = {lot: np.zeros(T, dtype=int) for lot in lots}

    # Mark occupied hours
    for _, r in df.iterrows():
        lot = int(r["Lot ID"])
        day = int(r["Day"])          # 1..7
        start = int(r["Start Hour"]) # 0..23
        end   = int(r["End Hour"])   # 1..24
        over  = 0 if pd.isna(r["Overstay Hours"]) else int(r["Overstay Hours"])
        end_effective = min(24, end + over)

        # index range within that day
        base = (day - 1) * 24
        for h in range(start, end_effective):
            idx = base + h
            if 0 <= idx < T:
                occ[lot][idx] += 1

    # Compute A_l = share of hours with occupancy < capacity
    rows = []
    for lot in lots:
        capacity = int(caps.get(lot, 0))
        if capacity <= 0:
            a_l = np.nan
        else:
            free_mask = occ[lot] < capacity
            a_l = free_mask.mean()
        rows.append({"Lot": lot, "Availability": a_l})

    out = pd.DataFrame(rows).sort_values("Lot")
    out["System_mean_A"] = out["Availability"].mean()
    out.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # Plot
    plt.figure(figsize=(8,5))
    plt.bar(out["Lot"].astype(str), out["Availability"])
    plt.axhline(out["System_mean_A"].iloc[0], linestyle="--")
    plt.ylim(0, 1.05)
    plt.xlabel("Lot")
    plt.ylabel("Share of hours with ≥1 free space")
    plt.title("Availability per Lot (A_l) with System-wide A")
    plt.tight_layout()
    plt.savefig(outfile_png, dpi=150)
    plt.show()
    print(f"Saved: {outfile_png}")

# ---------- 2) Discounts per lot (boxplot; model-agnostic) ----------
def plot_discounts_per_lot_box(
    csv_path: str = "gen_bookings_with_price.csv",
    dmax: float = 20.0,
    reserved_only: bool = True,
    outfile_png: str = "boxplot_discounts_per_lot.png",
    summary_csv: str = "tbl_discounts_per_lot.csv"
):
    """
    Boxplot of (capped) loyalty discount (%) per lot.
    Uses Dloyalty_final, capped at Dmax. Independent of pricing model.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}")

    df = pd.read_csv(path)
    needed = ["Lot ID","Dloyalty_final","Reserved"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df[["Lot ID","Dloyalty_final","Reserved"]] = df[["Lot ID","Dloyalty_final","Reserved"]].apply(pd.to_numeric, errors="coerce")
    if reserved_only:
        df = df[df["Reserved"] == 1].copy()

    df["Discount_effective_%"] = np.clip(df["Dloyalty_final"], 0, dmax)

    lots = sorted(df["Lot ID"].dropna().astype(int).unique().tolist())
    data = [df[df["Lot ID"]==lot]["Discount_effective_%"].dropna().values for lot in lots]

    # Summary table (median, IQR)
    rows = []
    for lot, arr in zip(lots, data):
        if len(arr) == 0:
            rows.append({"Lot": lot, "Median_%": np.nan, "IQR_%": np.nan})
        else:
            q1, med, q3 = np.percentile(arr, [25, 50, 75])
            rows.append({"Lot": lot, "Median_%": med, "IQR_%": (q3 - q1)})

    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # Boxplot
    plt.figure(figsize=(10,5))
    plt.boxplot(data, labels=[str(l) for l in lots], showfliers=False)
    plt.xlabel("Lot")
    plt.ylabel("Effective discount (%)")
    plt.title("Loyalty Discounts by Lot (median and IQR)")
    plt.tight_layout()
    plt.savefig(outfile_png, dpi=150)
    plt.show()
    print(f"Saved: {outfile_png}")

# ---------- 3) Overstay penalties per lot (boxplot & incidence bar) ----------
def plot_overstay_penalties_per_lot(
    csv_path: str = "gen_bookings_with_price.csv",
    reserved_only: bool = True,
    outfile_box: str = "boxplot_overstay_penalties_per_lot.png",
    outfile_inc: str = "bar_overstay_incidence_per_lot.png",
    summary_csv: str = "tbl_overstay_penalties_per_lot.csv"
):
    """
    (a) Boxplot of Poverstay_total by lot (bookings with O==1).
    (b) Bar chart of overstay incidence (% of bookings with O==1) by lot.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}")

    df = pd.read_csv(path)
    needed = ["Lot ID","O","Poverstay_total","Reserved"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df[["Lot ID","O","Poverstay_total","Reserved"]] = df[["Lot ID","O","Poverstay_total","Reserved"]].apply(pd.to_numeric, errors="coerce")
    if reserved_only:
        df = df[df["Reserved"] == 1].copy()

    lots = sorted(df["Lot ID"].dropna().astype(int).unique().tolist())

    # (a) penalties (only O==1)
    pen_data = [df[(df["Lot ID"]==lot) & (df["O"]==1)]["Poverstay_total"].dropna().values for lot in lots]

    # Summary (median, IQR, incidence)
    rows = []
    for lot, arr in zip(lots, pen_data):
        lot_df = df[df["Lot ID"]==lot]
        n_all = len(lot_df)
        n_over = int((lot_df["O"]==1).sum())
        incidence = (n_over / n_all) if n_all else np.nan

        if len(arr) == 0:
            med = np.nan; iqr = np.nan
        else:
            q1, med, q3 = np.percentile(arr, [25, 50, 75])
            iqr = q3 - q1

        rows.append({
            "Lot": lot,
            "Penalty_median": med,
            "Penalty_IQR": iqr,
            "Overstay_incidence": incidence
        })

    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # Plot (a) boxplot of penalties
    plt.figure(figsize=(10,5))
    plt.boxplot(pen_data, labels=[str(l) for l in lots], showfliers=False)
    plt.xlabel("Lot")
    plt.ylabel("Overstay penalty")
    plt.title("Overstay Penalties by Lot (O=1)")
    plt.tight_layout()
    plt.savefig(outfile_box, dpi=150)
    plt.show()
    print(f"Saved: {outfile_box}")

    # Plot (b) incidence bar
    inc_df = pd.DataFrame(rows)
    plt.figure(figsize=(8,5))
    plt.bar(inc_df["Lot"].astype(str), inc_df["Overstay_incidence"])
    plt.ylim(0, 1.05)
    plt.xlabel("Lot")
    plt.ylabel("Share of bookings with overstay (O=1)")
    plt.title("Overstay Incidence by Lot")
    plt.tight_layout()
    plt.savefig(outfile_inc, dpi=150)
    plt.show()
    print(f"Saved: {outfile_inc}")

def _infer_capacity_by_lot(df: pd.DataFrame) -> dict:
    """Capacity = max(Average Occupancy) per lot (your convention)."""
    return (
        df.groupby("Lot ID")["Average Occupancy"]
          .max()
          .fillna(0)
          .astype(int)
          .to_dict()
    )

def _infer_pbase_by_lot(df: pd.DataFrame) -> dict:
    """Infer Pbase per lot from CSV (mode of Pbase for that lot)."""
    pmap = {}
    for lot, g in df.groupby("Lot ID"):
        vals = g["Pbase"].dropna().values
        if len(vals):
            # mode-like: most frequent rounded value
            rounded = pd.Series(np.round(vals, 6))
            pmap[int(lot)] = float(rounded.mode().iloc[0])
    return pmap

def _weekend_weight(day: int) -> float:
    """W factor (1.3 on weekend, else 1.0). Day: 1=Mon ... 7=Sun."""
    return 1.3 if day in (6, 7) else 1.0

def _Th_for(lot_id: int, day: int, hour: int, lot_conditions: dict) -> float:
    """Time-of-day factor Th from lot_conditions."""
    lc = lot_conditions.get(lot_id, {})
    peak_days  = lc.get("peak_days", [])
    peak_hours = lc.get("peak_hours", (0, -1))
    Th_peak    = lc.get("Th_peak", 1.0)
    Th_off     = lc.get("Th_off",  1.0)
    if (day in peak_days) and (peak_hours[0] <= hour <= peak_hours[1]):
        return Th_peak
    return Th_off

def _occupancy_24h_for_lot_day(df: pd.DataFrame, lot_id: int, day: int) -> np.ndarray:
    """
    Build hourly occupancy array (length 24) for one lot & one day,
    counting Reserved==1 bookings plus overstays.
    """
    T = 24
    occ = np.zeros(T, dtype=int)
    d = df[(df["Lot ID"] == lot_id) & (df["Day"] == day)].copy()
    d["Reserved"] = pd.to_numeric(d["Reserved"], errors="coerce").fillna(0).astype(int)
    d["Overstay Hours"] = pd.to_numeric(d["Overstay Hours"], errors="coerce").fillna(0).astype(int)
    d["Start Hour"] = pd.to_numeric(d["Start Hour"], errors="coerce").fillna(0).astype(int)
    d["End Hour"]   = pd.to_numeric(d["End Hour"],   errors="coerce").fillna(0).astype(int)

    for _, r in d.iterrows():
        if r["Reserved"] != 1:
            continue
        start = int(r["Start Hour"])
        end   = int(r["End Hour"])
        over  = int(r["Overstay Hours"])
        end_eff = min(24, end + over)
        for h in range(max(0, start), max(0, end_eff)):
            if 0 <= h < 24:
                occ[h] += 1
    return occ

def plot_hourly_price_profiles(
    bookings_csv: str = "gen_bookings_with_price.csv",
    lot_conditions: dict = None,
    Pbase_values: dict = None,
    day: int = None,
    lot_id: int = None,
    onpeak_multiplier: float = 1.05,
    dmin: float = 0.7,
    outfile: str = "fig_hourly_price_profiles.png"
):
    """
    Figure 5.1 — Hourly price profiles (M1–M4) for a representative day and lot.
    - Picks the busiest day for the chosen lot if `day` is None.
    - Requires lot_conditions for Th (peak/off-peak windows).
    - Uses Pbase_values if provided; otherwise infers from CSV.

    Models:
      M1: P_h = Pbase * Dh * Th * W
      M2: P_h = Pbase * Dh * W
      M3: P_h = Pbase
      M4: P_h = Ppeak if peak hour else Poff, with Ppeak = onpeak_multiplier * Poff
    Where Dh = max(dmin, occupancy / capacity), W = 1.3 on weekend else 1.0
    """
    if lot_conditions is None:
        raise ValueError("lot_conditions dict is required (for peak windows and Th).")

    path = Path(bookings_csv)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {bookings_csv}")

    df = pd.read_csv(path)
    needed = ["Lot ID","Day","Start Hour","End Hour","Overstay Hours","Pbase","Average Occupancy","Reserved"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # coerce types
    for c in ["Lot ID","Day","Start Hour","End Hour","Overstay Hours","Pbase","Average Occupancy","Reserved"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # choose lot if not given -> pick the most active lot overall
    if lot_id is None:
        lot_id = int(df["Lot ID"].mode().iloc[0])

    # choose day if not given -> busiest day for that lot
    if day is None:
        counts = df[(df["Lot ID"]==lot_id) & (df["Reserved"]==1)].groupby("Day").size()
        day = int(counts.idxmax()) if len(counts) else 1

    # infer capacity & Pbase per lot
    cap_map = _infer_capacity_by_lot(df)
    if Pbase_values is None or lot_id not in Pbase_values:
        pmap = _infer_pbase_by_lot(df)
        Pbase = pmap.get(lot_id, float(df[df["Lot ID"]==lot_id]["Pbase"].dropna().iloc[0]))
    else:
        Pbase = float(Pbase_values[lot_id])

    C = max(1, int(cap_map.get(lot_id, 1)))  # avoid division by zero
    occ = _occupancy_24h_for_lot_day(df, lot_id, day)

    # compute W for that day
    W = _weekend_weight(day)

    # hourly prices per model
    hours = np.arange(24)
    M1 = np.zeros(24)
    M2 = np.zeros(24)
    M3 = np.ones(24) * Pbase
    M4 = np.zeros(24)

    for h in hours:
        Dh = max(dmin, occ[h] / C)
        Th = _Th_for(lot_id, day, int(h), lot_conditions)

        M1[h] = Pbase * Dh * Th * W
        M2[h] = Pbase * Dh * W

        # time-based static
        is_peak = (day in lot_conditions[lot_id]["peak_days"]) and \
                  (lot_conditions[lot_id]["peak_hours"][0] <= h <= lot_conditions[lot_id]["peak_hours"][1])
        Poff = Pbase
        Ppeak = onpeak_multiplier * Poff
        M4[h] = Ppeak if is_peak else Poff

    # plot
    plt.figure(figsize=(10,5))
    plt.plot(hours, M1, label="M1: time-based dynamic")
    plt.plot(hours, M2, label="M2: flat-time dynamic")
    plt.plot(hours, M3, label="M3: static")
    plt.plot(hours, M4, label="M4: time-based static")
    plt.xlabel("Hour of day")
    plt.ylabel("Hourly price (₦)")
    plt.title(f"Hourly Price Profiles — Lot {lot_id}, Day {day}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.show()
    print(f"Saved: {outfile}")

def plot_occupancy_vs_capacity_for_day(
    bookings_csv: str = "gen_bookings_with_price.csv",
    day: int = None,
    lot_id: int = None,
    outfile: str = "fig_occupancy_vs_capacity.png"
):
    """
    Figure 5.2 — Occupancy profile vs. capacity for a lot on the same representative day.
    - If `day` is None, chooses busiest day for that lot.
    - Capacity = max(Average Occupancy) per lot (your convention).
    """
    path = Path(bookings_csv)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {bookings_csv}")

    df = pd.read_csv(path)
    needed = ["Lot ID","Day","Start Hour","End Hour","Overstay Hours","Average Occupancy","Reserved"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    for c in ["Lot ID","Day","Start Hour","End Hour","Overstay Hours","Average Occupancy","Reserved"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if lot_id is None:
        lot_id = int(df["Lot ID"].mode().iloc[0])

    # pick busiest day if not provided
    if day is None:
        counts = df[(df["Lot ID"]==lot_id) & (df["Reserved"]==1)].groupby("Day").size()
        day = int(counts.idxmax()) if len(counts) else 1

    cap_map = _infer_capacity_by_lot(df)
    C = max(1, int(cap_map.get(lot_id, 1)))
    occ = _occupancy_24h_for_lot_day(df, lot_id, day)

    hours = np.arange(24)
    plt.figure(figsize=(10,5))
    plt.plot(hours, occ, label="Occupancy Oₗ,h")
    plt.axhline(C, linestyle="--", label=f"Capacity Cₗ={C}")
    plt.xlabel("Hour of day")
    plt.ylabel("Number of occupied stalls")
    plt.title(f"Occupancy vs Capacity — Lot {lot_id}, Day {day}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.show()
    print(f"Saved: {outfile}")

if __name__ == "__main__":
    # generate_all_lot_lineplots()
    # plot_peak_offpeak_rev_per_hour(lot_conditions)
    # plot_weekday_weekend_rev_per_hour()
    # plot_availability_per_lot()
    # plot_discounts_per_lot_box()
    # plot_overstay_penalties_per_lot()
    plot_hourly_price_profiles(lot_conditions=lot_conditions, Pbase_values=Pbase_values, day=3, lot_id=2)
    plot_occupancy_vs_capacity_for_day( day=3, lot_id=2)