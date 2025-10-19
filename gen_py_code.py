# Install python

import pandas as pd # pip install pandas
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

# Simulation clock
simulation_time = {"day": 1, "hour": 0, "minute": 0}  # Starts on Day 1 (Monday) at 00:00
time_increment = 1  # Minutes to advance per loop iteration
bookings_count = 24*60/time_increment*7

# Configuration for peak conditions based on each lot's specifics
lot_conditions = {
    1: {"peak_days": list(range(1, 8)), "peak_hours": (8, 21), "Th_peak": 1.25, "Th_off": 1, "current_occupancy": 0, "average_occupancy": 3},
    2: {"peak_days": list(range(1, 8)), "peak_hours": (18, 22), "Th_peak": 1.5, "Th_off": 1, "current_occupancy": 0, "average_occupancy": 3},
    3: {"peak_days": list(range(1, 6)), "peak_hours": (8, 17), "Th_peak": 1.5, "Th_off": 1, "current_occupancy": 0, "average_occupancy": 2},
    4: {"peak_days": [3], "peak_hours": (17, 22), "Th_peak": 1.5, "Th_off": 1, "current_occupancy": 0, "average_occupancy": 6},
    5: {"peak_days": [6, 7], "peak_hours": (10, 21), "Th_peak": 1.5, "Th_off": 1, "current_occupancy": 0, "average_occupancy": 4}
}

# Constants for the model
alpha = 0.2
beta = 0.1
Dmax = 20  # Maximum loyalty discount as percentage
Onpeak = 1.05 # For time based static pricing
Pbase_values = {
    1: 200,
    2: 180,
    3: 160,
    4: 150,
    5: 180
}

# File paths
users_file = 'gen_users.csv'
bookings_file = 'gen_bookings_with_price.csv'
bookings_only_file = 'gen_bookings.csv'
lot_occupancy_file = 'gen_lot_occupancy.csv' 
simulation_time_file = "gen_simulation_time.json"
requests_file = 'gen_requests.csv'   # log of accepted/denied booking requests
metrics_overview_file = 'gen_metrics_overview.csv'
availability_by_lot_file = 'gen_availability_by_lot.csv'
plots_dir = 'plots'
simulated_occupancy = 0

# Initialize DataFrames
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def log_request(day, lot_id, start_hour, end_hour, status):
    """Append a single booking request outcome to CSV."""
    row = pd.DataFrame([{
        "Day": day,
        "Lot ID": lot_id,
        "Start Hour": start_hour,
        "End Hour": end_hour,
        "Status": status  # "accepted" or "denied" (only initial bookings)
    }])
    if os.path.exists(requests_file):
        row.to_csv(requests_file, mode='a', header=False, index=False)
    else:
        row.to_csv(requests_file, index=False)

def build_occupancy_timeline(bookings_df, lot_conditions):
    """
    Rebuild per-hour occupancy from bookings (RESERVED==1).
    Returns a DataFrame with columns: Lot ID, Day, Hour, Occupancy, Capacity.
    """
    records = []
    for lot_id, cfg in lot_conditions.items():
        cap = int(cfg["average_occupancy"])
        for day in range(1, 8):
            for hour in range(0, 24):
                # count regular reserved hours
                mask = (
                    (bookings_df["Reserved"] == 1) &
                    (bookings_df["Lot ID"] == lot_id) &
                    (bookings_df["Day"] == day) &
                    (bookings_df["Start Hour"] <= hour) &
                    (bookings_df["End Hour"] > hour)
                )
                occ = int(mask.sum())

                # count overstay hours (if any)
                mask_over = (
                    (bookings_df["Reserved"] == 1) &
                    (bookings_df["Lot ID"] == lot_id) &
                    (bookings_df["Day"] == day) &
                    (~bookings_df["Overstay Start Hour"].isna()) &
                    (~bookings_df["Overstay End Hour"].isna()) &
                    (bookings_df["Overstay Start Hour"] <= hour) &
                    (bookings_df["Overstay End Hour"] > hour)
                )
                occ += int(mask_over.sum())

                records.append({
                    "Lot ID": lot_id,
                    "Day": day,
                    "Hour": hour,
                    "Occupancy": occ,
                    "Capacity": cap
                })
    return pd.DataFrame.from_records(records)

def med_iqr(series):
    """Return (median, IQR) where IQR = Q3 - Q1."""
    q1 = series.quantile(0.25)
    med = series.quantile(0.5)
    q3 = series.quantile(0.75)
    return float(med), float(q3 - q1)

def summarize_metrics(bookings_df, lot_conditions):
    """
    Compute tables & plots:
    - Revenue (total, per-hour, per-booking) for models 1–4
    - Availability A_l and system-wide A
    - Denied-booking rate from requests_file
    - Median [IQR] distributions for prices, discounts, overstay penalties
    """
    ensure_dir(plots_dir)

    # Use only completed/used bookings for revenue (Reserved==1)
    acc = bookings_df[bookings_df["Reserved"] == 1].copy()
    if acc.empty:
        print("No accepted (Reserved==1) bookings to summarize.")
        return

    # Hours per booking = booked hours + overstay hours
    acc["Hours"] = (acc["End Hour"] - acc["Start Hour"]).astype(float) + acc["Overstay Hours"].fillna(0).astype(float)
    total_hours = acc["Hours"].sum()

    # Revenue per model
    models = [
        ("Model1", "Pfinal_Model1"),
        ("Model2", "Pfinal_Model2"),
        ("Model3", "Pfinal_Model3"),
        ("Model4", "Pfinal_Model4"),
    ]

    overview_rows = []
    price_series_by_model = {}

    for label, col in models:
        s = acc[col].astype(float)
        price_series_by_model[label] = s
        total_rev = float(s.sum())
        rev_per_hour = float(total_rev / total_hours) if total_hours > 0 else np.nan
        rev_per_booking = float(s.mean()) if len(s) > 0 else np.nan
        med, iqr = med_iqr(s)
        overview_rows.append({
            "Model": label,
            "TotalRevenue": round(total_rev, 2),
            "RevenuePerHour": round(rev_per_hour, 4),
            "RevenuePerBooking": round(rev_per_booking, 2),
            "MedianPrice": round(med, 2),
            "IQRPrice": round(iqr, 2)
        })

    overview_df = pd.DataFrame(overview_rows)
    overview_df.to_csv(metrics_overview_file, index=False)
    print(f"Saved revenue overview -> {metrics_overview_file}")

    # Availability from reconstructed occupancy
    occ_df = build_occupancy_timeline(acc, lot_conditions)
    occ_df["has_space"] = (occ_df["Occupancy"] < occ_df["Capacity"]).astype(int)
    A_by_lot = occ_df.groupby("Lot ID")["has_space"].mean().reset_index().rename(columns={"has_space": "A_l"})
    A_by_lot["A_l"] = A_by_lot["A_l"].round(4)
    A_overall = float(A_by_lot["A_l"].mean()) if not A_by_lot.empty else np.nan

    # Save availability table (include overall A in a last row)
    overall_row = {"Lot ID": "SYSTEM", "A_l": round(A_overall, 4)}
    pd.concat([A_by_lot, pd.DataFrame([overall_row])], ignore_index=True)\
      .to_csv(availability_by_lot_file, index=False)
    print(f"Saved availability by lot -> {availability_by_lot_file}")

    # Denied booking rate from request log (accepted/denied for initial bookings)
    denied_rate = np.nan
    if os.path.exists(requests_file):
        req = pd.read_csv(requests_file)
        if not req.empty:
            total_req = len(req)
            denied = int((req["Status"] == "denied").sum())
            denied_rate = denied / total_req if total_req > 0 else np.nan
            print(f"Denied-booking rate: {denied_rate:.4f}  ({denied}/{total_req})")
    else:
        print("No request log found; denied-booking rate not computed.")

    # --------- Plots ---------
    # Boxplot of final prices by model
    plt.figure()
    data = [price_series_by_model[m] for m, _ in models]
    plt.boxplot(data, tick_labels=[m for m, _ in models], showfliers=False)
    plt.title("Final Prices by Model (boxplot)")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "boxplot_prices_by_model.png"))
    plt.close()

    # Boxplot of effective discounts (percent)
    discounts = acc["Dloyalty_final"].clip(upper=Dmax).astype(float)
    plt.figure()
    plt.boxplot([discounts], tick_labels=["Effective Discount (%)"], showfliers=False)
    plt.title("Effective Discounts (Median [IQR])")
    plt.ylabel("Percent")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "boxplot_discounts.png"))
    plt.close()

    # Boxplot of overstay penalties (only non-zero)
    overstay_pen = acc["Poverstay_total"].fillna(0).astype(float)
    nonzero_pen = overstay_pen[overstay_pen > 0]
    plt.figure()
    if len(nonzero_pen) > 0:
        plt.boxplot([nonzero_pen], tick_labels=["Overstay Penalty"], showfliers=False)
    else:
        plt.boxplot([overstay_pen], tick_labels=["Overstay Penalty"], showfliers=False)
    plt.title("Overstay Penalties (boxplot)")
    plt.ylabel("Currency")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "boxplot_overstay_penalties.png"))
    plt.close()

    # Bar of A_l and overall A
    plt.figure()
    plt.bar(A_by_lot["Lot ID"].astype(str), A_by_lot["A_l"])
    plt.axhline(A_overall, linestyle='--')
    plt.title("Availability per Lot (A_l) with System-wide A")
    plt.xlabel("Lot")
    plt.ylabel("Share of hours with ≥1 free space")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "availability_per_lot.png"))
    plt.close()

    print(f"Saved plots to ./{plots_dir}")
    print("Summary complete.")

def load_data():
    if os.path.exists(users_file):
        users_df = pd.read_csv(users_file)
    else:
        users_df = pd.DataFrame(columns=["User ID", "User Name", "F", "Ttotal", "M", "Baccumulated", "total_reservations", "num_overstays"])
        # Hardcoded list of user data with only User ID and User Name provided, other fields set to 0
        users_data = [
            {"User ID": 1, "User Name": "Alice", "F": 0, "Ttotal": 0, "M": 0, "Baccumulated": 0, "total_reservations": 0, "num_overstays": 0},
            {"User ID": 2, "User Name": "Bob", "F": 0, "Ttotal": 0, "M": 0, "Baccumulated": 0, "total_reservations": 0, "num_overstays": 0},
            {"User ID": 3, "User Name": "Charlie", "F": 0, "Ttotal": 0, "M": 0, "Baccumulated": 0, "total_reservations": 0, "num_overstays": 0},
            {"User ID": 4, "User Name": "Diana", "F": 0, "Ttotal": 0, "M": 0, "Baccumulated": 0, "total_reservations": 0, "num_overstays": 0},
            {"User ID": 5, "User Name": "Edward", "F": 0, "Ttotal": 0, "M": 0, "Baccumulated": 0, "total_reservations": 0, "num_overstays": 0},
            {"User ID": 6, "User Name": "Fiona", "F": 0, "Ttotal": 0, "M": 0, "Baccumulated": 0, "total_reservations": 0, "num_overstays": 0},
            {"User ID": 7, "User Name": "George", "F": 0, "Ttotal": 0, "M": 0, "Baccumulated": 0, "total_reservations": 0, "num_overstays": 0},
            {"User ID": 8, "User Name": "Hannah", "F": 0, "Ttotal": 0, "M": 0, "Baccumulated": 0, "total_reservations": 0, "num_overstays": 0},
            {"User ID": 9, "User Name": "Ian", "F": 0, "Ttotal": 0, "M": 0, "Baccumulated": 0, "total_reservations": 0, "num_overstays": 0},
            {"User ID": 10, "User Name": "Jane", "F": 0, "Ttotal": 0, "M": 0, "Baccumulated": 0, "total_reservations": 0, "num_overstays": 0}
        ]
        # Append the list of dictionaries to the DataFrame
        users_df = pd.concat([users_df, pd.DataFrame(users_data)], ignore_index=True)
    
    if os.path.exists(bookings_file):
        bookings_df = pd.read_csv(bookings_file)
    else:
        bookings_df = pd.DataFrame(columns=[
            "Day", "User ID", "Booking Number", "Lot ID", "Start Hour", "End Hour", "Pbase", "Current Occupancy", "Average Occupancy", "Dh",
        "Th", "W", "Ph", "Ptotal_Model1", "Ptotal_Model2", "Ptotal_Model3", "Ptotal_Model4", "F", "M", "Reserved", "gamma", "Baccumulated",
        "Bcurrent", "O", "Dloyalty_final", "Overstay Start Hour", "Overstay End Hour", "Overstay Hours", "Poverstay_total", "Pfinal_Model1", "Pfinal_Model2",
        "Pfinal_Model3", "Pfinal_Model4", "decremented", "decremented_overstay"
        ])
        for col in ["decremented", "decremented_overstay"]:
            if col not in bookings_df.columns:
                bookings_df[col] = False
            bookings_df[col] = bookings_df[col].fillna(False).astype(bool)
    if os.path.exists(bookings_only_file):
        bookings_only_df = pd.read_csv(bookings_only_file)
    else:
        bookings_only_df = pd.DataFrame(columns=[
            "Day", "User ID", "Booking Number", "Lot ID", "Start Hour", "End Hour", "Current Occupancy", "Average Occupancy",
        "F", "M", "Reserved", "Baccumulated", "Bcurrent", "O", "Overstay Start Hour", "Overstay End Hour", "Overstay Hours"
        ])

    # Load or initialize lot occupancy data
    if os.path.exists(lot_occupancy_file):
        lot_occupancy_df = pd.read_csv(lot_occupancy_file)
    else:
        # Initialize lot_occupancy_df with default values for all days (Monday to Sunday)
        lot_occupancy_data = []
        for lot_id, conditions in lot_conditions.items():
            for day in range(1, 8):  # Days from Monday (1) to Sunday (7)
                lot_occupancy_data.append({
                    "Lot ID": lot_id,
                    "Day": day,
                    "Current Occupancy": conditions["current_occupancy"],
                    "Average Occupancy": conditions["average_occupancy"]
                })
        
        # Create a DataFrame with the initialized data
        lot_occupancy_df = pd.DataFrame(lot_occupancy_data)
    
    return users_df, bookings_df, lot_occupancy_df

def save_simulation_time():
    """Save the current simulation time to a file."""
    with open(simulation_time_file, "w") as file:
        json.dump(simulation_time, file)

def load_simulation_time():
    """Load the simulation time from a file if it exists."""
    global simulation_time
    if os.path.exists(simulation_time_file):
        with open(simulation_time_file, "r") as file:
            simulation_time = json.load(file)
        print(f"Simulation time loaded: Day {simulation_time['day']}, Hour {simulation_time['hour']:02}:{simulation_time['minute']:02}")

def save_data(users_df, bookings_df, lot_occupancy_df):
    users_df.to_csv(users_file, index=False)
    bookings_df.to_csv(bookings_file, index=False) 
    user_booking_columns = ["Day", "User ID", "Booking Number", "Lot ID", "Start Hour", "End Hour", "Current Occupancy", "Average Occupancy",
        "F", "M", "Reserved", "Baccumulated", "Bcurrent", "O", "Overstay Start Hour", "Overstay End Hour", "Overstay Hours"]
    user_bookings_df = bookings_df[user_booking_columns]
    user_bookings_df.to_csv(bookings_only_file, index=False)
    lot_occupancy_df.to_csv(lot_occupancy_file, index=False)

def get_current_occupancy(lot_id, lot_occupancy_df, day):
    day_occupancy = lot_occupancy_df[(lot_occupancy_df["Lot ID"] == lot_id) & (lot_occupancy_df["Day"] == day)]
    if not day_occupancy.empty:
        return day_occupancy.iloc[-1]["Current Occupancy"]
    else:
        # If no occupancy data exists for the day, fall back to initial conditions
        return lot_conditions[lot_id]["current_occupancy"]

def create_user(users_df):
    name = input("Enter new user's name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return users_df
    if users_df.empty:
        user_id = 1
    else:
        user_id = users_df["User ID"].max() + 1
    new_user = {
        "User ID": user_id,
        "User Name": name,
        "F": 0,  # Frequency of visits
        "Ttotal": 0,  # Total time spent (hours)
        "M": 0,  # Number of missed reservations
        "Baccumulated": 0,  # Accumulated bonus points
        "total_reservations": 0,  # Total number of reservations
        "num_overstays": 0  # Number of overstays
    }
    # Use pd.concat to add the new user
    users_entry_df = pd.DataFrame([new_user])
    users_df = pd.concat([users_df, users_entry_df], ignore_index=True)
    
    print(f"User '{name}' created with User ID {user_id}.")
    return users_df

def calculate_th(lot_id, day_of_week, hour):
    conditions = lot_conditions.get(lot_id)
    if conditions:
        peak_days = conditions["peak_days"]
        peak_hours = conditions["peak_hours"]
        Th_peak = conditions["Th_peak"]
        Th_off = conditions["Th_off"]
        
        if day_of_week in peak_days and peak_hours[0] <= hour <= peak_hours[1]:
            return Th_peak
        else:
            return Th_off
    return 1

def calculate_dh(current_occupancy, average_occupancy):
    avg = max(int(average_occupancy), 1)  # treat avg as capacity; avoid /0
    ratio = current_occupancy / avg
    return max(0.7, ratio)

def calculate_ph(Pbase, Dh, Th, W):
    return Pbase * Dh * Th * W

def calculate_loyalty_discount(F, Ttotal, start_hour, end_hour, M, total_reservations, O, Bcurrent, Baccumulated):
    gamma = M / total_reservations if total_reservations > 0 else 0
    if total_reservations < 5:
        Dloyalty_adj = 0 - gamma * M
    else:
        Dloyalty_adj = alpha * (F+1) + beta * (Ttotal + end_hour - start_hour) - gamma * M
    if Dloyalty_adj < Dmax:
        Dloyalty_final = Dloyalty_adj + (Bcurrent * O) + Baccumulated
        return Dloyalty_final, gamma
    else:
        return Dmax, gamma

def calculate_overstay_penalty(lot_id, day_of_week, Pbase, Dh, Th, W,
                               overstay_start_hour, overstay_end_hour,
                               num_overstays, total_reservations):
    lambda_factor = (1 + num_overstays / total_reservations) if total_reservations > 0 else 1
    Poverstay_total = 0
    if overstay_start_hour is None or overstay_end_hour is None:
        return 0

    start = int(overstay_start_hour)
    end = min(int(overstay_end_hour), 24)
    for hour in range(start, end):
        Th_overstay = calculate_th(lot_id, day_of_week, hour)
        Ph_extra = calculate_ph(Pbase, Dh, Th_overstay, W)
        Poverstay_total += 0.7 * Ph_extra

    return lambda_factor * Poverstay_total

def get_available_lots(day, lot_occupancy_df):
    available_lots = []
    for lot_id, conditions in lot_conditions.items():  # Iterate over all lots
        # Find the current occupancy for the specific lot and day in lot_occupancy_df
        lot_day_data = lot_occupancy_df[(lot_occupancy_df["Lot ID"] == lot_id) & (lot_occupancy_df["Day"] == day)]
        
        if not lot_day_data.empty:
            current_occupancy = lot_day_data["Current Occupancy"].values[0]
        else:
            # Use the default occupancy from lot_conditions if no data is found
            current_occupancy = conditions["current_occupancy"]
        
        average_occupancy = conditions["average_occupancy"]
        
        # Check if the current occupancy is less than the average occupancy
        if current_occupancy < average_occupancy:
            available_lots.append(lot_id)
    
    return available_lots

def time_based_dynamic_pricing(lot_id, start_hour, end_hour, day_of_week, current_occupancy, average_occupancy):
    # Calculate Th and Ph for each hour of booking
    total_price = 0
    for hour in range(start_hour, end_hour):
        Th = calculate_th(lot_id, day_of_week, hour)
        Dh = calculate_dh(current_occupancy, average_occupancy)
        W = 1.3 if day_of_week >= 6 else 1.0
        Ph = calculate_ph(Pbase_values[lot_id], Dh, Th, W)
        total_price += Ph
    return total_price,Th, Dh, W, Ph

def time_based_static_pricing(lot_id, start_hour, end_hour):
    Poff_peak = Pbase_values[lot_id]
    Ppeak = Onpeak*Poff_peak
    total_price = 0
    for hour in range(start_hour, end_hour):
        if lot_conditions[lot_id]["peak_hours"][0] <= hour <= lot_conditions[lot_id]["peak_hours"][1]:
            total_price += Ppeak
        else:
            total_price += Poff_peak
    return total_price

def static_pricing(lot_id, start_hour, end_hour):
    return Pbase_values[lot_id] * (end_hour - start_hour)

def flat_time_dynamic_pricing(lot_id, start_hour, end_hour, day_of_week, current_occupancy, average_occupancy):
    W = 1.3 if day_of_week >= 6 else 1.0
    Pbase = Pbase_values[lot_id]
    Dh = calculate_dh(current_occupancy, average_occupancy)
    total_price = 0
    for _ in range(start_hour, end_hour):
        total_price += Pbase * Dh * W
    return total_price

def calculate_final_price(Ptotal, Dloyalty_final, Poverstay_total):
    return Ptotal - (min(Dloyalty_final, Dmax) / 100) * Ptotal + Poverstay_total

def check_availability(bookings_df, lot_id, day):
    lot_day_bookings = bookings_df[(bookings_df["Lot ID"] == lot_id) & (bookings_df["Day"] == day)]
    available_hours = list(range(0, 24))
    
    for _, booking in lot_day_bookings.iterrows():
        booked_hours = list(range(booking["Start Hour"], booking["End Hour"]))
        available_hours = [hour for hour in available_hours if hour not in booked_hours]
    
    return available_hours

def decrement_occupancy(lot_id, simulation_time_day, simulation_time_hour, users_df, bookings_df, lot_occupancy_df):
    current_day = simulation_time_day
    current_hour = simulation_time_hour

    # bookings that need an occupancy decrement (end passed; not yet flagged)
    expired_bookings = bookings_df[
        (bookings_df["Day"] == current_day) &
        (bookings_df["Lot ID"] == lot_id) &
        (
            # no overstay: end passed and not decremented
            (bookings_df["Overstay End Hour"].isna() & (bookings_df["End Hour"] <= current_hour) & (~bookings_df["decremented"].fillna(False))) |
            # with overstay: overstay end passed and not decremented_overstay
            (~bookings_df["Overstay End Hour"].isna() & (bookings_df["Overstay End Hour"] <= current_hour) & (~bookings_df["decremented_overstay"].fillna(False)))
        )
    ]

    lot_mask = (lot_occupancy_df["Lot ID"] == lot_id) & (lot_occupancy_df["Day"] == current_day)

    for _, booking in expired_bookings.iterrows():
        # decrement occupancy and clamp at 0
        lot_occupancy_df.loc[lot_mask, "Current Occupancy"] = (
            lot_occupancy_df.loc[lot_mask, "Current Occupancy"] - 1
        ).clip(lower=0)

        # set the right boolean flag (don't use isna on booleans)
        if pd.isna(booking["Overstay End Hour"]):
            bookings_df.loc[bookings_df["Booking Number"] == booking["Booking Number"], "decremented"] = True
        else:
            bookings_df.loc[bookings_df["Booking Number"] == booking["Booking Number"], "decremented_overstay"] = True

    if not lot_occupancy_df.loc[lot_mask, "Current Occupancy"].empty:
        print(
            f'Occupancy decremented for Lot {lot_id} on day {current_day}. '
            f'Current occupancy: {int(lot_occupancy_df.loc[lot_mask, "Current Occupancy"].values[0])}'
        )

    save_data(users_df, bookings_df, lot_occupancy_df)

def advance_time(users_df, bookings_df, lot_occupancy_df):
    global simulation_time
    simulation_time["minute"] += time_increment

    if simulation_time["minute"] >= 60:
        simulation_time["minute"] -= 60
        simulation_time["hour"] += 1

    if simulation_time["hour"] >= 24:
        simulation_time["hour"] -= 24
        simulation_time["day"] = 1 if simulation_time["day"] == 7 else simulation_time["day"] + 1

    # decrement only for the current simulated day
    for lot_id in lot_conditions.keys():
        decrement_occupancy(lot_id, simulation_time["day"], simulation_time["hour"], users_df, bookings_df, lot_occupancy_df)

def get_simulation_time():
    # """Retrieve the current simulation time as a formatted string."""
    return f"Day {simulation_time['day']} (Hour {simulation_time['hour']:02}:{simulation_time['minute']:02})"

def will_lot_be_available(bookings_df, users_df, lot_occupancy_df, lot_id, day, start_hour, end_hour):
    # seed from recorded occupancy for that lot/day
    lot_day = lot_occupancy_df[(lot_occupancy_df["Lot ID"] == lot_id) & (lot_occupancy_df["Day"] == day)]
    current = int(lot_day["Current Occupancy"].iloc[0]) if not lot_day.empty else lot_conditions[lot_id]["current_occupancy"]
    capacity = int(lot_conditions[lot_id]["average_occupancy"])  # you defined avg == capacity

    # consider overlapping RESERVED bookings on that day
    lot_day_bookings = bookings_df[(bookings_df["Lot ID"] == lot_id) & (bookings_df["Day"] == day)]
    for _, b in lot_day_bookings.iterrows():
        overlaps = not (end_hour <= b["Start Hour"] or start_hour >= b["End Hour"])
        if overlaps and b.get("Reserved", 0) == 1:
            current += 1
            if current > capacity:
                return False
    return True

def add_booking(users_df, bookings_df, lot_occupancy_df):
    current_day = simulation_time["day"]
    current_hour = simulation_time["hour"]
    # Select or create user
    # user_choice = input("Do you want to add a booking for an existing user? (yes/no): ").strip().lower()
    user_choice = 'yes'
    if user_choice == 'yes':
        # user_id = int(input("Enter existing User ID: "))
        user_id = random.randint(1, 10)
        if user_id not in users_df["User ID"].values:
            print("User ID not found.")
            return users_df, bookings_df, lot_occupancy_df
    else:
        return users_df, bookings_df, lot_occupancy_df
    
    # day_of_week = int(input("Enter day of the week (1=Monday, ..., 7=Sunday): "))
    day_of_week = random.randint(current_day, 7)
    if day_of_week not in range(1, 8):
        print("Day must be between 1 and 7.")
        return users_df, bookings_df, lot_occupancy_df
    
    # Check if the day is in the past
    if day_of_week < current_day:
        print("You cannot book for a past day. Booking denied.")
        return users_df, bookings_df, lot_occupancy_df

    # Input booking details
    # lot_id = int(input("Enter Lot ID (1-5): "))
    lot_id = random.randint(1, 5)
    if lot_id not in lot_conditions:
        print("Invalid Lot ID.")
        return users_df, bookings_df, lot_occupancy_df
    
    try:
        # start_hour = int(input("Enter booking start hour (0-23): "))
        # end_hour = int(input("Enter booking end hour (1-24): "))
        start_hour = random.randint(current_hour, 23)
        end_hour = random.randint(start_hour+1, 24)
        if not (0 <= start_hour <= 23 and 1 <= end_hour <= 24 and start_hour < end_hour):
            print("Invalid start or end hour.")
            return users_df, bookings_df, lot_occupancy_df
        # Check if the booking is for the current day and start_hour is valid
        if day_of_week == current_day and start_hour < current_hour:
            print(f"You cannot book for a start hour earlier than the current hour ({current_hour}). Booking denied.")
            return users_df, bookings_df, lot_occupancy_df
    except ValueError:
        print("Invalid input for hours.")
        return users_df, bookings_df, lot_occupancy_df
    
    # Retrieve occupancy values from lot conditions
    lot_data = lot_occupancy_df[(lot_occupancy_df["Lot ID"] == lot_id) & (lot_occupancy_df["Day"] == day_of_week)]
    current_occupancy = lot_data["Current Occupancy"].values[0]  # Get current occupancy for the specific day
    average_occupancy = lot_conditions[lot_id]["average_occupancy"]
    
    try:
        # Check if the reservation was missed and increment M if true
        # missed_reservation = input("Was the reservation missed? (yes/no): ").strip().lower()
        options = ['yes', 'no']
        weights = [0.05, 0.95]  # 70% chance for 'yes', 30% chance for 'no'
        # Choose randomly with weights
        missed_reservation = random.choices(options, weights=weights, k=1)[0]
        if missed_reservation == 'yes':
            M = users_df.loc[users_df["User ID"] == user_id, "M"].values[0] + 1
            R = 0
        else:
            R = 1
            # Check if lot will have availability
            if not will_lot_be_available(bookings_df, users_df, lot_occupancy_df, lot_id, day_of_week, start_hour, end_hour):
                print(f"The lot {lot_id} will be full during the requested time. Booking denied.")
                # List other available lots
                available_lots = get_available_lots(day_of_week, lot_occupancy_df)
                if available_lots:
                    print(f"Available lots for day {day_of_week}:", ', '.join(map(str, available_lots)))
                else:
                    print(f"No available lots for day {day_of_week}.")
                log_request(day_of_week, lot_id, start_hour, end_hour, "denied")   # <-- ADD THIS
                return users_df, bookings_df, lot_occupancy_df
            else:
                log_request(day_of_week, lot_id, start_hour, end_hour, "accepted") # <-- ADD THIS
                M = users_df.loc[users_df["User ID"] == user_id, "M"].values[0]
    except ValueError:
        print("Invalid input for missed reservations.")
        return users_df, bookings_df, lot_occupancy_df
    
    try:
        # Bcurrent = float(input("Enter current bonus for inconvenience (Bcurrent) in % (e.g., 5 for 5%): "))
        options = [0, 1, 3]
        weights = [0.9, 0.07, 0.03]  # 70% chance for 'yes', 30% chance for 'no'
        # Choose randomly with weights
        Bcurrent = random.choices(options, weights=weights, k=1)[0]
        if not (0 <= Bcurrent <= 100):
            print("Bcurrent must be between 0 and 100.")
            return users_df, bookings_df, lot_occupancy_df
    except ValueError:
        print("Invalid input for Bcurrent.")
        return users_df, bookings_df, lot_occupancy_df
    
    try:
        # overstay_choice = input("Did you overstay? (yes/no): ").strip().lower()
        options = ['yes', 'no']
        weights = [0.2, 0.8]  # 70% chance for 'yes', 30% chance for 'no'
        # Choose randomly with weights
        overstay_choice = random.choices(options, weights=weights, k=1)[0]
        if overstay_choice == 'yes':
            # overstay_start_hour = int(input("Enter overstay start hour (0-23): "))
            # overstay_end_hour = int(input("Enter overstay end hour (1-24): "))
            overstay_start_hour = end_hour
            overstay_end_hour = random.randint(overstay_start_hour+1, 24)
            if not (0 <= overstay_start_hour <= 23 and 1 <= overstay_end_hour <= 24 and overstay_start_hour < overstay_end_hour):
                print("Invalid overstay start or end hour.")
                return users_df, bookings_df, lot_occupancy_df
            overstay_hours = overstay_end_hour - overstay_start_hour
            # Check if the overstay is for the current day and valid
            if day_of_week == current_day and overstay_start_hour < current_hour:
                print(f"You cannot set an overstay start hour earlier than the current hour ({current_hour}). Booking denied.")
                return users_df, bookings_df, lot_occupancy_df
            if not will_lot_be_available(bookings_df, users_df, lot_occupancy_df, lot_id, day_of_week, overstay_start_hour, overstay_end_hour):
                print("Overstay denied. The lot is full.")
                log_request(day_of_week, lot_id, overstay_start_hour, overstay_end_hour, "overstay_denied")
                O = 0
            else:
                O = 1
        else:
            overstay_start_hour = None
            overstay_end_hour = None
            overstay_hours = 0
            O = 0
    except ValueError:
        print("Invalid input for overstay hours.")
        return users_df, bookings_df, lot_occupancy_df
    
    if current_occupancy < average_occupancy:
        lot_occupancy_df.loc[(lot_occupancy_df["Lot ID"] == lot_id) & (lot_occupancy_df["Day"] == day_of_week),"Current Occupancy"] += 1
            
    # Retrieve user cumulative data
    user_row = users_df.loc[users_df["User ID"] == user_id].iloc[0]
    F = user_row["F"]
    Ttotal = user_row["Ttotal"]
    Baccumulated = user_row["Baccumulated"]
    total_reservations = user_row["total_reservations"] + 1
    num_overstays = user_row["num_overstays"] + (1 if O == 1 else 0)

    # Model Calculations
    # Time-Based Dynamic Pricing
    Ptotal_model1, Th, Dh, W, Ph = time_based_dynamic_pricing(lot_id, start_hour, end_hour, day_of_week, current_occupancy, average_occupancy)
    # Flat-Time Dynamic Pricing
    Ptotal_model2 = flat_time_dynamic_pricing(lot_id, start_hour, end_hour, day_of_week, current_occupancy, average_occupancy)
    # Static Pricing
    Ptotal_model3 = static_pricing(lot_id, start_hour, end_hour)
    # Time-Based Static Pricing
    Ptotal_model4 = time_based_static_pricing(lot_id, start_hour, end_hour)
    
     # Calculate Dloyalty
    Dloyalty_final, gamma = calculate_loyalty_discount(F, Ttotal, start_hour, end_hour, M, total_reservations, O, Bcurrent, Baccumulated)

    # Calculate overstay penalty
    if O == 1 and overstay_start_hour is not None and overstay_end_hour is not None:
        Poverstay_total = calculate_overstay_penalty(
            lot_id,
            day_of_week,
            Pbase=Pbase_values[lot_id],
            Dh=Dh,
            Th=Th,
            W=W,
            overstay_start_hour=overstay_start_hour,
            overstay_end_hour=overstay_end_hour,
            num_overstays=num_overstays,
            total_reservations=total_reservations
        )
    else:
        Poverstay_total = 0

    # Model Calculations
    # Time-Based Dynamic Pricing
    Pfinal_model1 = calculate_final_price(Ptotal_model1, Dloyalty_final, Poverstay_total)
    # Flat-Time Dynamic Pricing
    Pfinal_model2 = calculate_final_price(Ptotal_model2, Dloyalty_final, Poverstay_total)
    # Static Pricing
    Pfinal_model3 = calculate_final_price(Ptotal_model3, Dloyalty_final, Poverstay_total)
    # Time-Based Static Pricing
    Pfinal_model4 = calculate_final_price(Ptotal_model4, Dloyalty_final, Poverstay_total)
    
    # Update Baccumulated
    previous_M = user_row["M"]
    if M > previous_M:
        Baccumulated += Bcurrent
    else:
        Baccumulated = 0
    
    # Update user cumulative data
    users_df.loc[users_df["User ID"] == user_id, ["F", "Ttotal", "M", "Baccumulated", "total_reservations", "num_overstays"]] = [
        F + 1,  # Frequency increment
        Ttotal + (end_hour - start_hour),  # Total time spent
        M,  # Update missed reservations
        Baccumulated,  # Update accumulated bonus
        total_reservations,  # Update total reservations
        num_overstays  # Update number of overstays
    ]
    
    # Determine Booking Number
    booking_number = bookings_df[bookings_df["User ID"] == user_id]["Booking Number"].max()
    if pd.isna(booking_number):
        booking_number = 1
    else:
        booking_number += 1
    
    # Create booking entry
    booking_entry = {
        "Day": day_of_week,
        "User ID": user_id,
        "Booking Number": booking_number,
        "Lot ID": lot_id,
        "Start Hour": start_hour,
        "End Hour": end_hour,
        "Pbase": Pbase_values[lot_id],
        "Current Occupancy": current_occupancy,
        "Average Occupancy": average_occupancy,
        "Dh": Dh,
        "Th": Th,
        "W": W,
        "Ph": Ph,
        "Ptotal_Model1": Ptotal_model1,
        "Ptotal_Model2": Ptotal_model2,
        "Ptotal_Model3": Ptotal_model3,
        "Ptotal_Model4": Ptotal_model4,
        "F": F + 1,
        "M": M,
        "Reserved": R,
        "gamma": gamma,
        "Baccumulated": Baccumulated,
        "Bcurrent": Bcurrent,
        "O": O,
        "Dloyalty_final": Dloyalty_final,
        "Overstay Start Hour": overstay_start_hour,
        "Overstay End Hour": overstay_end_hour,
        "Overstay Hours": overstay_hours,
        "Poverstay_total": Poverstay_total,
        "Pfinal_Model1": Pfinal_model1,
        "Pfinal_Model2": Pfinal_model2,
        "Pfinal_Model3": Pfinal_model3,
        "Pfinal_Model4": Pfinal_model4,
        "decremented": False,
        "decremented_overstay": False

    }
    
    # Create booking entry as a DataFrame
    booking_entry_df = pd.DataFrame([booking_entry])

    # Concatenate with bookings_df, ensuring consistency in data types
    bookings_df = pd.concat([bookings_df, booking_entry_df], ignore_index=True)
    bookings_df["decremented"] = bookings_df["decremented"].fillna(False).astype(bool)
    bookings_df["decremented_overstay"] = bookings_df["decremented_overstay"].fillna(False).astype(bool)
        
    # Save data to CSV
    save_data(users_df, bookings_df, lot_occupancy_df)
    
    print("\nBooking successfully added!")
    print(f"Final Price (Time based dynamic pricing): ₦{Pfinal_model1:.2f}")
    print(f"Final Price (Flat time dynamic pricing): ₦{Pfinal_model2:.2f}")
    print(f"Final Price (Time based static pricing): ₦{Pfinal_model4:.2f}")
    print(f"Final Price (Static pricing): ₦{Pfinal_model3:.2f}")
    return users_df, bookings_df, lot_occupancy_df

def main():
    users_df, bookings_df, lot_occupancy_df = load_data()
    
    while True:
        print("\n--- Smart Parking Dynamic Pricing System ---")
        print("0. Enter 0 to generate 100 bookings. \n1. Enter 1 to cancel")
        
        choice = input("Enter your choice (0-1): ").strip()
        
        if choice == '0':
            for i in range(int(bookings_count)):
                users_df, bookings_df, lot_occupancy_df = add_booking(users_df, bookings_df, lot_occupancy_df)
                # Advance the simulation time automatically at the end of the loop
                advance_time(users_df, bookings_df, lot_occupancy_df)
                save_simulation_time()
            summarize_metrics(bookings_df, lot_conditions)
        if choice == '1':
            break
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
