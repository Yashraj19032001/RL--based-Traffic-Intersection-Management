import pandas as pd

# Load the file, ignoring problematic lines
df = pd.read_csv("rl_waiting_times.csv", header=None, on_bad_lines='skip')

# Optionally save the cleaned data for future use
df.to_csv("rl_waiting_times_cleaned.csv", index=False, header=False)
