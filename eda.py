import pandas as pd
import matplotlib.pyplot as plt


dev = pd.read_csv("data/dev.csv")
eval_df = pd.read_csv("data/eval.csv")

target = "default.payment.next.month"

target_counts = dev[target].value_counts().sort_index()
default_count = int(target_counts.loc[1])
default_rate = default_count / len(dev) * 100

duplicate_rows = int(dev.duplicated().sum() + eval_df.duplicated().sum())
duplicate_ids = int(dev["ID"].duplicated().sum() + eval_df["ID"].duplicated().sum())
id_overlap = len(set(dev["ID"]) & set(eval_df["ID"]))

expected_cols = [c for c in dev.columns if c != target]
missing_expected = int(
    dev[expected_cols + [target]].isna().sum().sum()
    + eval_df[expected_cols].isna().sum().sum()
)

limit_bins = pd.qcut(dev["LIMIT_BAL"], q=6, duplicates="drop")
limit_rates = dev.groupby(limit_bins, observed=True)[target].mean() * 100
limit_low = limit_rates.iloc[0]
limit_high = limit_rates.iloc[-1]

recent_delay = dev["PAY_0"] > 0
delay_rates = dev.groupby(recent_delay)[target].mean() * 100
no_delay_rate = delay_rates.loc[False]
delay_rate = delay_rates.loc[True]

summary = pd.DataFrame(
    {
        "Metric": [
            "Development shape",
            "Evaluation shape",
            "Default distribution",
            "Duplicate rows / duplicate IDs",
            "Development/evaluation ID overlap",
            "Missing values in expected columns",
            "Lowest vs. highest LIMIT_BAL bin default rate",
            "Recent delay default rate, delayed vs. not delayed",
        ],
        "Value": [
            f"{dev.shape[0]:,} rows, {dev.shape[1]} cols.",
            f"{eval_df.shape[0]:,} rows, {eval_df.shape[1]} cols.",
            f"{default_count:,} / {len(dev):,}, {default_rate:.1f}%",
            f"{duplicate_rows} / {duplicate_ids}",
            f"{id_overlap}",
            f"{missing_expected}",
            f"{limit_low:.1f}% vs. {limit_high:.1f}%",
            f"{delay_rate:.1f}% vs. {no_delay_rate:.1f}%",
        ],
    }
)

summary.to_csv("assets/eda_summary_table.csv", index=False)
print(summary)

colors = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "teal": "#72B7B2",
    "red": "#E45756",
}

fig, axes = plt.subplots(2, 1, figsize=(4.2, 5.0), constrained_layout=True)

shares = target_counts / len(dev) * 100

bars = axes[0].bar(
    ["Non-default", "Default"],
    [shares.loc[0], shares.loc[1]],
    color=[colors["blue"], colors["orange"]],
    width=0.58,
)

axes[0].set_title("Target distribution", pad=12)
axes[0].set_ylabel("Share of development set (%)")
axes[0].set_ylim(0, 90)
axes[0].spines[["top", "right"]].set_visible(False)

for bar, cls in zip(bars, [0, 1]):
    height = bar.get_height()
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        height - 4,
        f"{height:.1f}%\n(n={target_counts.loc[cls]:,})",
        ha="center",
        va="top",
        color="white",
        fontweight="bold",
    )

bars = axes[1].bar(
    ["No recent delay", "Recent delay"],
    [no_delay_rate, delay_rate],
    color=[colors["teal"], colors["red"]],
    width=0.58,
)

axes[1].set_title("Default rate by PAY_0 delay status", pad=12)
axes[1].set_ylabel("Default rate (%)")
axes[1].set_ylim(0, 60)
axes[1].spines[["top", "right"]].set_visible(False)

for bar in bars:
    height = bar.get_height()
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        height - 3,
        f"{height:.1f}%",
        ha="center",
        va="top",
        color="white",
        fontweight="bold",
    )

fig.savefig("assets/report_eda_summary.pdf")
plt.show()