def plot_learning_curves_F2(df, out_dir):
    """
    Figure F2 (Section 10): Learning Curves vs Target Data Budget (Absolute Samples), with mean +/- 1 std bands.
    """
    targets = df["target"].unique()

    for tgt in targets:
        sub_tgt = df[df["target"] == tgt].copy()
        
        # Determine budget identifier and absolute samples
        budget_col = "budget_tag" if "budget_tag" in sub_tgt.columns else "n_target"
        unique_budgets = sub_tgt[budget_col].unique()
        if len(unique_budgets) <= 1:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
        strats = sub_tgt["strategy"].unique()

        # Grouping by budget and absolute sample count
        group_keys = [budget_col]
        if "n_samples" in sub_tgt.columns:
            group_keys.append("n_samples")

        for strat in strats:
            s_df = sub_tgt[sub_tgt["strategy"] == strat]
            agg = s_df.groupby(group_keys)[["mean_r2", "mape"]].agg(["mean", "std"]).reset_index()

            # Sort numerically by sample count
            if "n_samples" in agg.columns:
                agg["sort_key"] = agg["n_samples"].astype(int)
                x_labels = [f"{row[budget_col]} (N={row['n_samples']})" if str(row[budget_col]).lower() == "full" else f"N={row['n_samples']}" for _, row in agg.iterrows()]
            else:
                agg["sort_key"] = agg[budget_col].apply(lambda x: 9999 if str(x) == "full" else int(x))
                x_labels = agg[budget_col].astype(str).tolist()

            agg = agg.sort_values("sort_key")
            x_pos = np.arange(len(agg))

            r2_mean = agg[("mean_r2", "mean")]
            r2_std = agg[("mean_r2", "std")].fillna(0)
            mape_mean = agg[("mape", "mean")]
            mape_std = agg[("mape", "std")].fillna(0)

            col = COLOR_MAP.get(strat, "#000000")
            lab = STRATEGY_LABELS.get(strat, strat)

            ax1.plot(x_pos, r2_mean, marker="o", label=lab, color=col, linewidth=2)
            ax1.fill_between(x_pos, r2_mean - r2_std, r2_mean + r2_std, alpha=0.15, color=col)

            ax2.plot(x_pos, mape_mean, marker="s", label=lab, color=col, linewidth=2)
            ax2.fill_between(x_pos, mape_mean - mape_std, mape_mean + mape_std, alpha=0.15, color=col)

        for ax, title, ylabel in [(ax1, f"Mean $R^2$ vs Budget ({tgt})", "Mean $R^2$ (Test Set)"),
                                  (ax2, f"MAPE vs Budget ({tgt})", "MAPE (%)")]:
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Target Data Budget: Absolute Sample Count (Train + Val)", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=15)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=9)

        plt.suptitle(f"Figure F2: Learning Curves on Unseen Test Set - Target {tgt}", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"fig_F2_learning_curves_{tgt}.png")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Generated Figure F2 with absolute sample counts: {out_png}")
