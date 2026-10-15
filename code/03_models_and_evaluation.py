    n = len(y_true); prev = np.mean(y_true); nb_m, nb_a = [], []
    for pt in thresholds:
        pr = (y_prob >= pt).astype(int)
        tp = np.sum((pr == 1) & (y_true == 1)); fp = np.sum((pr == 1) & (y_true == 0))
        nb_m.append(tp / n - fp / n * (pt / (1 - pt)))
        nb_a.append(prev - (1 - prev) * (pt / (1 - pt)))
    return np.array(nb_m), np.array(nb_a)

thresholds = np.round(np.arange(0.01, 0.51, 0.01), 2)
nb_model, nb_all = decision_curve(yt, p_t3x, thresholds)
print("\n=== DECISION CURVE, Tier 3 XGBoost (pooled OOF) ===")
dca = pd.DataFrame({"pt": thresholds, "NB_model": nb_model, "NB_treat_all": nb_all,
                    "advantage": nb_model - nb_all})
print(dca[dca["pt"].isin([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])].round(4).to_string(index=False))

# ---------- Figure ----------
PLOT_MODEL = "XGBoost"
colors = {"Model 1 (Literature-Replication Baseline)": "#999999", T2: "#1f77b4", T3: "#d62728"}
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6), dpi=300)
ax = axes[0]
for tier_name in tiers:
    y_t, y_p = oof_pool[tier_name][PLOT_MODEL]
    fpr, tpr, _ = roc_curve(y_t, y_p)
    ax.plot(fpr, tpr, label=f"{tier_name.split(' (')[0]} ({roc_auc_score(y_t, y_p):.3f})",
            color=colors[tier_name], linewidth=1.3)
ax.plot([0, 1], [0, 1], ':', color='black', linewidth=0.8)
ax.set_xlabel("FPR", fontsize=8); ax.set_ylabel("TPR", fontsize=8); ax.set_title("(a) ROC", fontsize=9)
ax.legend(fontsize=5, loc='lower right'); ax.tick_params(labelsize=6)

ax = axes[1]
frac_pos, mean_pred = calibration_curve(yt, p_t3x, n_bins=10, strategy='quantile')
ax.plot(mean_pred, frac_pos, marker='s', markersize=3, color=colors[T3],
        label=f"Brier={brier_score_loss(yt, p_t3x):.3f}, ECE={compute_ece(yt, p_t3x):.3f}")
ax.plot([0, 1], [0, 1], ':', color='black', linewidth=0.8)
ax.set_xlabel("Mean predicted prob.", fontsize=8); ax.set_ylabel("Fraction positive", fontsize=8)
ax.set_title("(b) Reliability (Tier 3)", fontsize=9); ax.legend(fontsize=5, loc='lower right'); ax.tick_params(labelsize=6)

ax = axes[2]
ax.plot(thresholds, nb_model, color=colors[T3], linewidth=1.3, label="Model")
ax.plot(thresholds, nb_all, ':', color='gray', linewidth=1, label="Treat all")
ax.axhline(0, linestyle=':', color='black', linewidth=0.8, label="Treat none")
ax.set_ylim(-0.02, max(nb_model.max(), nb_all.max()) * 1.05)
ax.set_xlabel("Threshold probability", fontsize=8); ax.set_ylabel("Net benefit", fontsize=8)
ax.set_title("(c) Decision Curve (Tier 3)", fontsize=9); ax.legend(fontsize=5, loc='upper right'); ax.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig4_ieee_model_evaluation.pdf", bbox_inches='tight')
plt.savefig(RESULTS_DIR / "fig4_ieee_model_evaluation.png", bbox_inches='tight')
plt.show()

with open(RESULTS_DIR / "oof_pool.pkl", "wb") as f:
    pickle.dump(oof_pool, f)

# ---------- Subgroups (adults only; NHANES top-codes age at 80) ----------
df_clean['age_group'] = pd.cut(df_clean['age'], bins=[20, 40, 60, 80, 200], right=False,
                               labels=['20-39', '40-59', '60-79', '80+'])

def subgroup_table(y_true, y_prob, groups, n_boot=500):
    rows = []
    for g in sorted(groups.dropna().unique(), key=str):
        m = (groups == g).values
        if m.sum() < 20 or len(set(y_true[m])) < 2:
            continue
        yg, pg = y_true[m], y_prob[m]; bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(yg), len(yg))
            if yg[i].min() != yg[i].max():
                bs.append(roc_auc_score(yg[i], pg[i]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        s, c = calib_slope_intercept(yg, pg)
        rows.append({"Subgroup": g, "N": int(m.sum()), "Prevalence": round(yg.mean(), 3),
                     "AUROC [95% CI]": f"{roc_auc_score(yg, pg):.3f} [{lo:.3f}, {hi:.3f}]",
                     "Brier": round(brier_score_loss(yg, pg), 4), "Cal slope": round(s, 2)})
    return pd.DataFrame(rows)

print("\n=== SUBGROUPS: Tier 3 XGBoost ===")
print("\n--- By sex ---")
print(subgroup_table(yt, p_t3x, df_clean['gender'].map({0: 'Female', 1: 'Male'})).to_string(index=False))
print("\n--- By age group ---")
print(subgroup_table(yt, p_t3x, df_clean['age_group']).to_string(index=False))
eth_cols = [c for c in df_clean.columns if c.startswith('ethnicity_')]
if eth_cols:
    print("\n--- By ethnicity ---")
    print(subgroup_table(yt, p_t3x, df_clean[eth_cols].idxmax(axis=1).str.replace('ethnicity_', '')).to_string(index=False))

# ---------- Benchmarks: minimal rule + stronger transparent model ----------
Xb_all = X.assign(age_sq=X["age"] ** 2)
def run_benchmark(cols):
    Xb = Xb_all[cols]; oof = np.zeros(len(Xb)); fold_auc = []
    for tr, va in outer_cv.split(Xb, y):
        prep = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        Xtr, Xva = prep.fit_transform(Xb.iloc[tr]), prep.transform(Xb.iloc[va])
        s = RandomizedSearchCV(Pipeline([('clf', LogisticRegression(max_iter=2000, random_state=42))]),
                               {'clf__C': [0.01, 0.1, 1.0, 10.0]}, n_iter=4, scoring='roc_auc', cv=inner_cv, random_state=42)
        s.fit(Xtr, y.iloc[tr])
        best = s.best_estimator_.named_steps['clf'].fit(Xtr, y.iloc[tr])
        cal = CalibratedClassifierCV(best, method='isotonic', cv=3).fit(Xtr, y.iloc[tr])
        oof[va] = cal.predict_proba(Xva)[:, 1]; fold_auc.append(roc_auc_score(y.iloc[va], oof[va]))
    return oof, fold_auc

print("\n=== BENCHMARKS ===")
for name, cols in [("Minimal (age, diabetes, SBP)", ["age", "diabetes_diagnosed", "bp_systolic"]),
                   ("Stronger LR (age, age^2, sex, diabetes, SBP, DBP, BMI, ever_smoked)",
                    ["age", "age_sq", "gender", "diabetes_diagnosed", "bp_systolic", "bp_diastolic", "bmi", "ever_smoked"])]:
    oof_b, fa = run_benchmark(cols)
    m, ci = paired_auc_diff(yt, p_t3x, oof_b)
    print(f"{name}\n  fold AUROC {np.mean(fa):.4f} +/- {np.std(fa):.4f} | pooled AUROC {roc_auc_score(yt, oof_b):.4f} "
          f"| Brier {brier_score_loss(yt, oof_b):.4f} | T3-XGB minus this: {m:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")
