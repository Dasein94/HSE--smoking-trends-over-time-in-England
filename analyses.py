import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import pingouin as pg

import missingno as msno
# -----------------------------
# 1. LOAD DATA
# -----------------------------


df99 = pd.read_stata("HSE_1999.dta", convert_categoricals=False)
df19 = pd.read_stata("HSE_2019.dta", convert_categoricals=False)
print("aqui")
print(df99.shape)
print(df19.shape)
# -----------------------------
# 2. SELECT VARIABLES
# -----------------------------
df99 = df99[["pserial","booklet","cignow", "cigdyal","kcigreg","kcigregg","errorwt"]].copy()
df19 = df19[["SerialA","SCType","cignow_19", "cigdyal","KCigReg", "kcigregg", "wt_int"]].copy()

# rename 
df99 = df99.rename(columns={"errorwt":"weight"})
df19 = df19.rename(columns={"wt_int":"weight"})

df99["year"] = 1999
df19["year"] = 2019

df99["id"] = "1999_" + df99["pserial"].astype(str)
df19["id"] = "2019_" + df19["SerialA"].astype(str)

# -----------------------------
# 3. HANDLE MISSING VALUES 
# -----------------------------
missing_codes = [-99,-98,-97,-96,-95,-94,-93,-92,-91,
                 -9,-8,-7,-6,-5,-4,-3,-2,-1]

for col in df99.columns:
    df99[col] = df99[col].replace(missing_codes, np.nan)

for col in df19.columns:
    df19[col] = df19[col].replace(missing_codes, np.nan)

# -----------------------------
# 4. DEFINE SMOKING (MAIN OUTCOME FOCUSED ON CIGARETTES)
# -----------------------------
#1999 - smoke variable with nans
df99["smoke"] = np.nan

#Adults
df99.loc[df99["cignow"] == 1, "smoke"] = 1

df99.loc[df99["cignow"] == 2, "smoke"] = 0

#Children
df99.loc[df99["kcigregg"] == 1, "smoke"] = 0
df99.loc[df99["kcigregg"].isin([2, 3]), "smoke"] = 1

# 2019 - smoke variable with nans
df19["smoke"] = np.nan

#Adults
df19.loc[df19["cignow_19"] == 1, "smoke"] = 1

df19.loc[df19["cignow_19"] == 2, "smoke"] = 0

#Children
df19.loc[df19["kcigregg"] == 1, "smoke"] = 0
df19.loc[df19["kcigregg"].isin([2, 3]), "smoke"] = 1

# -----------------------------
# 5. DESCRIPTIVES df99 and df19
# -----------------------------
print(df99.shape)
print(df99.columns)
print(df99.head(5))
print(df99.info())
print(df99.isnull().sum())

print(df99.groupby("smoke")["cigdyal"].describe())

print(df19.groupby("smoke")["cigdyal"].describe())


print(df19.shape)
print(df19.columns)
print(df19.head(5))
print(df19.info())
print(df19.isnull().sum())

msno.bar(df99)
print(plt.show())

msno.bar(df19)
print(plt.show())

# -----------------------------
# 6. COMBINE DATASET
# -----------------------------
df = pd.concat([
    df99[["id","year","weight","smoke"]],
    df19[["id","year","weight","smoke"]]
], ignore_index=True)

df = df.dropna(subset=["smoke"])

# -----------------------------
# 7. DESCRIBE df
# -----------------------------
print(df.shape)
print(df.columns)
print(df.head(5))
print(df.info())
print(df.isnull().sum())



# -----------------------------
# 8. WEIGHTED LOGISTIC REGRESSION
# -----------------------------
df["year2019"] = (df["year"] == 2019).astype(int)

model = sm.GLM(
    df["smoke"],
    sm.add_constant(df["year2019"]),
    family=sm.families.Binomial(),
    freq_weights=df["weight"]
).fit()

print(model.summary())
print(np.exp(model.params))

# -----------------------------
# 7. WEIGHTED PREVALENCE
# -----------------------------

for year, group in df.groupby("year"):

    valid = group[group["smoke"].notna()]

    weighted_prev = (
        (valid["smoke"] * valid["weight"]).sum()
        / valid["weight"].sum()
    ) * 100

    unweighted_prev = valid["smoke"].mean() * 100

    smokers = (valid["smoke"] == 1).sum()

    print(
        year,
        "N =", len(valid),
        "Smokers =", smokers,
        "Weighted Prev =", round(weighted_prev, 2),
        "Prev =", round(unweighted_prev, 2)
    )



# -----------------------------
# 8. ADULT ANALYSIS 
# -----------------------------
ad99 = df99[df99["booklet"] == 4]
ad19 = df19[df19["SCType"] == 1] 

ad99 = ad99[ad99["cigdyal"] > 0]
ad19 = ad19[ad19["cigdyal"] > 0]

print(ad99["cigdyal"].describe())
print(ad19["cigdyal"].describe())

print(pg.mwu(
    x=ad99["cigdyal"],
    y=ad19["cigdyal"]
))

# -----------------------------
# 9. YOUTH ANALYSIS 
# -----------------------------
yu99 = df99[df99["booklet"].isin([1,2])]
yu19 = df19[df19["SCType"].isin([2,3])]

valid = [2,3,4,5,6]

yu99 = yu99[yu99["kcigreg"].isin(valid)]
yu19 = yu19[yu19["KCigReg"].isin(valid)]

print(yu99["kcigreg"].describe())
print(yu19["KCigReg"].describe())

print(pg.mwu(
    x=yu99["kcigreg"],
    y=yu19["KCigReg"]
))


# -----------------------------
# 10. Graphics 
# -----------------------------


def weighted_prev_ci(x):
    """Weighted prevalence and 95% CI for a binary outcome.

    Parameters:
        x (DataFrame): Must contain 'smoke' (0/1) and 'weight'.

    Returns:
        Series with prevalence and 95% CI.
    """    
    y = x["smoke"]
    w = x["weight"]

    p = np.average(y, weights=w)

    n_eff = (w.sum() ** 2) / (w**2).sum()
    se = np.sqrt(p * (1 - p) / n_eff)

    return pd.Series({
        "prev": p * 100,
        "lower": (p - 1.96 * se) * 100,
        "upper": (p + 1.96 * se) * 100
    })

results = (
    df.groupby("year")
      .apply(weighted_prev_ci)
      .reset_index()
)

plt.figure(figsize=(4,4))

plt.errorbar(
    results["year"],
    results["prev"],
    yerr=[
        results["prev"] - results["lower"],
        results["upper"] - results["prev"]
    ],
    fmt="o-",
    capsize=5
)

plt.xticks([1999, 2019]) 
plt.ylim(0, max(results["upper"]) + 5)

plt.ylabel("Smoking prevalence (%)")
plt.xlabel("Year")
plt.title("Weighted smoking prevalence (1999 vs 2019)")

print(plt.show())


#ADDITIONAL SENSITIVITY ANALISES - GRAPHS
# Adults 

adult_df = pd.concat([
    pd.DataFrame({"year":"1999", "cigs":ad99["cigdyal"]}),
    pd.DataFrame({"year":"2019", "cigs":ad19["cigdyal"]})
])

print(adult_df["cigs"].describe())

plt.figure(figsize=(7,5))

sns.boxplot(
    data=adult_df,
    x="year",
    y="cigs",
    width=0.5
)
plt.xticks([0, 1], ["1999", "2019"]) 
plt.ylabel("Cigarettes/day")
plt.title("Number of cigarettes among adults smokers (1999 vs 2019)")

plt.tight_layout()
print(plt.show())


#Young people 

youth_df = pd.concat([
    pd.DataFrame({"year": "1999", "smoking": yu99["kcigreg"]}),
    pd.DataFrame({"year": "2019", "smoking": yu19["KCigReg"]})
])

labels = {
    2: "Once/twice",
    3: "Used to smoke",
    4: "Sometimes",
    5: "1-6/week",
    6: ">6/week"
}

youth_df["smoking"] = youth_df["smoking"].map(labels)

freq = (
    youth_df
    .groupby("year")["smoking"]
    .value_counts(normalize=True)
    .mul(100)
    .rename("percent")
    .reset_index()
)

sns.barplot(
    data=freq,
    x="smoking",
    y="percent",
    hue="year",
    order= [
        "Once/twice",
        "Used to smoke",
        "Sometimes",
        "1-6/week",
        ">6/week"
    ])

plt.ylabel("Percentage (%)")
plt.xlabel("")
plt.title("Smoking behaviour among young people (1999 vs 2019)")

print(plt.show())



