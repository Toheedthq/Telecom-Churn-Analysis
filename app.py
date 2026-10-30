import os
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📞", layout="wide")

SCALE = ["tenure", "MonthlyCharges", "TotalCharges"]
FEATS = ["tenure", "PaperlessBilling", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
         "Contract_One year", "Contract_Two year",
         "PaymentMethod_Credit card (automatic)", "PaymentMethod_Mailed check",
         "InternetService_Fiber optic", "InternetService_No",
         "MultipleLines_Yes", "StreamingTV_Yes", "StreamingMovies_Yes"]   # final model from notebook


def load(name, upload):
    return pd.read_csv(upload) if upload else pd.read_csv(name)


@st.cache_data
def train(churn, cust, net):
    """Same steps as the notebook: merge -> encode -> split -> scale -> logistic regression (GLM)."""
    d = churn.merge(cust, on="customerID").merge(net, on="customerID")
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
    d = d.dropna(subset=["TotalCharges"])
    for c in ["PaperlessBilling", "Churn"]:
        d[c] = d[c].map({"Yes": 1, "No": 0})
    d["Contract_One year"] = (d["Contract"] == "One year").astype(int)
    d["Contract_Two year"] = (d["Contract"] == "Two year").astype(int)
    d["PaymentMethod_Credit card (automatic)"] = (d["PaymentMethod"] == "Credit card (automatic)").astype(int)
    d["PaymentMethod_Mailed check"] = (d["PaymentMethod"] == "Mailed check").astype(int)
    d["InternetService_Fiber optic"] = (d["InternetService"] == "Fiber optic").astype(int)
    d["InternetService_No"] = (d["InternetService"] == "No").astype(int)
    for c in ["MultipleLines", "StreamingTV", "StreamingMovies"]:
        d[f"{c}_Yes"] = (d[c] == "Yes").astype(int)

    X, y = d[FEATS].astype(float), d["Churn"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=100)
    Xtr, Xte = Xtr.copy(), Xte.copy()
    sc = StandardScaler()
    Xtr[SCALE] = sc.fit_transform(Xtr[SCALE])
    Xte[SCALE] = sc.transform(Xte[SCALE])

    res = sm.GLM(ytr, sm.add_constant(Xtr), family=sm.families.Binomial()).fit()
    p_te = res.predict(sm.add_constant(Xte, has_constant="add"))

    # scaled coefs -> raw units so:  z = c + a1*x1 + a2*x2 ...   and   P(churn) = 1/(1+e^-z)
    b = res.params
    raw = {f: (b[f] / sc.scale_[SCALE.index(f)] if f in SCALE else b[f]) for f in FEATS}
    raw_c = b["const"] - sum(b[f] * sc.mean_[i] / sc.scale_[i] for i, f in enumerate(SCALE))
    return dict(scaled=b, raw=raw, raw_c=raw_c, p=res.pvalues, y_te=yte.values, p_te=p_te.values,
                churn_rate=y.mean(), n=len(d))


st.title("📞 Telecom Churn Predictor")
st.caption("Logistic regression from the notebook:  z = c + a·x + b·z + ...   →   P(churn) = 1 / (1 + e^−z)")

st.sidebar.header("Data files")
ups = [st.sidebar.file_uploader(n, type="csv", key=n) for n in ("churn_data.csv", "customer_data.csv", "internet_data.csv")]
try:
    churn, cust, net = (load(n, u) for n, u in zip(("churn_data.csv", "customer_data.csv", "internet_data.csv"), ups))
except FileNotFoundError:
    st.info("Put **churn_data.csv, customer_data.csv, internet_data.csv** next to app.py, or upload them in the sidebar.")
    st.stop()

m = train(churn, cust, net)
thr = st.sidebar.slider("Churn threshold", 0.05, 0.95, 0.50, 0.05,
                        help="Predict 'will churn' when probability is above this value.")

pred = (m["p_te"] >= thr).astype(int)
yt = m["y_te"]
tp, tn = int(((pred == 1) & (yt == 1)).sum()), int(((pred == 0) & (yt == 0)).sum())
fp, fn = int(((pred == 1) & (yt == 0)).sum()), int(((pred == 0) & (yt == 1)).sum())
acc, rec, pre = (tp + tn) / len(yt), tp / max(tp + fn, 1), tp / max(tp + fp, 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{m['n']:,}")
c2.metric("Churn rate", f"{m['churn_rate']:.1%}")
c3.metric("Test accuracy", f"{acc:.1%}")
c4.metric("Recall (churners caught)", f"{rec:.1%}")

tab_pred, tab_coef, tab_eval = st.tabs(["🔮 Predict churn", "📊 Coefficients", "✅ Model check"])

# ---------- Prediction ----------
with tab_pred:
    st.subheader("Describe the customer")
    l, r = st.columns(2)
    with l:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0, step=1.0)
        auto = st.checkbox("Total charges = tenure × monthly", value=True)
        total = tenure * monthly if auto else st.number_input("Total charges", 0.0, 10000.0, 840.0, step=10.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        pay = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
    with r:
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        mov = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

    x = {"tenure": tenure, "PaperlessBilling": paperless == "Yes", "MonthlyCharges": monthly,
         "TotalCharges": total, "SeniorCitizen": senior == "Yes",
         "Contract_One year": contract == "One year", "Contract_Two year": contract == "Two year",
         "PaymentMethod_Credit card (automatic)": pay == "Credit card (automatic)",
         "PaymentMethod_Mailed check": pay == "Mailed check",
         "InternetService_Fiber optic": internet == "Fiber optic", "InternetService_No": internet == "No",
         "MultipleLines_Yes": lines == "Yes", "StreamingTV_Yes": tv == "Yes", "StreamingMovies_Yes": mov == "Yes"}
    x = {k: float(v) for k, v in x.items()}

    rows = [{"Feature": f, "Value (x)": x[f], "Coefficient": m["raw"][f], "Contribution": m["raw"][f] * x[f]}
            for f in FEATS]
    rows.append({"Feature": "Intercept (c)", "Value (x)": 1.0, "Coefficient": m["raw_c"], "Contribution": m["raw_c"]})
    calc = pd.DataFrame(rows)
    z = calc["Contribution"].sum()
    prob = 1 / (1 + np.exp(-z))

    st.divider()
    a, b = st.columns(2)
    a.metric("Churn probability", f"{prob:.1%}")
    b.metric("Prediction", "⚠️ Will churn" if prob >= thr else "✅ Will stay")
    st.progress(float(prob))

    st.markdown("**Equation**")
    st.code(f"z = " + " + ".join(f"({m['raw'][f]:.4g})({x[f]:g})" for f in FEATS) + f" + ({m['raw_c']:.4g})\n"
            f"z = {z:.4f}\nP(churn) = 1 / (1 + e^(-z)) = 1 / (1 + e^({-z:.4f})) = {prob:.4f}  ->  {prob:.1%}", language="text")
    st.dataframe(calc.style.format({"Value (x)": "{:g}", "Coefficient": "{:,.5f}", "Contribution": "{:,.4f}"}),
                 width="stretch", hide_index=True)

# ---------- Coefficients ----------
with tab_coef:
    st.subheader("Model coefficients")
    names = ["Intercept (c)"] + FEATS
    coef = pd.DataFrame({
        "Feature": names,
        "Scaled coef": [m["scaled"]["const"]] + [m["scaled"][f] for f in FEATS],
        "Raw coef (real units)": [m["raw_c"]] + [m["raw"][f] for f in FEATS],
        "Odds ratio": [np.exp(m["scaled"]["const"])] + [np.exp(m["raw"][f]) for f in FEATS],
        "p-value": [m["p"]["const"]] + [m["p"][f] for f in FEATS],
    })
    st.dataframe(coef.style.format({"Scaled coef": "{:.4f}", "Raw coef (real units)": "{:.5f}",
                                    "Odds ratio": "{:.3f}", "p-value": "{:.4f}"}),
                 width="stretch", hide_index=True)
    st.bar_chart(coef[coef["Feature"] != "Intercept (c)"].set_index("Feature")["Scaled coef"])
    st.caption("Positive scaled coef → raises churn risk. Negative → lowers it. "
               "Odds ratio > 1 means higher churn odds per +1 unit (1 month, 1 currency unit, or Yes vs No).")

# ---------- Evaluation ----------
with tab_eval:
    st.subheader(f"Confusion matrix on test data (threshold {thr:.2f})")
    st.dataframe(pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual: stay", "Actual: churn"],
                              columns=["Predicted: stay", "Predicted: churn"]), width="stretch")
    st.dataframe(pd.DataFrame({"Metric": ["Accuracy", "Precision", "Recall"], "Value": [acc, pre, rec]})
                 .style.format({"Value": "{:.1%}"}), hide_index=True)
    st.caption("Lower the threshold in the sidebar to catch more churners (higher recall) at the cost of more false alarms.")
