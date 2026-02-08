# ============================================================
# PCOS Non-Invasive Risk Screening + Lifestyle Recommendations
# + PDF Report (Streamlit App)
#
# Requirements:
#   pip install streamlit xgboost reportlab scikit-learn pandas numpy
#
# Run:
#   streamlit run pcos_app.py
#
# Note:
#   This is a screening / educational tool, NOT a diagnostic system.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
import io
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------
# 1. Model training on synthetic dataset
# ------------------------------------------------

@st.cache_resource
def train_pcos_model(csv_path: str = "synthetic_pcos_balanced_v2.csv"):
    df = pd.read_csv(csv_path)

    # Basic checks
    if "pcos" not in df.columns:
        raise ValueError("Dataset must contain a 'pcos' column as label.")

    X = df.drop("pcos", axis=1)
    y = df["pcos"].astype(int)

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
        )

    model = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Validation AUC
    proba_val = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_val)

    # Cross-validation AUC (for reporting)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    cv_auc_mean = cv_auc.mean()

    return model, list(X.columns), auc, cv_auc_mean


# ------------------------------------------------
# 2. Risk level and confidence logic
# ------------------------------------------------

def categorize_risk(prob: float) -> str:
    if prob < 0.25:
        return "Low"
    elif prob < 0.5:
        return "Moderate"
    else:
        return "High"


def confidence_level(prob: float) -> str:
    # Distance from 0.5 tells how confident the risk is
    if prob < 0.15 or prob > 0.85:
        return "HIGH CONFIDENCE"
    elif 0.35 <= prob <= 0.65:
        return "LOW CONFIDENCE (uncertain zone)"
    else:
        return "MEDIUM CONFIDENCE"


# ------------------------------------------------
# 3. Lifestyle recommendation engine
# ------------------------------------------------

def generate_recommendations(
    age,
    bmi,
    cycle_length_days,
    cycle_irregular_flag,
    symptom_score,
    hair_loss,
    pimples,
    skin_darkening,
    hair_growth,
    exercise_level,
    diet_quality,
    stress_score,
    sleep_hours,
    risk_level
):
    recs = {
        "weight_bmi": [],
        "cycle_menstrual": [],
        "symptoms": [],
        "stress_sleep": [],
        "lifestyle": [],
        "risk_specific": []
    }

    # Weight & BMI
    if bmi >= 30:
        recs["weight_bmi"].append(
            "Your BMI is in the obese range. Even a 5–10% reduction in weight can "
            "significantly improve cycle regularity and reduce PCOS-related risks."
        )
    elif bmi >= 25:
        recs["weight_bmi"].append(
            "Your BMI is in the overweight range. Gradual weight reduction through "
            "balanced diet and activity can support hormonal balance and ovulation."
        )
    elif bmi < 18.5:
        recs["weight_bmi"].append(
            "Your BMI is below the typical healthy range. Please ensure adequate nutrition "
            "and consider discussing your weight and eating patterns with a clinician."
        )
    else:
        recs["weight_bmi"].append(
            "Your BMI is within the general healthy range. Focus on maintaining it through "
            "consistent habits rather than crash dieting or extreme exercise."
        )

    # Menstrual cycle
    if cycle_irregular_flag == 1 or cycle_length_days > 35 or cycle_length_days < 24:
        recs["cycle_menstrual"].append(
            "Your cycle pattern suggests possible irregularity. Keeping a period diary or "
            "using a tracking app for a few months can help you and your doctor understand "
            "your pattern better."
        )
        recs["cycle_menstrual"].append(
            "If you frequently skip periods (gap > 35 days) or have very unpredictable cycles, "
            "it is advisable to consult a gynecologist for further evaluation."
        )
    else:
        recs["cycle_menstrual"].append(
            "Your cycle length appears to be within the expected range. Continue observing your "
            "pattern and note any major changes in frequency, flow, or pain."
        )

    # Symptoms (androgenic / metabolic)
    if symptom_score >= 2:
        recs["symptoms"].append(
            "You reported multiple symptoms (such as acne, hair loss, excess hair growth, or skin "
            "darkening). These can be signs of hormonal imbalance and may warrant a clinical check-up."
        )
    else:
        recs["symptoms"].append(
            "Symptom burden appears mild based on your inputs. If you ever notice worsening acne, "
            "progressive hair loss, or coarse hair growth, consider re-screening or consulting a doctor."
        )

    if skin_darkening == 1:
        recs["symptoms"].append(
            "Skin darkening around the neck or underarms can be related to insulin resistance. "
            "This is worth discussing with a doctor, especially if combined with weight gain."
        )

    # Stress & Sleep
    if stress_score >= 0.7:
        recs["stress_sleep"].append(
            "Your stress levels appear high. Chronic stress can worsen hormonal imbalance and "
            "sleep quality. Consider daily short practices like deep breathing, journaling, "
            "yoga, or mindful walks without screens."
        )
    else:
        recs["stress_sleep"].append(
            "Your reported stress level is in a manageable range. Continue any routines you have "
            "for relaxation and emotional wellbeing."
        )

    if sleep_hours < 7:
        recs["stress_sleep"].append(
            "You are sleeping less than the recommended 7–9 hours. Poor sleep can aggravate "
            "metabolic and hormonal issues. Try to stabilize your sleep schedule, with a fixed "
            "bedtime and wake time."
        )
    elif sleep_hours > 9:
        recs["stress_sleep"].append(
            "You reported more than 9 hours of sleep. If this is linked with fatigue or low mood, "
            "it may be worth discussing with a clinician."
        )
    else:
        recs["stress_sleep"].append(
            "Your sleep duration is within the typical recommended range. Maintaining consistent "
            "sleep timing helps support hormonal health."
        )

    # Activity & diet
    if exercise_level < 0.4:
        recs["lifestyle"].append(
            "Your activity level appears low. Aim gradually for at least 150 minutes per week "
            "of moderate exercise (brisk walking, cycling, dancing), spread across most days."
        )
    elif exercise_level < 0.7:
        recs["lifestyle"].append(
            "You have a moderate activity level. Adding 1–2 more active days per week or including "
            "light resistance training can further support metabolic health."
        )
    else:
        recs["lifestyle"].append(
            "You report a fairly active lifestyle. Keep combining cardio with light strength training "
            "if possible, and remember rest days for recovery."
        )

    if diet_quality < 0.4:
        recs["lifestyle"].append(
            "Your diet quality might be heavily tilted towards processed or high-sugar foods. "
            "Try to substitute sugary drinks with water, add more vegetables, and choose whole "
            "grains over refined carbs."
        )
    elif diet_quality < 0.7:
        recs["lifestyle"].append(
            "Your diet is moderately balanced. Small improvements like increasing fiber, reducing "
            "deep-fried items, and ensuring enough protein can have a meaningful impact."
        )
    else:
        recs["lifestyle"].append(
            "You report a relatively good diet pattern. Continue focusing on whole foods, diverse "
            "vegetables, lean proteins, and minimal ultra-processed snacks."
        )

    # Risk-level specific
    if risk_level == "High":
        recs["risk_specific"].append(
            "Your calculated PCOS risk falls in the HIGH category in this screening tool. This is "
            "NOT a diagnosis, but it strongly suggests that you should consult a gynecologist or "
            "endocrinologist for a comprehensive evaluation (hormonal tests, ultrasound, and physical exam)."
        )
    elif risk_level == "Moderate":
        recs["risk_specific"].append(
            "Your risk is MODERATE. It is advisable to monitor your cycles and symptoms over the next "
            "few months and discuss them with a healthcare provider, especially if symptoms worsen or "
            "periods become more irregular."
        )
    else:
        recs["risk_specific"].append(
            "Your current estimated risk is LOW. This does not completely rule out PCOS, but it is "
            "reassuring. Keep up with healthy habits and re-screen or consult a clinician if you notice "
            "significant menstrual or symptom changes."
        )

    return recs


# ------------------------------------------------
# 4. PDF report generator
# ------------------------------------------------

def build_pdf_report(
    name,
    age,
    height_cm,
    weight_kg,
    bmi,
    cycle_length_days,
    cycle_irregular_flag,
    symptom_score,
    exercise_level,
    diet_quality,
    stress_score,
    sleep_hours,
    risk_prob,
    risk_level,
    confidence,
    recs_dict
):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x_margin = 20 * mm
    y = height - 25 * mm

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x_margin, y, "PCOS Non-Invasive Risk Screening Report")
    y -= 10 * mm
    c.setFont("Helvetica", 10)
    c.drawString(x_margin, y, "This report is generated by an AI-based screening tool using non-invasive inputs.")
    y -= 8 * mm
    c.drawString(x_margin, y, "It is NOT a medical diagnosis. Please consult a qualified clinician for clinical decisions.")
    y -= 12 * mm

    # Patient info
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x_margin, y, "1. Client Information")
    y -= 8 * mm
    c.setFont("Helvetica", 10)
    c.drawString(x_margin, y, f"Name: {name if name else 'Not provided'}")
    y -= 6 * mm
    c.drawString(x_margin, y, f"Age: {age} years")
    y -= 6 * mm
    c.drawString(x_margin, y, f"Height: {height_cm} cm    Weight: {weight_kg} kg    BMI: {bmi:.1f}")
    y -= 6 * mm
    c.drawString(x_margin, y, f"Average Cycle Length: {cycle_length_days} days")
    y -= 6 * mm
    c.drawString(
        x_margin,
        y,
        f"Cycle Pattern: {'Irregular / Possibly Oligomenorrheic' if cycle_irregular_flag == 1 else 'Within expected range'}"
    )
    y -= 10 * mm

    # Risk summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x_margin, y, "2. PCOS Risk Summary (Screening Only)")
    y -= 8 * mm
    c.setFont("Helvetica", 10)
    c.drawString(x_margin, y, f"Estimated PCOS Risk Probability: {risk_prob:.2f}")
    y -= 6 * mm
    c.drawString(x_margin, y, f"Risk Category: {risk_level}")
    y -= 6 * mm
    c.drawString(x_margin, y, f"Confidence Level: {confidence}")
    y -= 10 * mm

    # Lifestyle and health factors
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x_margin, y, "3. Key Lifestyle & Health Indicators")
    y -= 8 * mm
    c.setFont("Helvetica", 10)
    c.drawString(x_margin, y, f"Symptom Score (0–4): {symptom_score}")
    y -= 6 * mm
    c.drawString(x_margin, y, f"Activity Level (0–1): {exercise_level:.2f}")
    y -= 6 * mm
    c.drawString(x_margin, y, f"Diet Quality (0–1): {diet_quality:.2f}")
    y -= 6 * mm
    c.drawString(x_margin, y, f"Stress Level (0–1): {stress_score:.2f}")
    y -= 6 * mm
    c.drawString(x_margin, y, f"Sleep Duration: {sleep_hours:.1f} hours/night")
    y -= 10 * mm

    # Recommendations (grouped)
    def draw_section(title, lines, y):
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x_margin, y, title)
        y -= 6 * mm
        c.setFont("Helvetica", 10)
        for line in lines:
            # If we are near bottom, new page
            if y < 25 * mm:
                c.showPage()
                y = height - 25 * mm
                c.setFont("Helvetica", 10)
            # Wrap text rudimentarily
            max_chars = 100
            while len(line) > max_chars:
                c.drawString(x_margin, y, line[:max_chars])
                line = line[max_chars:]
                y -= 5 * mm
            c.drawString(x_margin, y, line)
            y -= 5 * mm
        y -= 4 * mm
        return y

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x_margin, y, "4. Personalized Lifestyle Recommendations")
    y -= 8 * mm

    y = draw_section("4.1 Weight & BMI", recs_dict["weight_bmi"], y)
    y = draw_section("4.2 Menstrual Cycle & Period Tracking", recs_dict["cycle_menstrual"], y)
    y = draw_section("4.3 Symptoms & Hormonal Health", recs_dict["symptoms"], y)
    y = draw_section("4.4 Stress & Sleep Hygiene", recs_dict["stress_sleep"], y)
    y = draw_section("4.5 Activity & Nutrition Guidance", recs_dict["lifestyle"], y)
    y = draw_section("4.6 Overall Risk Interpretation", recs_dict["risk_specific"], y)

    # Final disclaimer
    if y < 40 * mm:
        c.showPage()
        y = height - 30 * mm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(x_margin, y, "5. Important Disclaimer")
    y -= 7 * mm
    c.setFont("Helvetica", 9)
    disclaimer_lines = [
        "This report is generated by an AI-based screening model using synthetic and non-invasive data patterns.",
        "It is intended ONLY for awareness, education, and research demonstration purposes.",
        "It does not replace clinical evaluation, diagnostic tests, or professional medical advice.",
        "Please consult a qualified healthcare provider before making any decisions about your health."
    ]
    for line in disclaimer_lines:
        if y < 25 * mm:
            c.showPage()
            y = height - 25 * mm
        c.drawString(x_margin, y, line)
        y -= 5 * mm

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# ------------------------------------------------
# 5. Streamlit app
# ------------------------------------------------

def main():
    st.set_page_config(
        page_title="PCOS Non-Invasive Risk & Lifestyle Tool",
        layout="centered"
    )

    st.title("💗 PCOS Risk Screening & Lifestyle Recommendation Tool")
    st.caption(
        "This tool provides a non-invasive **PCOS risk estimate** and **personalized lifestyle recommendations**.\n"
        "It is for **screening and educational purposes only** and does **not** provide a medical diagnosis."
    )

    # Train model
    with st.spinner("Training risk prediction model on synthetic balanced dataset..."):
        model, feature_cols, auc, cv_auc = train_pcos_model()

    st.success(f"Model ready. Test AUC ≈ **{auc:.3f}**, 5-fold CV AUC ≈ **{cv_auc:.3f}**")

    st.markdown("---")

    st.header("🧍 Client Information")
    name = st.text_input("Name (optional)", "")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (years)", min_value=15, max_value=50, value=24)
    with col2:
        height_cm = st.number_input("Height (cm)", min_value=140, max_value=190, value=160)
    with col3:
        weight_kg = st.number_input("Weight (kg)", min_value=35.0, max_value=130.0, value=60.0, step=0.5)

    bmi = weight_kg / ((height_cm / 100) ** 2)
    st.write(f"**Calculated BMI:** `{bmi:.1f}`")

    st.header("🩸 Menstrual Cycle Pattern")
    col4, col5 = st.columns(2)
    with col4:
        cycle_length_days = st.number_input("Average Cycle Length (days)", min_value=20, max_value=60, value=30)
    with col5:
        cycle_length_variation = st.number_input(
            "Typical variation in length (days)", min_value=0, max_value=15, value=3
        )

    cycle_irregular_flag = int(
        (cycle_length_days > 35) or (cycle_length_days < 24) or (cycle_length_variation > 7)
    )
    if cycle_irregular_flag == 1:
        st.warning("Your cycle inputs suggest **possible irregularity** based on length/variation.")

    st.header("✨ Visible Symptoms")
    col6, col7 = st.columns(2)
    with col6:
        hair_loss = st.checkbox("Hair thinning / hair loss")
        pimples = st.checkbox("Persistent acne")
    with col7:
        skin_darkening = st.checkbox("Skin darkening around neck/underarms")
        hair_growth = st.checkbox("Excess coarse hair (face/body)")

    hair_loss_int = int(hair_loss)
    pimples_int = int(pimples)
    skin_darkening_int = int(skin_darkening)
    hair_growth_int = int(hair_growth)
    symptom_score = hair_loss_int + pimples_int + skin_darkening_int + hair_growth_int

    st.write(f"**Symptom score (0–4):** `{symptom_score}`")

    st.header("🏃 Lifestyle & Wellbeing")
    exercise_level = st.slider("Activity level (0 = very low, 1 = very active)", 0.0, 1.0, 0.4, 0.05)
    diet_quality = st.slider("Diet quality (0 = mostly junk, 1 = very wholesome)", 0.0, 1.0, 0.5, 0.05)
    stress_score = st.slider("Stress level (0 = very low, 1 = very high)", 0.0, 1.0, 0.5, 0.05)
    sleep_hours = st.slider("Average sleep per night (hours)", 4.0, 10.0, 7.0, 0.5)

    if st.button("🔍 Estimate PCOS Risk & Get Recommendations"):
        # Build feature vector matching synthetic dataset
        bmi_symptom_interaction = bmi * symptom_score
        cycle_stress_interaction = cycle_length_days * stress_score

        input_dict = {
            "age": age,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "cycle_length_days": cycle_length_days,
            "cycle_length_variation": cycle_length_variation,
            "cycle_irregular_flag": cycle_irregular_flag,
            "hair_loss": hair_loss_int,
            "pimples": pimples_int,
            "skin_darkening": skin_darkening_int,
            "hair_growth": hair_growth_int,
            "symptom_score": symptom_score,
            "exercise_level": exercise_level,
            "diet_quality": diet_quality,
            "stress_score": stress_score,
            "sleep_hours": sleep_hours,
            "bmi_symptom_interaction": bmi_symptom_interaction,
            "cycle_stress_interaction": cycle_stress_interaction,
        }

        # Align with training columns
        input_df = pd.DataFrame([input_dict])
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0.0
        input_df = input_df[feature_cols]

        prob = model.predict_proba(input_df.values)[0, 1]
        risk_level = categorize_risk(prob)
        conf = confidence_level(prob)

        st.markdown("---")
        st.subheader("📊 PCOS Risk Screening Result")
        st.write(f"**Estimated PCOS Risk Probability:** `{prob:.2f}`")
        st.write(f"**Risk Category:** `{risk_level}`")
        st.write(f"**Confidence Level:** `{conf}`")

        st.caption(
            "This is a non-diagnostic screening result based on synthetic patterns and may not "
            "reflect your exact clinical status."
        )

        st.subheader("💡 Personalized Lifestyle Recommendations")
        recs = generate_recommendations(
            age=age,
            bmi=bmi,
            cycle_length_days=cycle_length_days,
            cycle_irregular_flag=cycle_irregular_flag,
            symptom_score=symptom_score,
            hair_loss=hair_loss_int,
            pimples=pimples_int,
            skin_darkening=skin_darkening_int,
            hair_growth=hair_growth_int,
            exercise_level=exercise_level,
            diet_quality=diet_quality,
            stress_score=stress_score,
            sleep_hours=sleep_hours,
            risk_level=risk_level
        )

        # Show recommendations grouped
        for section_name, section_title in [
            ("weight_bmi", "Weight & BMI"),
            ("cycle_menstrual", "Menstrual Cycle & Period Tracking"),
            ("symptoms", "Symptoms & Hormonal Health"),
            ("stress_sleep", "Stress & Sleep Hygiene"),
            ("lifestyle", "Activity & Nutrition Guidance"),
            ("risk_specific", "Overall Risk Interpretation"),
        ]:
            st.markdown(f"**{section_title}:**")
            for line in recs[section_name]:
                st.write(f"- {line}")
            st.markdown("")

        st.info(
            "If you are in the **moderate or high risk** category, or if your symptoms bother you, "
            "please consult a gynecologist or endocrinologist. This tool does **not** replace "
            "medical consultation."
        )

        # PDF generation
        st.markdown("---")
        st.subheader("📄 Download Structured PDF Report")
        pdf_buffer = build_pdf_report(
            name=name,
            age=age,
            height_cm=height_cm,
            weight_kg=weight_kg,
            bmi=bmi,
            cycle_length_days=cycle_length_days,
            cycle_irregular_flag=cycle_irregular_flag,
            symptom_score=symptom_score,
            exercise_level=exercise_level,
            diet_quality=diet_quality,
            stress_score=stress_score,
            sleep_hours=sleep_hours,
            risk_prob=prob,
            risk_level=risk_level,
            confidence=conf,
            recs_dict=recs
        )

        st.download_button(
            label="⬇️ Download PDF Screening Report",
            data=pdf_buffer,
            file_name="pcos_risk_screening_report.pdf",
            mime="application/pdf"
        )


if __name__ == "__main__":
    main()
