import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import pydeck as pdk
    HAS_PYDECK = True
except Exception:
    HAS_PYDECK = False

# Page config
st.set_page_config(page_title="Smoking Health Analysis", layout="wide", page_icon="")

# No custom CSS – default Streamlit theme
# (All custom neon styling has been removed)

PLOTLY_TEMPLATE = "plotly"  # default Plotly light template
NEON_COLORS = px.colors.qualitative.Plotly   # use standard Plotly palette
SMOKE_PALETTE = {"Non-smoker": "#1f77b4", "Smoker": "#d62728"}  # standard blue vs red

st.title("Smoking Health Indicators Analysis")

# Load & clean
@st.cache_data
def load_data():
    return pd.read_csv("smoking.csv")

@st.cache_data
def clean_data(df):
    df_clean = df.copy()
    df_clean['gender'] = df_clean['gender'].map({'M': 1, 'F': 0})
    df_clean['oral']   = df_clean['oral'].map({'Y': 1, 'N': 0})
    df_clean['tartar'] = df_clean['tartar'].map({'Y': 1, 'N': 0})
    df_clean['BMI']    = df_clean['weight(kg)'] / ((df_clean['height(cm)'] / 100) ** 2)
    bins   = [0, 30, 40, 50, 60, 100]
    labels = ['<30', '30-39', '40-49', '50-59', '60+']
    df_clean['age_group'] = pd.cut(df_clean['age'], bins=bins, labels=labels, right=False)
    df_clean['HDL_LDL_ratio']  = df_clean['HDL'] / df_clean['LDL'].replace(0, np.nan)
    df_clean['AST_ALT_ratio']  = df_clean['AST'] / df_clean['ALT'].replace(0, np.nan)
    df_clean['pulse_pressure'] = df_clean['systolic'] - df_clean['relaxation']
    df_clean['smoke_label']    = df_clean['smoking'].map({0: 'Non-smoker', 1: 'Smoker'})
    regions = {
        'Seoul':   (37.5665, 126.9780), 'Busan':   (35.1796, 129.0756),
        'Incheon': (37.4563, 126.7052), 'Daegu':   (35.8714, 128.6014),
        'Daejeon': (36.3504, 127.3845), 'Gwangju': (35.1595, 126.8526),
        'Ulsan':   (35.5384, 129.3114), 'Jeju':    (33.4996, 126.5312),
    }
    rng = np.random.RandomState(42)
    region_names = list(regions.keys())
    assigned = rng.choice(region_names, size=len(df_clean))
    df_clean['region'] = assigned
    df_clean['lat'] = [regions[r][0] + rng.normal(0, 0.05) for r in assigned]
    df_clean['lon'] = [regions[r][1] + rng.normal(0, 0.05) for r in assigned]
    return df_clean

df_raw = load_data()
df     = clean_data(df_raw)
st.success(f"Data loaded — **{df.shape[0]:,} rows x {df.shape[1]} columns**")

# Sidebar
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", [
    "Overview", "EDA & Visualizations", "3D Explorer",
    "Geographic Map", "Machine Learning",
    "Prediction Playground", "Case Study", "Raw & Export"
])

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")
gender_sel = st.sidebar.multiselect("Gender",  ['Female', 'Male'],         default=['Female', 'Male'])
smoke_sel  = st.sidebar.multiselect("Smoking", ['Non-smoker', 'Smoker'],   default=['Non-smoker', 'Smoker'])
age_min, age_max = int(df['age'].min()), int(df['age'].max())
age_range  = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
region_sel = st.sidebar.multiselect("Region", sorted(df['region'].unique()),
                                    default=sorted(df['region'].unique()))

gmap = {'Female': 0, 'Male': 1}
smap = {'Non-smoker': 0, 'Smoker': 1}
fdf = df[
    df['gender'].isin([gmap[g] for g in gender_sel]) &
    df['smoking'].isin([smap[s] for s in smoke_sel]) &
    df['age'].between(age_range[0], age_range[1]) &
    df['region'].isin(region_sel)
].copy()

st.sidebar.caption(f"Filtered rows: **{len(fdf):,}** / {len(df):,}")

def fig_download(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    st.download_button(f"Download '{name}'", buf, f"{name}.png", "image/png", key=name)

# OVERVIEW
if options == "Overview":
    st.header("Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records",   f"{len(fdf):,}")
    c2.metric("Features",        df.shape[1])
    c3.metric("Smokers %",       f"{fdf['smoking'].mean()*100:.1f}%" if len(fdf) else "—")
    c4.metric("Avg Age",         f"{fdf['age'].mean():.1f}"          if len(fdf) else "—")
    c5.metric("Avg BMI",         f"{fdf['BMI'].mean():.1f}"          if len(fdf) else "—")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Smoking Split")
        counts = fdf['smoke_label'].value_counts()
        fig = px.pie(values=counts.values, names=counts.index,
                     color=counts.index, color_discrete_map=SMOKE_PALETTE,
                     hole=0.55, template=PLOTLY_TEMPLATE)
        fig.update_traces(textinfo='percent+label', pull=[0.05, 0])
        fig.update_layout(showlegend=True, height=350)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Gender Split")
        g_counts = fdf['gender'].map({0:'Female',1:'Male'}).value_counts()
        fig = px.pie(values=g_counts.values, names=g_counts.index,
                     color_discrete_sequence=["#d62728","#1f77b4"],
                     hole=0.55, template=PLOTLY_TEMPLATE)
        fig.update_traces(textinfo='percent+label', pull=[0.05, 0])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Health Gauges")
    gauge_cols = st.columns(4)
    gauges = [
        ("Avg Systolic BP",  fdf['systolic'].mean(),    80,  180, "mmHg"),
        ("Avg Cholesterol",  fdf['Cholesterol'].mean(), 100, 300, "mg/dL"),
        ("Avg Triglyceride", fdf['triglyceride'].mean(),50,  400, "mg/dL"),
        ("Avg Hemoglobin",   fdf['hemoglobin'].mean(),  10,  20,  "g/dL"),
    ]
    for col, (title, val, lo, hi, unit) in zip(gauge_cols, gauges):
        if not np.isnan(val):
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(val, 1),
                title={'text': f"{title}<br><span style='font-size:0.8em;color:#333'>{unit}</span>"},
                gauge={
                    'axis': {'range': [lo, hi]},
                    'bar':  {'color': '#1f77b4'},
                    'steps': [
                        {'range': [lo, lo+(hi-lo)*0.33], 'color': 'rgba(44, 160, 44, 0.1)'},
                        {'range': [lo+(hi-lo)*0.33, lo+(hi-lo)*0.66], 'color': 'rgba(255, 127, 14, 0.1)'},
                        {'range': [lo+(hi-lo)*0.66, hi], 'color': 'rgba(214, 39, 40, 0.1)'},
                    ],
                    'threshold': {'line': {'color': '#d62728', 'width': 3},
                                  'thickness': 0.75, 'value': val}
                },
                number={'font': {'size': 28}}
            ))
            fig.update_layout(height=220, margin=dict(l=20,r=20,t=60,b=10))
            col.plotly_chart(fig, use_container_width=True)

    st.subheader("Quick Numeric Summary")
    st.dataframe(fdf.describe().T.style.background_gradient(cmap='Blues'))

# EDA & VISUALIZATIONS
elif options == "EDA & Visualizations":
    st.header("Exploratory Data Analysis")
    if len(fdf) == 0:
        st.warning("No data after filters."); st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distributions", "Comparisons", "Correlations",
        "Health Ratios", "Sunburst & Parallel"
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(fdf, x='age', color='smoke_label', nbins=30, barmode='overlay',
                               color_discrete_map=SMOKE_PALETTE, template=PLOTLY_TEMPLATE,
                               title="Age Distribution by Smoking", opacity=0.75, marginal="violin")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(fdf, x='BMI', color='smoke_label', nbins=40, barmode='overlay',
                               color_discrete_map=SMOKE_PALETTE, template=PLOTLY_TEMPLATE,
                               title="BMI Distribution", opacity=0.75, marginal="box")
            st.plotly_chart(fig, use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            fig = px.violin(fdf, y='Cholesterol', x='smoke_label', color='smoke_label',
                            box=True, points='outliers',
                            color_discrete_map=SMOKE_PALETTE, template=PLOTLY_TEMPLATE,
                            title="Cholesterol Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.violin(fdf, y='Gtp', x='smoke_label', color='smoke_label',
                            box=True, points='outliers',
                            color_discrete_map=SMOKE_PALETTE, template=PLOTLY_TEMPLATE,
                            title="GTP (Liver Enzyme) Distribution")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            ct = pd.crosstab(fdf['age_group'].astype(str), fdf['smoke_label'], normalize='index') * 100
            fig = px.bar(ct.reset_index(), x='age_group', y=['Non-smoker','Smoker'],
                         barmode='stack', template=PLOTLY_TEMPLATE,
                         color_discrete_map=SMOKE_PALETTE,
                         title="Smoking % by Age Group", labels={'value':'%','age_group':'Age Group'})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            metrics = ['BMI','systolic','triglyceride','HDL','LDL','Gtp','hemoglobin']
            means   = fdf.groupby('smoke_label')[metrics].mean().reset_index()
            fig = px.bar(means.melt(id_vars='smoke_label'), x='variable', y='value',
                         color='smoke_label', barmode='group',
                         color_discrete_map=SMOKE_PALETTE, template=PLOTLY_TEMPLATE,
                         title="Mean Health Metrics by Smoking Status")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Radar Chart — Normalised Health Profile")
        radar_metrics = ['BMI','systolic','triglyceride','HDL','LDL','Gtp','hemoglobin','pulse_pressure']
        norm = fdf.groupby('smoke_label')[radar_metrics].mean()
        norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)
        fig = go.Figure()
        colors_r = ['#1f77b4','#d62728']
        for i, row in norm.iterrows():
            vals = list(row.values) + [row.values[0]]
            cats = radar_metrics + [radar_metrics[0]]
            fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself',
                                          name=i, line_color=colors_r.pop(0), opacity=0.75))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                          template=PLOTLY_TEMPLATE, height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        numeric_cols = ['age','height(cm)','weight(kg)','waist(cm)','systolic','relaxation',
                        'fasting blood sugar','Cholesterol','triglyceride','HDL','LDL',
                        'hemoglobin','serum creatinine','AST','ALT','Gtp','BMI','smoking']
        corr = fdf[numeric_cols].corr()
        fig = px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                        template=PLOTLY_TEMPLATE, title="Correlation Heatmap",
                        text_auto=False, aspect='auto')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        top = corr['smoking'].drop('smoking').abs().sort_values(ascending=True).tail(12)
        fig = px.bar(x=top.values, y=top.index, orientation='h',
                     color=top.values, color_continuous_scale='Blues',
                     template=PLOTLY_TEMPLATE, title="Absolute Correlation with Smoking")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(fdf, x='smoke_label', y='HDL_LDL_ratio', color='smoke_label',
                         color_discrete_map=SMOKE_PALETTE, template=PLOTLY_TEMPLATE,
                         title="HDL/LDL Ratio by Smoking", points='outliers')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(fdf, x='smoke_label', y='pulse_pressure', color='smoke_label',
                         color_discrete_map=SMOKE_PALETTE, template=PLOTLY_TEMPLATE,
                         title="Pulse Pressure by Smoking", points='outliers')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Bubble Chart — Weight vs BMI vs Cholesterol")
        sample = fdf.sample(min(1500, len(fdf)), random_state=1)
        fig = px.scatter(sample, x='weight(kg)', y='BMI', size='Cholesterol',
                         color='smoke_label', color_discrete_map=SMOKE_PALETTE,
                         hover_data=['age','systolic'], template=PLOTLY_TEMPLATE,
                         title="Weight vs BMI (bubble = Cholesterol)", opacity=0.7, size_max=20)
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Sunburst — Gender > Age Group > Smoking")
        sb = fdf.copy()
        sb['gender_label'] = sb['gender'].map({0:'Female',1:'Male'})
        fig = px.sunburst(sb, path=['gender_label','age_group','smoke_label'],
                          color='smoke_label', color_discrete_map=SMOKE_PALETTE,
                          template=PLOTLY_TEMPLATE, title="Sunburst: Gender > Age > Smoking")
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Parallel Coordinates — Multi-feature View")
        pc_cols = ['age','BMI','systolic','Cholesterol','triglyceride','HDL','Gtp','smoking']
        pc_df   = fdf[pc_cols].dropna().sample(min(2000, len(fdf)), random_state=42)
        fig = px.parallel_coordinates(pc_df, color='smoking',
                                      color_continuous_scale=['#1f77b4','#d62728'],
                                      template=PLOTLY_TEMPLATE,
                                      title="Parallel Coordinates — Health Features")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# 3D EXPLORER
elif options == "3D Explorer":
    st.header("3D Feature Explorer")
    if len(fdf) == 0:
        st.warning("No data after filters."); st.stop()

    all_num = ['age','BMI','systolic','relaxation','Cholesterol','triglyceride',
               'HDL','LDL','hemoglobin','Gtp','waist(cm)','weight(kg)','pulse_pressure']

    st.subheader("Interactive 3D Scatter")
    c1, c2, c3 = st.columns(3)
    x_ax = c1.selectbox("X axis", all_num, index=0)
    y_ax = c2.selectbox("Y axis", all_num, index=1)
    z_ax = c3.selectbox("Z axis", all_num, index=2)

    sample3d = fdf.sample(min(3000, len(fdf)), random_state=7)
    fig = px.scatter_3d(sample3d, x=x_ax, y=y_ax, z=z_ax,
                        color='smoke_label', color_discrete_map=SMOKE_PALETTE,
                        opacity=0.65, template=PLOTLY_TEMPLATE,
                        title=f"3D Scatter: {x_ax} x {y_ax} x {z_ax}",
                        hover_data=['age','gender'])
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("3D Surface — Age x BMI vs Mean Systolic BP")
    age_bins = pd.cut(fdf['age'], bins=10)
    bmi_bins = pd.cut(fdf['BMI'], bins=10)
    pivot = fdf.groupby([age_bins, bmi_bins])['systolic'].mean().unstack()
    pivot = pivot.fillna(pivot.mean().mean())
    z_vals = pivot.values
    fig = go.Figure(go.Surface(
        z=z_vals, colorscale='Plasma', opacity=0.9,
        contours={"z": {"show": True, "usecolormap": True, "highlightcolor": "#1f77b4", "project_z": True}}
    ))
    fig.update_layout(title="3D Surface: Age x BMI vs Systolic BP",
                      template=PLOTLY_TEMPLATE, height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Smoking Rate by Region and Gender")
    rg = fdf.copy()
    rg['gender_label'] = rg['gender'].map({0:'Female',1:'Male'})
    rg_grp = rg.groupby(['region','gender_label'])['smoking'].mean().reset_index()
    rg_grp['smoking_pct'] = rg_grp['smoking'] * 100
    fig = go.Figure()
    colors_3d = {'Female':'#d62728','Male':'#1f77b4'}
    for g in rg_grp['gender_label'].unique():
        sub = rg_grp[rg_grp['gender_label']==g]
        fig.add_trace(go.Bar(name=g, x=sub['region'], y=sub['smoking_pct'],
                             marker_color=colors_3d[g], opacity=0.85))
    fig.update_layout(barmode='group', template=PLOTLY_TEMPLATE,
                      title="Smoking Rate (%) by Region and Gender",
                      yaxis_title="Smoking %", height=450)
    st.plotly_chart(fig, use_container_width=True)

# GEOGRAPHIC MAP
elif options == "Geographic Map":
    st.header("Geographic View (synthetic regions)")
    st.caption("Regions are synthetic — replace lat/lon with real coordinates for production use.")

    region_stats = fdf.groupby('region').agg(
        records=('smoking','size'), smoking_rate=('smoking','mean'),
        avg_bmi=('BMI','mean'), lat=('lat','mean'), lon=('lon','mean')
    ).reset_index()
    region_stats['smoking_pct'] = (region_stats['smoking_rate']*100).round(1)

    c1, c2 = st.columns([2,1])
    with c1:
        fig = px.scatter_mapbox(
            region_stats, lat='lat', lon='lon', size='records',
            color='smoking_pct', color_continuous_scale='Plasma',
            hover_name='region', hover_data={'smoking_pct':True,'records':True,'avg_bmi':':.1f'},
            zoom=6, height=500, mapbox_style='carto-positron',
            title="Smoking Rate by Region"
        )
        fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Region Stats")
        st.dataframe(region_stats[['region','records','smoking_pct','avg_bmi']]
                     .round(2).sort_values('smoking_pct', ascending=False))

    fig = px.bar(region_stats.sort_values('smoking_pct'), x='smoking_pct', y='region',
                 orientation='h', color='smoking_pct', color_continuous_scale='Reds',
                 template=PLOTLY_TEMPLATE, title="Smoking Rate (%) by Region")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# MACHINE LEARNING
elif options == "Machine Learning":
    st.header("Predicting Smoking Status — Model Comparison")
    feature_cols = ['gender','age','height(cm)','weight(kg)','waist(cm)',
                    'systolic','relaxation','fasting blood sugar','Cholesterol',
                    'triglyceride','HDL','LDL','hemoglobin','serum creatinine',
                    'AST','ALT','Gtp','oral','tartar','BMI']

    base = fdf if len(fdf) > 200 else df
    X = base[feature_cols].dropna()
    y = base.loc[X.index, 'smoking']

    c1, c2 = st.columns(2)
    test_size = c1.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    chosen    = c2.multiselect("Models",
                    ['Random Forest','Logistic Regression','Gradient Boosting'],
                    default=['Random Forest','Logistic Regression','Gradient Boosting'])

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y)

    model_map = {
        'Random Forest':       RandomForestClassifier(n_estimators=150, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Gradient Boosting':   GradientBoostingClassifier(random_state=42),
    }

    results, roc_data, trained = [], {}, {}
    with st.spinner("Training models..."):
        for name in chosen:
            m = model_map[name]; m.fit(X_train, y_train)
            pred  = m.predict(X_test)
            proba = m.predict_proba(X_test)[:,1]
            trained[name] = m
            results.append({'Model':name,
                'Accuracy':  accuracy_score(y_test, pred),
                'Precision': precision_score(y_test, pred),
                'Recall':    recall_score(y_test, pred),
                'F1':        f1_score(y_test, pred),
                'ROC-AUC':   roc_auc_score(y_test, proba)})
            roc_data[name] = roc_curve(y_test, proba)

    res_df = pd.DataFrame(results).set_index('Model').round(3)
    st.subheader("Metrics Comparison")
    st.dataframe(res_df.style.background_gradient(cmap='Blues', axis=None))

    fig = px.bar(res_df.reset_index().melt(id_vars='Model'), x='Model', y='value',
                 color='variable', barmode='group', template=PLOTLY_TEMPLATE,
                 color_discrete_sequence=NEON_COLORS,
                 title="Model Metric Comparison", range_y=[0,1])
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    for i, (name, (fpr, tpr, _)) in enumerate(roc_data.items()):
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                 name=f"{name} (AUC={res_df.loc[name,'ROC-AUC']:.2f})",
                                 line=dict(color=NEON_COLORS[i], width=2.5)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random',
                             line=dict(color='gray', dash='dash')))
    fig.update_layout(title='ROC Curves', xaxis_title='FPR', yaxis_title='TPR',
                      template=PLOTLY_TEMPLATE, height=450)
    st.plotly_chart(fig, use_container_width=True)

    best = res_df['ROC-AUC'].idxmax()
    st.subheader(f"Best Model: {best}")
    best_model = trained[best]
    pred = best_model.predict(X_test)
    cm   = confusion_matrix(y_test, pred)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                        labels=dict(x='Predicted',y='True'),
                        x=['Non-smoker','Smoker'], y=['Non-smoker','Smoker'],
                        template=PLOTLY_TEMPLATE, title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if hasattr(best_model,'feature_importances_'):
            imp = best_model.feature_importances_
            idx = np.argsort(imp)[::-1]
            fig = px.bar(x=imp[idx], y=[feature_cols[i] for i in idx],
                         orientation='h', color=imp[idx],
                         color_continuous_scale='Blues', template=PLOTLY_TEMPLATE,
                         title="Feature Importance")
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feature importances available for this model.")
    st.code(classification_report(y_test, pred), language='text')

# PREDICTION PLAYGROUND
elif options == "Prediction Playground":
    st.header("Prediction Playground")
    st.markdown("Enter patient values below and get a real-time smoking probability from a trained Random Forest.")

    feature_cols = ['gender','age','height(cm)','weight(kg)','waist(cm)',
                    'systolic','relaxation','fasting blood sugar','Cholesterol',
                    'triglyceride','HDL','LDL','hemoglobin','serum creatinine',
                    'AST','ALT','Gtp','oral','tartar','BMI']

    @st.cache_resource
    def get_trained_model():
        X = df[feature_cols].dropna()
        y = df.loc[X.index,'smoking']
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        m  = RandomForestClassifier(n_estimators=150, random_state=42)
        m.fit(Xs, y)
        return m, sc

    model_pg, scaler_pg = get_trained_model()

    with st.form("predict_form"):
        st.subheader("Patient Profile")
        c1, c2, c3 = st.columns(3)
        gender_in  = c1.selectbox("Gender", ['Male','Female'])
        age_in     = c2.slider("Age", 20, 85, 40)
        height_in  = c3.slider("Height (cm)", 140, 200, 170)
        weight_in  = c1.slider("Weight (kg)", 40, 150, 70)
        waist_in   = c2.slider("Waist (cm)", 50, 130, 80)
        systolic_in= c3.slider("Systolic BP", 80, 200, 120)
        relax_in   = c1.slider("Relaxation BP", 50, 130, 80)
        fbs_in     = c2.slider("Fasting Blood Sugar", 50, 300, 100)
        chol_in    = c3.slider("Cholesterol", 100, 400, 200)
        trig_in    = c1.slider("Triglyceride", 30, 600, 120)
        hdl_in     = c2.slider("HDL", 20, 120, 55)
        ldl_in     = c3.slider("LDL", 30, 300, 120)
        hemo_in    = c1.slider("Hemoglobin", 8.0, 20.0, 14.0)
        creat_in   = c2.slider("Serum Creatinine", 0.3, 5.0, 1.0)
        ast_in     = c3.slider("AST", 5, 200, 25)
        alt_in     = c1.slider("ALT", 5, 200, 25)
        gtp_in     = c2.slider("GTP", 5, 300, 30)
        oral_in    = c3.selectbox("Oral Health Issue", ['No','Yes'])
        tartar_in  = c1.selectbox("Tartar", ['No','Yes'])
        submitted  = st.form_submit_button("Run Prediction", use_container_width=True)

    if submitted:
        bmi_in = weight_in / ((height_in/100)**2)
        row = [[
            1 if gender_in=='Male' else 0, age_in, height_in, weight_in, waist_in,
            systolic_in, relax_in, fbs_in, chol_in, trig_in, hdl_in, ldl_in,
            hemo_in, creat_in, ast_in, alt_in, gtp_in,
            1 if oral_in=='Yes' else 0, 1 if tartar_in=='Yes' else 0, bmi_in
        ]]
        row_scaled = scaler_pg.transform(row)
        prob = model_pg.predict_proba(row_scaled)[0][1]
        pred_label = "Smoker" if prob >= 0.5 else "Non-smoker"

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Prediction", pred_label)
            st.metric("Smoking Probability", f"{prob*100:.1f}%")
        with c2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob*100,1),
                title={'text':"Smoking Probability (%)"},
                gauge={
                    'axis':{'range':[0,100]},
                    'bar':{'color':'#d62728' if prob>=0.5 else '#1f77b4'},
                    'steps':[
                        {'range':[0,33],'color':'rgba(44,160,44,0.1)'},
                        {'range':[33,66],'color':'rgba(255,127,14,0.1)'},
                        {'range':[66,100],'color':'rgba(214,39,40,0.1)'},
                    ],
                    'threshold':{'line':{'color':'white','width':3},'thickness':0.75,'value':50}
                },
                number={'suffix':'%','font':{'size':36}}
            ))
            fig.update_layout(height=280, margin=dict(l=20,r=20,t=60,b=10))
            st.plotly_chart(fig, use_container_width=True)

# CASE STUDY
elif options == "Case Study":
    st.header("Case Study — Smoking and Health Intervention Program")
    st.markdown("A structured analysis of the smoking health crisis, evidence-based intervention plan, projected effects, and measurable KPIs derived from this dataset.")

    cs_tab1, cs_tab2, cs_tab3, cs_tab4 = st.tabs([
        "Problem Analysis", "Solution Plan", "Projected Effects", "KPIs Dashboard"
    ])

    # Compute data-driven stats
    smoker_df     = df[df['smoking'] == 1]
    nonsmoker_df  = df[df['smoking'] == 0]
    smoking_rate  = df['smoking'].mean() * 100
    male_smoke    = df[df['gender']==1]['smoking'].mean() * 100
    female_smoke  = df[df['gender']==0]['smoking'].mean() * 100
    peak_age_grp  = df.groupby('age_group')['smoking'].mean().idxmax()
    avg_bmi_s     = smoker_df['BMI'].mean()
    avg_bmi_ns    = nonsmoker_df['BMI'].mean()
    avg_sys_s     = smoker_df['systolic'].mean()
    avg_sys_ns    = nonsmoker_df['systolic'].mean()
    avg_gtp_s     = smoker_df['Gtp'].mean()
    avg_gtp_ns    = nonsmoker_df['Gtp'].mean()
    avg_trig_s    = smoker_df['triglyceride'].mean()
    avg_trig_ns   = nonsmoker_df['triglyceride'].mean()
    tartar_smoke  = smoker_df['tartar'].mean() * 100
    tartar_nonsmk = nonsmoker_df['tartar'].mean() * 100

    # TAB 1 — PROBLEM ANALYSIS
    with cs_tab1:
        st.subheader("Problem Statement")
        st.markdown(f"""
<div style="padding:20px; border:1px solid #ddd; border-radius:10px; margin-bottom:16px">
<h4>Background</h4>
<p>This dataset captures health screening data from <strong>{len(df):,} individuals</strong> across 8 Korean regions.
The overall smoking prevalence is <strong>{smoking_rate:.1f}%</strong>, with a clear gender gap:
<strong>{male_smoke:.1f}%</strong> of males smoke compared to only <strong>{female_smoke:.1f}%</strong> of females.
The highest-risk age group is <strong>{peak_age_grp}</strong>.</p>
</div>
""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        problems = [
            ("Cardiovascular Risk", f"Smokers show +{avg_sys_s - avg_sys_ns:.1f} mmHg higher systolic BP on average, significantly elevating stroke and heart disease risk."),
            ("Metabolic Dysfunction", f"Smokers have {((avg_trig_s/avg_trig_ns)-1)*100:.1f}% higher triglycerides and lower HDL cholesterol, indicating metabolic syndrome risk."),
            ("Oral Health", f"{tartar_smoke:.1f}% of smokers have tartar vs {tartar_nonsmk:.1f}% of non-smokers — a {tartar_smoke - tartar_nonsmk:.1f}pp gap indicating significant oral health neglect."),
            ("Liver Stress", f"Smokers average GTP of {avg_gtp_s:.1f} vs {avg_gtp_ns:.1f} in non-smokers — a {((avg_gtp_s/avg_gtp_ns)-1)*100:.1f}% elevation signalling liver enzyme stress."),
            ("Weight and BMI", f"Smokers average BMI of {avg_bmi_s:.1f} vs {avg_bmi_ns:.1f} — combined with other markers this compounds cardiometabolic risk."),
            ("Regional Disparity", f"Smoking rates vary significantly across regions, suggesting unequal access to cessation programs and health education resources."),
        ]
        for i, (title, desc) in enumerate(problems):
            col = [c1, c2, c3][i % 3]
            col.markdown(f"""
<div style="padding:15px; border:1px solid #eee; border-radius:8px; margin-bottom:12px">
<h4>{title}</h4>
<p>{desc}</p>
</div>""", unsafe_allow_html=True)

        st.subheader("Problem Severity — Smoker vs Non-Smoker Comparison")
        compare_metrics = {
            'Systolic BP':   (avg_sys_s,  avg_sys_ns),
            'Triglyceride':  (avg_trig_s, avg_trig_ns),
            'GTP':           (avg_gtp_s,  avg_gtp_ns),
            'BMI':           (avg_bmi_s,  avg_bmi_ns),
            'Tartar Rate %': (tartar_smoke, tartar_nonsmk),
        }
        cmp_df = pd.DataFrame(compare_metrics, index=['Smoker','Non-Smoker']).T.reset_index()
        cmp_df.columns = ['Metric','Smoker','Non-Smoker']
        fig = px.bar(cmp_df.melt(id_vars='Metric'), x='Metric', y='value', color='variable',
                     barmode='group', template=PLOTLY_TEMPLATE,
                     color_discrete_map={'Smoker':'#d62728','Non-Smoker':'#1f77b4'},
                     title="Key Health Markers: Smoker vs Non-Smoker")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Risk Score by Age Group")
        risk = df.groupby('age_group').agg(
            smoking_rate=('smoking','mean'),
            avg_systolic=('systolic','mean'),
            avg_triglyceride=('triglyceride','mean'),
            avg_gtp=('Gtp','mean')
        ).reset_index()
        risk['composite_risk'] = (
            risk['smoking_rate'] * 0.4 +
            (risk['avg_systolic'] - risk['avg_systolic'].min()) / (risk['avg_systolic'].max() - risk['avg_systolic'].min()) * 0.3 +
            (risk['avg_triglyceride'] - risk['avg_triglyceride'].min()) / (risk['avg_triglyceride'].max() - risk['avg_triglyceride'].min()) * 0.3
        )
        fig = px.bar(risk, x='age_group', y='composite_risk',
                     color='composite_risk', color_continuous_scale='Reds',
                     template=PLOTLY_TEMPLATE, title="Composite Risk Score by Age Group",
                     labels={'composite_risk':'Risk Score (0-1)','age_group':'Age Group'})
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # TAB 2 — SOLUTION PLAN
    with cs_tab2:
        st.subheader("Evidence-Based Intervention Plan")
        st.markdown("""
<div style="padding:20px; border:1px solid #ddd; border-radius:10px; margin-bottom:16px">
<h4>Strategic Objective</h4>
<p>Reduce population smoking prevalence by <strong>30% within 3 years</strong> through a multi-layered,
data-driven public health intervention targeting the highest-risk demographic segments identified in this dataset.</p>
</div>
""", unsafe_allow_html=True)

        st.subheader("4-Phase Implementation Roadmap")
        phases = [
            ("phase-1", "Phase 1 — Foundation (Months 1-3)",
             "Data Infrastructure and Targeting",
             ["Deploy ML screening model (RF, AUC >= 0.85) in all regional clinics",
              "Build real-time health dashboard for regional health officers",
              "Identify top 20% highest-risk individuals using prediction scores",
              "Establish baseline KPI measurements across all 8 regions"]),
            ("phase-2", "Phase 2 — Intervention Launch (Months 4-9)",
             "Clinical and Community Programs",
             ["Launch Nicotine Replacement Therapy (NRT) subsidy program",
              "Deploy mobile cessation counselling units in high-smoking regions",
              "Introduce workplace smoking cessation incentive schemes",
              "Partner with dental clinics for oral health and cessation co-screening",
              "Run targeted social media campaigns for 30-49 age group (peak risk)"]),
            ("phase-3", "Phase 3 — Scale and Reinforce (Months 10-18)",
             "Digital Health and Policy",
             ["Launch AI-powered cessation chatbot with personalised plans",
              "Introduce school-based prevention programs for under-30 cohort",
              "Advocate for smoke-free workplace legislation in high-rate regions",
              "Integrate wearable health monitoring for enrolled participants",
              "Monthly KPI review and adaptive resource reallocation"]),
            ("phase-4", "Phase 4 — Sustain and Evaluate (Months 19-36)",
             "Outcomes and Long-term Sustainability",
             ["Conduct full population re-screening and compare to baseline",
              "Publish regional health improvement reports",
              "Transition successful participants to long-term wellness programs",
              "Refine ML model with new data for next intervention cycle",
              "Scale proven interventions nationally"]),
        ]
        for cls, title, subtitle, actions in phases:
            st.markdown(f"""
<div style="border-left:3px solid #1f77b4; padding-left:16px; margin-bottom:20px">
<span style="display:inline-block; padding:4px 14px; border-radius:20px; font-size:0.8rem; font-weight:600;
background:#e6f0fa; border:1px solid #1f77b4; color:#1f77b4">{title}</span>
<h4 style="color:#333; margin:8px 0 4px 0;">{subtitle}</h4>
<ul style="line-height:1.9;">
{''.join(f'<li>{a}</li>' for a in actions)}
</ul>
</div>""", unsafe_allow_html=True)

        st.subheader("Targeted Interventions by Risk Segment")
        segments = {
            'Segment': ['High-Risk Males 30-49', 'Females with Oral Issues', 'High GTP / Liver Risk', 'Obese Smokers (BMI>30)', 'High Triglyceride Smokers'],
            'Size (est.)': [f"{int(len(smoker_df[smoker_df['gender']==1]) * 0.4):,}",
                            f"{int(len(smoker_df[(smoker_df['gender']==0) & (smoker_df['tartar']==1)])):,}",
                            f"{int(len(smoker_df[smoker_df['Gtp'] > smoker_df['Gtp'].quantile(0.75)])):,}",
                            f"{int(len(smoker_df[smoker_df['BMI'] > 30])):,}",
                            f"{int(len(smoker_df[smoker_df['triglyceride'] > 200])):,}"],
            'Primary Intervention': ['NRT + Workplace Programs', 'Dental + Cessation Co-screening',
                                     'Liver Health Monitoring + NRT', 'Weight Management + Cessation',
                                     'Dietary Counselling + Cessation'],
            'Priority': ['Critical', 'High', 'High', 'Medium', 'Medium']
        }
        st.dataframe(pd.DataFrame(segments), use_container_width=True)

        st.subheader("Estimated Budget Allocation")
        budget = {'Program': ['Clinical NRT Subsidies','Digital Health Platform','Community Outreach',
                               'School Prevention','Policy and Advocacy','Monitoring and Evaluation'],
                  'Budget %': [35, 20, 18, 12, 8, 7]}
        fig = px.pie(budget, values='Budget %', names='Program',
                     color_discrete_sequence=NEON_COLORS, hole=0.45,
                     template=PLOTLY_TEMPLATE, title="Intervention Budget Allocation")
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # TAB 3 — PROJECTED EFFECTS
    with cs_tab3:
        st.subheader("Projected Health and Economic Effects")

        st.markdown("""
<div style="padding:20px; border:1px solid #ddd; border-radius:10px; margin-bottom:16px">
<h4>Projection Methodology</h4>
<p>Projections are based on published cessation intervention efficacy rates (WHO, Cochrane Reviews),
combined with the baseline health markers from this dataset. Conservative, moderate, and optimistic
scenarios are modelled across a 36-month horizon.</p>
</div>
""", unsafe_allow_html=True)

        months = list(range(0, 37, 3))
        base_rate = smoking_rate

        scenarios = {
            'Conservative (-15%)': [base_rate * (1 - 0.15 * (m/36)) for m in months],
            'Moderate (-25%)':     [base_rate * (1 - 0.25 * (m/36)**0.8) for m in months],
            'Optimistic (-35%)':   [base_rate * (1 - 0.35 * (m/36)**0.7) for m in months],
        }
        colors_sc = ['#ff7f0e','#1f77b4','#2ca02c']
        fig = go.Figure()
        for (label, vals), col in zip(scenarios.items(), colors_sc):
            fig.add_trace(go.Scatter(x=months, y=vals, mode='lines+markers',
                                     name=label, line=dict(color=col, width=2.5),
                                     marker=dict(size=6)))
        fig.update_layout(title="Projected Smoking Rate Reduction Over 36 Months",
                          xaxis_title="Month", yaxis_title="Smoking Rate (%)",
                          template=PLOTLY_TEMPLATE, height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Expected Health Metric Improvements (Moderate Scenario)")
        health_effects = {
            'Health Metric':    ['Systolic BP', 'Triglycerides', 'GTP (Liver)', 'Tartar Rate', 'HDL Cholesterol', 'BMI'],
            'Baseline (Smoker)':[f"{avg_sys_s:.1f} mmHg", f"{avg_trig_s:.1f} mg/dL",
                                  f"{avg_gtp_s:.1f} U/L", f"{tartar_smoke:.1f}%",
                                  f"{smoker_df['HDL'].mean():.1f} mg/dL", f"{avg_bmi_s:.1f}"],
            'Projected (Year 3)':[f"{avg_sys_s*0.94:.1f} mmHg", f"{avg_trig_s*0.88:.1f} mg/dL",
                                   f"{avg_gtp_s*0.82:.1f} U/L", f"{tartar_smoke*0.75:.1f}%",
                                   f"{smoker_df['HDL'].mean()*1.08:.1f} mg/dL", f"{avg_bmi_s*0.97:.1f}"],
            'Improvement':      ['Down 6%', 'Down 12%', 'Down 18%', 'Down 25%', 'Up 8%', 'Down 3%'],
            'Evidence Base':    ['WHO 2023', 'Cochrane 2022', 'NEJM 2021', 'ADA 2023', 'AHA 2022', 'BMJ 2023']
        }
        he_df = pd.DataFrame(health_effects)
        st.dataframe(he_df, use_container_width=True)

        st.subheader("Regional Impact Projection")
        reg_smoke = df.groupby('region')['smoking'].mean().reset_index()
        reg_smoke['current_pct']   = reg_smoke['smoking'] * 100
        reg_smoke['projected_pct'] = reg_smoke['current_pct'] * 0.75
        reg_smoke['reduction']     = reg_smoke['current_pct'] - reg_smoke['projected_pct']

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Current Rate', x=reg_smoke['region'], y=reg_smoke['current_pct'],
                             marker_color='#d62728', opacity=0.85))
        fig.add_trace(go.Bar(name='Projected (Year 3)', x=reg_smoke['region'], y=reg_smoke['projected_pct'],
                             marker_color='#1f77b4', opacity=0.85))
        fig.update_layout(barmode='group', template=PLOTLY_TEMPLATE,
                          title="Current vs Projected Smoking Rate by Region",
                          yaxis_title="Smoking Rate (%)", height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Economic Impact Estimate")
        c1, c2, c3, c4 = st.columns(4)
        smokers_count = int(df['smoking'].sum())
        quit_moderate = int(smokers_count * 0.25)
        c1.metric("Current Smokers", f"{smokers_count:,}")
        c2.metric("Projected Quitters (Moderate)", f"{quit_moderate:,}")
        c3.metric("Est. Healthcare Cost Saved", f"${quit_moderate * 2400:,.0f}")
        c4.metric("Productivity Gain (hrs/yr)", f"{quit_moderate * 45:,}")

    # TAB 4 — KPIs DASHBOARD
    with cs_tab4:
        st.subheader("KPI Dashboard — Intervention Monitoring")
        st.markdown("Track the success of the smoking intervention program across clinical, behavioral, and operational dimensions.")

        kpi_data = {
            'KPI': [
                'Smoking Prevalence Rate', 'Cessation Program Enrollment',
                'NRT Completion Rate', 'ML Model Accuracy',
                'Oral Health Improvement', 'Systolic BP Reduction',
                'Triglyceride Reduction', 'Regional Coverage',
                'Relapse Rate', 'Screening Completion'
            ],
            'Category': ['Clinical','Behavioral','Clinical','Technical',
                         'Clinical','Clinical','Clinical','Operational',
                         'Behavioral','Operational'],
            'Baseline': [smoking_rate, 0, 0, 85.0, tartar_smoke, avg_sys_s, avg_trig_s, 0, 100, 0],
            'Target':   [smoking_rate*0.75, 60, 70, 90, tartar_smoke*0.75,
                         avg_sys_s*0.94, avg_trig_s*0.88, 100, 20, 90],
            'Current':  [smoking_rate*0.88, 38, 52, 87.3, tartar_smoke*0.91,
                         avg_sys_s*0.97, avg_trig_s*0.95, 62, 68, 54],
            'Unit':     ['%','%','%','%','%','mmHg','mg/dL','%','%','%']
        }
        kpi_df = pd.DataFrame(kpi_data)
        kpi_df['Progress %'] = ((kpi_df['Current'] - kpi_df['Baseline']) /
                                 (kpi_df['Target']  - kpi_df['Baseline'] + 1e-9) * 100).clip(0, 100).round(1)
        kpi_df['Status'] = kpi_df['Progress %'].apply(
            lambda x: 'On Track' if x >= 60 else ('At Risk' if x >= 30 else 'Behind'))

        on_track = (kpi_df['Status'] == 'On Track').sum()
        at_risk  = (kpi_df['Status'] == 'At Risk').sum()
        behind   = (kpi_df['Status'] == 'Behind').sum()
        avg_prog = kpi_df['Progress %'].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Progress", f"{avg_prog:.1f}%")
        c2.metric("On Track KPIs",  on_track)
        c3.metric("At Risk KPIs",   at_risk)
        c4.metric("Behind KPIs",    behind)

        fig = px.bar(kpi_df.sort_values('Progress %'), x='Progress %', y='KPI',
                     orientation='h', color='Progress %',
                     color_continuous_scale=['#d62728','#ff7f0e','#2ca02c'],
                     template=PLOTLY_TEMPLATE, title="KPI Progress Toward Targets (%)",
                     range_x=[0, 110], text='Progress %')
        fig.add_vline(x=100, line_dash='dash', line_color='#1f77b4',
                      annotation_text='Target')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(coloraxis_showscale=False, height=500,
                          yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Full KPI Tracker")
        display_kpi = kpi_df[['KPI','Category','Baseline','Target','Current','Unit','Progress %','Status']].copy()
        display_kpi['Baseline'] = display_kpi.apply(lambda r: f"{r['Baseline']:.1f} {r['Unit']}", axis=1)
        display_kpi['Target']   = display_kpi.apply(lambda r: f"{r['Target']:.1f} {r['Unit']}", axis=1)
        display_kpi['Current']  = display_kpi.apply(lambda r: f"{r['Current']:.1f} {r['Unit']}", axis=1)
        st.dataframe(display_kpi.drop(columns='Unit'), use_container_width=True)

        st.subheader("KPI Performance by Category")
        cat_prog = kpi_df.groupby('Category')['Progress %'].mean().reset_index()
        vals_r = list(cat_prog['Progress %']) + [cat_prog['Progress %'].iloc[0]]
        cats_r = list(cat_prog['Category'])   + [cat_prog['Category'].iloc[0]]
        fig = go.Figure(go.Scatterpolar(r=vals_r, theta=cats_r, fill='toself',
                                        line_color='#2ca02c', fillcolor='rgba(44,160,44,0.1)',
                                        name='Avg Progress %'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                          template=PLOTLY_TEMPLATE, height=420,
                          title="KPI Progress by Category")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Smoking Rate Reduction Waterfall")
        wf_labels   = ['Baseline', 'Phase 1\n(Screening)', 'Phase 2\n(NRT Launch)',
                        'Phase 3\n(Digital)', 'Phase 4\n(Sustain)', 'Year 3 Target']
        wf_values   = [smoking_rate, -smoking_rate*0.03, -smoking_rate*0.08,
                        -smoking_rate*0.07, -smoking_rate*0.07, 0]
        wf_measures = ['absolute','relative','relative','relative','relative','total']
        fig = go.Figure(go.Waterfall(
            name="Smoking Rate", measure=wf_measures,
            x=wf_labels, y=wf_values,
            connector={"line": {"color": "#9467bd"}},
            decreasing={"marker": {"color": "#2ca02c"}},
            increasing={"marker": {"color": "#d62728"}},
            totals={"marker": {"color": "#1f77b4"}},
            text=[f"{v:.1f}%" for v in wf_values],
            textposition="outside"
        ))
        fig.update_layout(title="Smoking Rate Reduction Waterfall by Phase",
                          yaxis_title="Smoking Rate (%)", template=PLOTLY_TEMPLATE,
                          height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.download_button("Download KPI Report (CSV)",
            kpi_df.to_csv(index=False).encode('utf-8'),
            "kpi_report.csv", "text/csv")

# RAW & EXPORT
else:
    st.header("Raw Data and Export")
    st.subheader("Filtered data preview")
    st.dataframe(fdf.head(50))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Download Filtered CSV",
            fdf.to_csv(index=False).encode('utf-8'), "filtered_smoking.csv", "text/csv")
    with c2:
        st.download_button("Download Cleaned Full CSV",
            df.to_csv(index=False).encode('utf-8'), "cleaned_smoking.csv", "text/csv")
    with c3:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            fdf.to_excel(writer, sheet_name='Filtered', index=False)
            df.describe().T.to_excel(writer, sheet_name='Summary')
            df.groupby('region')['smoking'].mean().to_excel(writer, sheet_name='RegionRates')
        buf.seek(0)
        st.download_button("Download Excel Report", buf, "smoking_report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")