import streamlit as st
st.set_page_config(page_title='AFSA - Student Performance Prediction', layout='wide')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import MinMaxScaler

# Optional: ReliefF
try:
    from skrebate import ReliefF
    HAS_SKREBATE = True
except Exception:
    HAS_SKREBATE = False

st.title('INTELLIGENT FUTURE TUNING FOR ACUURATE STUDENT PERFORMANCE FORECASTING')

with st.sidebar:
    st.header('Upload / Settings')
    uploaded = st.file_uploader(r"C:\Users\aksha\Downloads\student-mat.csv", type=['csv'])
    use_sample = st.checkbox('Use example student dataset (if not uploaded)', value=False)
    RANDOM_STATE = st.number_input('Random seed', value=42, step=1)
    TEST_SIZE = st.slider('Test set fraction', 0.05, 0.5, 0.2)
    CV_FOLDS = st.slider('CV folds', 2, 10, 5)
    n_neighbors_relief = st.slider('Relief n_neighbors', 1, 30, 10)
    start_k = st.number_input('Minimum top-k start', value=3, step=1)
    run_button = st.button('Run AFSA & Train Models')

# Helper functions (copied/adapted from provided notebook code)
@st.cache_data
def load_data_from_file(f):
    df = pd.read_csv(f, sep=None, engine='python')
    return df

@st.cache_data
def load_example():
    # If user asked for example but no internet, create a tiny synthetic demo or instruct user.
    st.info('No example dataset bundled. Please upload student-mat.csv and student-por.csv merged file.')
    return None

# Ranking functions

def rank_info_gain(X, y, random_state=42):
    X_enc = pd.get_dummies(X, drop_first=False)
    dt = DecisionTreeClassifier(criterion='entropy', random_state=random_state)
    dt.fit(X_enc, y.astype(int))
    imp = pd.Series(dt.feature_importances_, index=X_enc.columns).fillna(0.0)

    collapsed = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            grp = [c for c in X_enc.columns if c.startswith(col + "_")]
            collapsed[col] = imp[grp].sum() if grp else 0.0
        else:
            collapsed[col] = imp.get(col, 0.0)
    return pd.Series(collapsed).fillna(0.0)


def rank_mutual_info(X, y):
    X_enc = X.copy()
    for c in X_enc.columns:
        if X_enc[c].dtype == 'object':
            X_enc[c], _ = pd.factorize(X_enc[c], sort=True)
    scores = mutual_info_classif(X_enc.values, y.values.astype(int), random_state=0)
    return pd.Series(scores, index=X_enc.columns).fillna(0.0)


def rank_chi2(X, y):
    X_cat = pd.get_dummies(X, drop_first=False)
    orig_num = [c for c in X.columns if X[c].dtype != 'object']
    for n in orig_num:
        if n in X_cat.columns:
            X_cat[[n]] = MinMaxScaler().fit_transform(X_cat[[n]])
    chi2_scores, _ = chi2(X_cat.values, y.values.astype(int))
    chi2_series = pd.Series(chi2_scores, index=X_cat.columns).fillna(0.0)

    collapsed = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            grp = [c for c in X_cat.columns if c.startswith(col + "_")]
            collapsed[col] = chi2_series[grp].sum() if grp else 0.0
        else:
            collapsed[col] = chi2_series.get(col, 0.0)
    return pd.Series(collapsed).fillna(0.0)


def rank_rf_gini(X, y, random_state=42):
    X_cat = pd.get_dummies(X, drop_first=False)
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state,
                                class_weight='balanced', n_jobs=-1)
    rf.fit(X_cat, y.astype(int))
    imp = pd.Series(rf.feature_importances_, index=X_cat.columns).fillna(0.0)

    collapsed = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            grp = [c for c in X_cat.columns if c.startswith(col + "_")]
            collapsed[col] = imp[grp].sum() if grp else 0.0
        else:
            collapsed[col] = imp.get(col, 0.0)
    return pd.Series(collapsed).fillna(0.0)


def rank_relief(X, y, n_neighbors=10):
    X_enc = X.copy()
    X_num_df = pd.get_dummies(X_enc, drop_first=False)
    X_num = X_num_df.astype(float).values
    y_np = y.values.astype(int)

    if HAS_SKREBATE:
        rel = ReliefF(n_neighbors=n_neighbors, n_features_to_select=X_num.shape[1])
        rel.fit(X_num, y_np)
        scores = pd.Series(rel.feature_importances_, index=X_num_df.columns)
    else:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min(n_neighbors+1, X_num.shape[0])).fit(X_num)
        distances, indices = nbrs.kneighbors(X_num)
        indices = indices[:, 1:]

        scores_arr = np.zeros(X_num.shape[1], dtype=float)
        for i in range(X_num.shape[0]):
            xi = X_num[i]
            for j in indices[i]:
                xj = X_num[j]
                if y_np[i] == y_np[j]:
                    scores_arr -= np.abs(xi - xj)
                else:
                    scores_arr += np.abs(xi - xj)

        scores_arr = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min() + 1e-12)
        scores = pd.Series(scores_arr, index=X_num_df.columns)

    collapsed = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            grp = [c for c in scores.index if c.startswith(col + "_")]
            collapsed[col] = scores[grp].sum() if grp else 0.0
        else:
            collapsed[col] = scores.get(col, 0.0)

    return pd.Series(collapsed).fillna(0.0)


def evaluate_model_cv(model, X, y, cv_folds=5):
    scoring = {'accuracy':'accuracy', 'precision':'precision', 'recall':'recall', 'f1':'f1'}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1, return_train_score=False)
    result = {k.replace('test_',''): np.mean(v) for k,v in scores.items() if k.startswith('test_')}
    return result

# Main run
if run_button:
    # Load data
    if uploaded is not None:
        df = load_data_from_file(uploaded)
    elif use_sample:
        df = load_example()
        if df is None:
            st.stop()
    else:
        st.error('Please upload a CSV or enable example dataset.')
        st.stop()

    st.write('Raw dataset shape:', df.shape)
    st.dataframe(df.head())

    # Safe handling of missing values
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for c in num_cols:
        if df[c].isna().sum():
            df[c].fillna(df[c].median(), inplace=True)
    for c in cat_cols:
        if df[c].isna().sum():
            df[c].fillna(df[c].mode().iloc[0], inplace=True)

    # Target creation
    if 'G3' not in df.columns:
        st.error('The dataset must contain column G3 (final grade) to build target_pass (G3>=10).')
        st.stop()

    df['target_pass'] = (df['G3'] >= 10).astype(int)
    X = df.drop(columns=['G1','G2','G3','target_pass'], errors='ignore')
    y = df['target_pass']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    st.write('Train / Test sizes:', X_train.shape, X_test.shape)

    # Compute rankings
    with st.spinner('Computing feature rankings...'):
        mi_scores = rank_mutual_info(X_train, y_train)
        info_scores = rank_info_gain(X_train, y_train, random_state=RANDOM_STATE)
        chi2_scores = rank_chi2(X_train, y_train)
        rf_scores = rank_rf_gini(X_train, y_train, random_state=RANDOM_STATE)
        relief_scores = rank_relief(X_train, y_train, n_neighbors=n_neighbors_relief)

        ranks_df = pd.DataFrame({
            'info_gain': info_scores,
            'mutual_info': mi_scores,
            'chi2': chi2_scores,
            'rf_gini': rf_scores,
            'relief': relief_scores
        }).fillna(0.0)

        normed = ranks_df.copy()
        for c in normed.columns:
            ranks = normed[c].rank(ascending=False, method='average')
            normed[c] = 1.0 - (ranks - 1) / (len(ranks) - 1 if len(ranks) > 1 else 1)
        normed['agg_score'] = normed.mean(axis=1)
        agg_sorted = normed.sort_values('agg_score', ascending=False)

    st.subheader('Top features by AFSA aggregated score')
    st.dataframe(agg_sorted[['agg_score']].head(30))

    # Visualize ranking heatmap
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(ranks_df, annot=True, ax=ax)
    ax.set_title('Raw feature scores (each method)')
    st.pyplot(fig)

    # Correlation heatmap for numeric subset
    num_corr = df.select_dtypes(include=[np.number]).corr()
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(num_corr, annot=False, ax=ax2)
    ax2.set_title('Numeric feature correlation')
    st.pyplot(fig2)

    # Model definitions
    models = {
        'LogisticRegression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE, solver='liblinear'),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(probability=True, kernel='rbf', random_state=RANDOM_STATE),
        'NaiveBayes': GaussianNB(),
        'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
    }

    feature_order = list(agg_sorted.index)
    k_vals = list(range(int(max(3, start_k)), len(feature_order)+1))

    results = []

    with st.spinner('Running CV across top-k and models (this may take time)...'):
        for k in k_vals:
            topk = feature_order[:k]
            Xk_train = pd.get_dummies(X_train[topk], drop_first=False)
            orig_num = [c for c in topk if X_train[c].dtype != 'object']
            numeric_expanded = [c for c in Xk_train.columns if c in orig_num]
            if numeric_expanded:
                Xk_train[numeric_expanded] = StandardScaler().fit_transform(Xk_train[numeric_expanded])

            for name, model in models.items():
                try:
                    res = evaluate_model_cv(model, Xk_train, y_train, cv_folds=CV_FOLDS)
                    results.append({'k':k, 'model':name, 'accuracy': res.get('accuracy', np.nan), 'precision': res.get('precision', np.nan), 'recall': res.get('recall', np.nan), 'f1': res.get('f1', np.nan)})
                except Exception as e:
                    results.append({'k':k, 'model':name, 'accuracy':np.nan, 'precision':np.nan, 'recall':np.nan, 'f1':np.nan})

    results_df = pd.DataFrame(results)

    # Show CV results sample
    st.subheader('Cross-validated results (sample)')
    st.dataframe(results_df.head(50))

    # Plot CV F1 vs k
    fig3, ax3 = plt.subplots(figsize=(10,6))
    for name in models.keys():
        dfm = results_df[results_df['model']==name]
        ax3.plot(dfm['k'], dfm['f1'], label=name)
    ax3.set_xlabel('k (top features)')
    ax3.set_ylabel('CV mean F1')
    ax3.set_title('CV F1 vs top-k features')
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    # Find best k per model
    best_per_model = {}
    for name in results_df['model'].unique():
        dfm = results_df[results_df['model']==name].dropna(subset=['f1'])
        if dfm.empty:
            continue
        best_row = dfm.loc[dfm['f1'].idxmax()]
        best_per_model[name] = best_row.to_dict()

    # Final evaluation on test set for each best model
    final_results = []
    for name, best in best_per_model.items():
        best_k = int(best['k'])
        topk = feature_order[:best_k]
        Xtr = pd.get_dummies(X_train[topk], drop_first=False)
        Xte = pd.get_dummies(X_test[topk], drop_first=False)
        Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
        orig_num = [c for c in topk if X_train[c].dtype != 'object']
        numeric_expanded = [c for c in Xtr.columns if c in orig_num]
        if numeric_expanded:
            scaler = StandardScaler().fit(Xtr[numeric_expanded])
            Xtr[numeric_expanded] = scaler.transform(Xtr[numeric_expanded])
            Xte[numeric_expanded] = scaler.transform(Xte[numeric_expanded])
        model = models[name]
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1v = f1_score(y_test, y_pred, zero_division=0)
        final_results.append({'model': name, 'best_k': best_k, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1v})

    final_df = pd.DataFrame(final_results).sort_values('f1', ascending=False).reset_index(drop=True)
    st.subheader('Final test-set evaluation for best k per model')
    st.dataframe(final_df)

    # Allow user to download aggregated rankings, cv results, and final results
    def to_download(df):
        b = BytesIO()
        df.to_csv(b, index=False)
        b.seek(0)
        return b

    st.download_button('Download aggregated ranks (CSV)', data=to_download(agg_sorted.reset_index().rename(columns={'index':'feature'})), file_name='agg_ranks.csv')
    st.download_button('Download CV results (CSV)', data=to_download(results_df), file_name='cv_results.csv')
    st.download_button('Download final test results (CSV)', data=to_download(final_df), file_name='final_results.csv')
        # Generate "At Risk" or "Safe" predictions for best overall model
        # 🔽 Generate "At Risk" or "Safe" predictions for best overall model 🔽
    best_model_name = final_df.loc[final_df['f1'].idxmax(), 'model']
    best_model_k = int(final_df.loc[final_df['f1'].idxmax(), 'best_k'])
    topk = feature_order[:best_model_k]

    # Prepare training and full dataset with identical encoding
    Xtr = pd.get_dummies(X_train[topk], drop_first=False)
    X_full = pd.get_dummies(df[topk], drop_first=False)
    X_full = X_full.reindex(columns=Xtr.columns, fill_value=0)

    # Scale numeric columns
    orig_num = [c for c in topk if X_train[c].dtype != 'object']
    numeric_expanded = [c for c in Xtr.columns if c in orig_num]
    if numeric_expanded:
        scaler = StandardScaler().fit(Xtr[numeric_expanded])
        Xtr[numeric_expanded] = scaler.transform(Xtr[numeric_expanded])
        X_full[numeric_expanded] = scaler.transform(X_full[numeric_expanded])

    # Train the best model on training data and predict for all students
    best_model = models[best_model_name]
    best_model.fit(Xtr, y_train)
    y_pred_full = best_model.predict(X_full)

    # Add "Risk_Status" column
    df['Risk_Status'] = np.where(y_pred_full == 1, 'Safe', 'At Risk')

    st.subheader('Student Risk Prediction')
    st.dataframe(df[['Risk_Status']].head(20))

    # Downloadable CSV
    st.download_button(
        'Download Student Risk Predictions (CSV)',
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='student_risk_predictions.csv',
        mime='text/csv'
    )



    st.success('AFSA run complete!')

else:
    st.info('Configure options in the sidebar and click Run AFSA & Train Models')
