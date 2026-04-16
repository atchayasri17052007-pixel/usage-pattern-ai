import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------------
# Step 1: Create dataset (training)
# -------------------------------
data = pd.DataFrame({
    'login_count': [10, 2, 5, 8, 1, 6],
    'usage_time': [50, 10, 20, 35, 5, 25],
    'clicks': [200, 30, 80, 150, 10, 100],
    'active_days': [7, 2, 4, 6, 1, 5]
})

# Feature engineering
data['activity_score'] = data['login_count'] * data['usage_time']
data['engagement'] = data['clicks'] / data['login_count']
data['consistency'] = data['active_days'] / 7

features = data[['activity_score', 'engagement', 'consistency']]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Train model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# -------------------------------
# Step 2: Streamlit UI
# -------------------------------
st.title("📊 Usage Pattern Analysis App")

st.write("Enter user details to classify usage pattern")

login = st.number_input("Login Count", min_value=1)
time = st.number_input("Usage Time (minutes)", min_value=1)
clicks = st.number_input("Number of Clicks", min_value=1)
days = st.number_input("Active Days (per week)", min_value=1, max_value=7)

if st.button("Analyze User"):

    # Feature engineering (same as training)
    activity_score = login * time
    engagement = clicks / login
    consistency = days / 7

    input_data = [[activity_score, engagement, consistency]]

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict cluster
    cluster = kmeans.predict(input_scaled)[0]

    # Map clusters (adjust if needed)
    if cluster == 0:
        result = "🔥 High Activity User"
    elif cluster == 1:
        result = "📉 Low Activity User"
    else:
        result = "🔄 Irregular Usage User"

    st.subheader("Result:")
    st.success(result)
