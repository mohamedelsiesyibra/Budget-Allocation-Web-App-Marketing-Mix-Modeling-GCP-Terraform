# ===========================================
# Import Necessary Libraries
# ===========================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from lightweight_mmm import optimize_media
import plotly.graph_objects as go
import os
from google.cloud import storage
import tempfile

# ===========================================
# Set Streamlit Page Configuration
# ===========================================

st.set_page_config(
    page_title="📈 Marketing Mix Optimization",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================
# Define Marketing Channels and Revenue
# ===========================================

# Updated list of marketing channels based on the trained model
CHANNELS = [
    'google_ads',
    'tiktok_spend',
    'facebook_spend',
    'print_spend',
    'ooh_spend',
    'tv_spend',
    'podcast_radio_spend'
]

# Mapping of internal channel names to user-friendly display names
CHANNEL_DISPLAY_NAMES = {
    'google_ads': 'Google Ads',
    'tiktok_spend': 'TikTok Ads',
    'facebook_spend': 'Facebook Ads',
    'print_spend': 'Print Ads',
    'ooh_spend': 'Out-of-Home Ads',
    'tv_spend': 'TV Ads',
    'podcast_radio_spend': 'Podcast & Radio Ads'
}

# ===========================================
# Function to Load Model and Scalers from GCS
# ===========================================

@st.cache_resource
def load_model_from_gcs(bucket_name, blob_name):
    """
    Load the trained model and scalers from a pickle file in GCS.

    Args:
        bucket_name (str): Name of the GCS bucket.
        blob_name (str): Path to the blob in the GCS bucket.

    Returns:
        tuple: Contains the model, media scaler, target scaler, and cost scaler.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with tempfile.NamedTemporaryFile() as temp_file:
        blob.download_to_filename(temp_file.name)
        with open(temp_file.name, 'rb') as file:
            data = pickle.load(file)
    return data['model'], data['media_scaler'], data['target_scaler'], data['cost_scaler']

# ===========================================
# Load the Model and Scalers from GCS
# ===========================================

# GCS bucket and blob information
BUCKET_NAME = 'lightweight-mmm-pipeline'
MODEL_BLOB_PATH = 'models/latest_version/model_and_scalers.pkl'

# Load the model and scalers
try:
    mmm_model, media_scaler, target_scaler, cost_scaler = load_model_from_gcs(BUCKET_NAME, MODEL_BLOB_PATH)
except Exception as e:
    st.error(f"Failed to load the model from GCS: {e}")
    st.stop()

# ===========================================
# Page Title and Description
# ===========================================

st.title("📈 Marketing Mix Optimization")

st.markdown("""
Optimize your marketing budget allocation across various channels to maximize your **revenue** and **Return on Ad Spend (ROAS)**. 
Adjust the parameters below and click **Optimize** to see the optimal spend distribution.
""")
st.markdown("""
**Note:**  
- **Optimal** values represent the projected outcomes after optimizing the budget allocation.  
- **Actual** values represent the current performance with the existing budget allocation.
""", unsafe_allow_html=True)

st.markdown("""---""")  # Horizontal line for separation

# ===========================================
# Sidebar: User Input Form
# ===========================================

st.sidebar.header("Adjust Parameters")

with st.sidebar.form(key='optimization_form'):
    # Input for Total Budget
    budget = st.number_input(
        label="Total Budget ($)",
        min_value=10000,
        max_value=10000000,
        value=500000,
        step=1000,
        help="Enter the total marketing budget available for allocation."
    )

    # Input for Number of Time Periods (Weeks)
    n_time_periods = st.slider(
        label="Number of Periods (Weeks)",
        min_value=1,
        max_value=24,
        value=4,
        step=1,
        help="Select the number of weeks over which the budget will be allocated."
    )

    # Submit Button to Trigger Optimization
    optimize_button = st.form_submit_button(label='Optimize')

# ===========================================
# Initialize Session State for Optimization
# ===========================================

if 'optimization_done' not in st.session_state:
    st.session_state.optimization_done = False
    st.session_state.budget = budget
    st.session_state.n_time_periods = n_time_periods

    # Assume prices are 1 for simplicity; modify as needed based on actual pricing
    prices = np.ones(len(CHANNELS), dtype=float)

    # Run initial optimization with default values
    solution, kpi_without_optim, previous_budget_allocation = optimize_media.find_optimal_budgets(
        n_time_periods=n_time_periods,
        media_mix_model=mmm_model,
        budget=budget,
        prices=prices,
        media_scaler=media_scaler,
        target_scaler=target_scaler
    )

    # Calculate Key Performance Indicators (KPIs)
    current_revenue = abs(kpi_without_optim)
    optimal_revenue = abs(solution.fun)

    current_spend = max(abs(np.sum(previous_budget_allocation)), 1) 
    optimal_spend = max(abs(np.sum(prices * solution.x)), 1)

    current_roas = current_revenue / current_spend
    optimal_roas = optimal_revenue / optimal_spend

    # Calculate ROAS growth percentage
    roas_growth = ((optimal_roas - current_roas) / current_roas) * 100 if current_roas != 0 else 0

    # Store results in session state
    st.session_state.solution = solution
    st.session_state.kpi_without_optim = current_revenue
    st.session_state.previous_budget_allocation = previous_budget_allocation
    st.session_state.optimal_revenue = optimal_revenue
    st.session_state.current_revenue = current_revenue
    st.session_state.optimal_spend = optimal_spend
    st.session_state.current_spend = current_spend
    st.session_state.current_roas = current_roas
    st.session_state.optimal_roas = optimal_roas
    st.session_state.roas_growth = roas_growth
    st.session_state.optimization_done = True

# ===========================================
# Handle Form Submission for Optimization
# ===========================================

if optimize_button:
    # Update session state with new inputs
    st.session_state.budget = budget
    st.session_state.n_time_periods = n_time_periods

    # Assume prices are 1 for simplicity; modify as needed based on actual pricing
    prices = np.ones(len(CHANNELS), dtype=float)

    # Run optimization with updated inputs
    solution, kpi_without_optim, previous_budget_allocation = optimize_media.find_optimal_budgets(
        n_time_periods=n_time_periods,
        media_mix_model=mmm_model,
        budget=budget,
        prices=prices,
        media_scaler=media_scaler,
        target_scaler=target_scaler
    )

    # Calculate Key Performance Indicators (KPIs)
    current_revenue = abs(kpi_without_optim)
    optimal_revenue = abs(solution.fun)

    current_spend = max(abs(np.sum(previous_budget_allocation)), 1) 
    optimal_spend = max(abs(np.sum(prices * solution.x)), 1)

    current_roas = current_revenue / current_spend
    optimal_roas = optimal_revenue / optimal_spend

    # Calculate ROAS growth percentage
    roas_growth = ((optimal_roas - current_roas) / current_roas) * 100 if current_roas != 0 else 0

    # Store results in session state
    st.session_state.solution = solution
    st.session_state.kpi_without_optim = current_revenue
    st.session_state.previous_budget_allocation = previous_budget_allocation
    st.session_state.optimal_revenue = optimal_revenue
    st.session_state.current_revenue = current_revenue
    st.session_state.optimal_spend = optimal_spend
    st.session_state.current_spend = current_spend
    st.session_state.current_roas = current_roas
    st.session_state.optimal_roas = optimal_roas
    st.session_state.roas_growth = roas_growth
    st.session_state.optimization_done = True

# ===========================================
# Display Dashboard After Optimization
# ===========================================

if st.session_state.optimization_done:
    # Retrieve data from session state
    solution = st.session_state.solution
    kpi_without_optim = st.session_state.kpi_without_optim
    previous_budget_allocation = st.session_state.previous_budget_allocation
    optimal_revenue = st.session_state.optimal_revenue
    current_revenue = st.session_state.current_revenue
    optimal_spend = st.session_state.optimal_spend
    current_spend = st.session_state.current_spend
    current_roas = st.session_state.current_roas
    optimal_roas = st.session_state.optimal_roas
    roas_growth = st.session_state.roas_growth
    budget = st.session_state.budget

    # ===========================================
    # Display Key Performance Indicators (KPIs)
    # ===========================================

    st.subheader("Key Performance Indicators (KPIs)")

    # Arrange metrics in three equal-width columns
    col1, col2, col3 = st.columns([1, 1, 1])

    # Display Total Budget
    with col1:
        st.metric(
            label="💰 Total Budget",
            value=f"${budget:,.2f}",
            delta=None
        )

    # Display Optimal Revenue
    with col2:
        st.metric(
            label="💰 Optimal Revenue",
            value=f"${optimal_revenue:,.2f}",
            delta=None  
        )

    # Display Optimal ROAS with Growth Percentage
    with col3:
        # Format delta with '+' sign if positive
        delta_formatted = f"+{roas_growth:.2f}%" if roas_growth >= 0 else f"{roas_growth:.2f}%"
        st.metric(
            label="📈 Optimal ROAS",
            value=f"{optimal_roas:.2f}",
            delta=delta_formatted,
            delta_color="normal" 
        )

    st.markdown("""---""")  

    # ===========================================
    # Prepare and Display Spend Comparison Chart
    # ===========================================

    # Create a DataFrame for visualization with user-friendly channel names
    spend_df = pd.DataFrame({
        'Channel': [CHANNEL_DISPLAY_NAMES[channel] for channel in CHANNELS],
        'Actual Spend': previous_budget_allocation,
        'Optimal Spend': solution.x
    })

    # Calculate spend percentages relative to the total budget
    spend_df['Actual Spend Percentage'] = (spend_df['Actual Spend'] / budget) * 100
    spend_df['Optimal Spend Percentage'] = (spend_df['Optimal Spend'] / budget) * 100

    # Create a grouped horizontal bar chart using Plotly
    fig = go.Figure()

    # Add Actual Spend bars
    fig.add_trace(go.Bar(
        y=spend_df['Channel'],
        x=spend_df['Actual Spend'],
        name='Actual Spend',
        orientation='h',
        marker=dict(color='rgba(255, 75, 75, 1)'),  
        text=[f"${val:,.0f} ({pct:.0f}%)" for val, pct in zip(spend_df['Actual Spend'], spend_df['Actual Spend Percentage'])],
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(size=14, color='white'), 
    ))  # White text for visibility

    # Add Optimal Spend bars
    fig.add_trace(go.Bar(
        y=spend_df['Channel'],
        x=spend_df['Optimal Spend'],
        name='Optimal Spend',
        orientation='h',
        marker=dict(color='rgba(0, 128, 0, 1)'), 
        text=[f"${val:,.0f} ({pct:.0f}%)" for val, pct in zip(spend_df['Optimal Spend'], spend_df['Optimal Spend Percentage'])],
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(size=14, color='white'),  
    ))  

    # Update layout for better aesthetics
    fig.update_layout(
        barmode='group',
        title={
            'text': 'Actual vs Optimal Spend by Channel',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Spend ($)',
        yaxis_title='Marketing Channel',
        yaxis={
            'categoryorder': 'total ascending',
            'tickfont': dict(size=16, family='Arial', color='black', weight='bold')  
        },
        legend=dict(
            title='Spend Type',
            font=dict(
                size=14,
                family="Arial, sans-serif",
                color="black",
            )
        ),
        height=600,
        margin=dict(l=150, r=50, t=100, b=50),  
    )

    # Update legend font to be larger and bold
    fig.update_layout(legend_font=dict(size=14, family="Arial, sans-serif"))

    # Remove mode bar (zoom, download, etc.) for a cleaner look
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("""---""") 

    # ===========================================
    # Display Per Channel Breakdown Table
    # ===========================================

    st.subheader("Budget and Performance per Channel")

    # Calculate Actual and Optimal Revenue per channel
    spend_df['Actual Revenue'] = spend_df['Actual Spend'] * current_roas
    spend_df['Optimal Revenue'] = spend_df['Optimal Spend'] * optimal_roas

    # Rearrange and rename columns for clarity
    breakdown_df = spend_df[['Channel', 'Actual Spend', 'Optimal Spend', 'Actual Revenue', 'Optimal Revenue']].copy()

    # Rename columns for better presentation
    breakdown_df.rename(columns={
        'Actual Spend': 'Actual Spend ($)',
        'Optimal Spend': 'Optimal Spend ($)',
        'Actual Revenue': 'Actual Revenue ($)',
        'Optimal Revenue': 'Optimal Revenue ($)'
    }, inplace=True)

    # Format numerical columns for better readability
    breakdown_df['Actual Spend ($)'] = breakdown_df['Actual Spend ($)'].apply(lambda x: f"${x:,.2f}")
    breakdown_df['Optimal Spend ($)'] = breakdown_df['Optimal Spend ($)'].apply(lambda x: f"${x:,.2f}")
    breakdown_df['Actual Revenue ($)'] = breakdown_df['Actual Revenue ($)'].apply(lambda x: f"${x:,.2f}")
    breakdown_df['Optimal Revenue ($)'] = breakdown_df['Optimal Revenue ($)'].apply(lambda x: f"${x:,.2f}")

    # Define the order of columns
    column_order = ['Channel', 'Actual Spend ($)', 'Optimal Spend ($)', 'Actual Revenue ($)', 'Optimal Revenue ($)']

    # Display the breakdown table with centered text and no index
    st.dataframe(
        breakdown_df[column_order],
        use_container_width=True,
        hide_index=True, 
    )

    st.markdown("""
        ---
        **Developed by Mohamed Elsiesy**
        """)

else:
    # Prompt user to adjust parameters and optimize
    st.write("")
    st.write("Adjust the parameters in the sidebar and click **Optimize** to see the results.")
