import streamlit as st

st.set_page_config(page_title="OnboardingDashboard", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)
st.title("This app is down for maintenance. Check back later.")