from __future__ import annotations

import streamlit as st


def render_report_shimmer():
    """Render a shimmer loading effect for the report tab."""
    st.markdown(
        '''
        <div class="report-loading-shimmer">
            <div class="shimmer-block large"></div>
            <div class="shimmer-block medium"></div>
            <div class="shimmer-block small"></div>
            <div class="shimmer-block medium"></div>
            <div class="shimmer-block large"></div>
            <div class="shimmer-block small"></div>
            <div class="shimmer-block medium"></div>
            <div class="shimmer-block large"></div>
            <div class="shimmer-block small"></div>
            <div class="shimmer-block medium"></div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    st.caption("Generating investment analysis... This may take 30–60 seconds.")


