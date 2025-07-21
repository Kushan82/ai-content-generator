import streamlit as st
from typing import List, Dict, Any

class ContentComparison:
    """Component for comparing different content variations."""
    
    def __init__(self):
        pass
    
    def render(self, content_variations: List[str] = None):
        """Render content comparison interface."""
        st.subheader("📊 Content Variations")
        
        if not content_variations:
            st.info("No content variations available for comparison")
            return
        
        # Simple comparison display
        for i, content in enumerate(content_variations):
            with st.expander(f"Variation {i+1}"):
                st.write(content)
