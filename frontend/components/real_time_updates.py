
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

class RealTimeUpdates:
    """Component for displaying real-time workflow updates."""
    
    def __init__(self):
        pass
    
    def render(self, updates: List[Dict[str, Any]] = None):
        """Render real-time updates feed."""
        st.subheader("📡 Live Updates")
        
        if not updates:
            st.info("No real-time updates available")
            return
        
        # Display recent updates
        for update in updates[-5:]:  # Show last 5 updates
            timestamp = update.get('timestamp', datetime.now().strftime("%H:%M:%S"))
            message = update.get('message', 'Update received')
            st.write(f"`{timestamp}` {message}")
