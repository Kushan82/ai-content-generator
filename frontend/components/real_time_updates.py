import streamlit as st
from datetime import datetime
from typing import List, Dict, Any, Optional

class RealTimeUpdates:
    """Component for displaying real-time workflow updates."""
    
    def __init__(self, websocket_client=None):
        self.websocket_client = websocket_client
        self.update_cache = []
        self.max_updates = 50  # Maximum number of updates to keep
    
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
    
    def render_live_updates(self, updates: List[Dict[str, Any]]):
        """Render live updates with enhanced formatting."""
        st.markdown("### 📡 Live Updates")
        
        if not updates:
            st.info("⏳ Waiting for workflow updates...")
            return
        
        # Show recent updates with enhanced styling
        for update in updates[-10:]:  # Show last 10 updates
            self._render_update_item(update)
    
    def render_updates_panel(self):
        """Render comprehensive updates panel."""
        st.markdown("### 📺 System Updates")
        
        # Get updates from session state
        if 'real_time_updates' in st.session_state:
            updates = st.session_state.real_time_updates
        else:
            updates = []
        
        if not updates:
            self._render_no_updates_state()
            return
        
        # Filter controls
        self._render_update_filters()
        
        # Updates list
        self._render_updates_list(updates)
    
    def _render_update_item(self, update: Dict[str, Any]):
        """Render individual update item with styling."""
        timestamp = update.get('timestamp', datetime.now().strftime("%H:%M:%S"))
        message = update.get('message', 'Update received')
        step = update.get('step', 'unknown')
        agent = update.get('agent', 'system')
        progress = update.get('progress', 0)
        
        # Determine update type styling
        if 'completed' in message.lower():
            icon = '✅'
            color = '#28a745'
        elif 'started' in message.lower() or 'working' in message.lower():
            icon = '⚡'
            color = '#ffc107'
        elif 'error' in message.lower() or 'failed' in message.lower():
            icon = '❌'
            color = '#dc3545'
        else:
            icon = '📡'
            color = '#007bff'
        
        # Render update card
        st.markdown(f"""
        <div style="
            border-left: 4px solid {color};
            background: #f8f9fa;
            padding: 0.8rem;
            margin: 0.5rem 0;
            border-radius: 0 8px 8px 0;
            transition: all 0.3s ease;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="color: {color};">{icon} {timestamp}</strong>
                    <span style="margin-left: 1rem; color: #6c757d; font-size: 0.9em;">
                        Agent: {agent.replace('_', ' ').title()}
                    </span>
                </div>
                <div style="font-size: 0.8em; color: #6c757d;">
                    {progress}%
                </div>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.95em;">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_no_updates_state(self):
        """Render state when no updates are available."""
        st.markdown("""
        <div style="
            text-align: center;
            padding: 2rem;
            background: #f8f9fa;
            border-radius: 10px;
            border: 2px dashed #dee2e6;
        ">
            <div style="font-size: 3em; margin-bottom: 1rem;">📡</div>
            <h4 style="color: #6c757d; margin-bottom: 0.5rem;">No Updates Yet</h4>
            <p style="color: #6c757d; margin: 0;">
                Start a workflow to see real-time updates here
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_update_filters(self):
        """Render update filtering controls."""
        with st.expander("🔍 Filter Updates", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                update_type = st.selectbox(
                    "Type:",
                    ["All", "Completed", "Working", "Errors"],
                    key="update_type_filter"
                )
            
            with col2:
                agent_filter = st.selectbox(
                    "Agent:",
                    ["All", "Persona Researcher", "Content Strategist", "Creative Generator", "Quality Assurance"],
                    key="agent_filter"
                )
            
            with col3:
                time_range = st.selectbox(
                    "Time Range:",
                    ["Last 10", "Last 25", "All"],
                    key="time_range_filter"
                )
    
    def _render_updates_list(self, updates: List[Dict[str, Any]]):
        """Render filtered list of updates."""
        # Apply filters (simplified implementation)
        filtered_updates = updates
        
        # Apply time range filter
        time_range = st.session_state.get('time_range_filter', 'Last 10')
        if time_range == 'Last 10':
            filtered_updates = filtered_updates[-10:]
        elif time_range == 'Last 25':
            filtered_updates = filtered_updates[-25:]
        
        # Display updates
        if filtered_updates:
            for update in reversed(filtered_updates):  # Show newest first
                self._render_update_item(update)
        else:
            st.info("No updates match the current filters")
    
    def add_update(self, message: str, agent: str = "system", step: str = "unknown", progress: int = 0):
        """Add a new update to the feed."""
        update = {
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'message': message,
            'agent': agent,
            'step': step,
            'progress': progress
        }
        
        # Add to session state
        if 'real_time_updates' not in st.session_state:
            st.session_state.real_time_updates = []
        
        st.session_state.real_time_updates.append(update)
        
        # Keep only the last N updates to prevent memory issues
        if len(st.session_state.real_time_updates) > self.max_updates:
            st.session_state.real_time_updates = st.session_state.real_time_updates[-self.max_updates:]
    
    def render_compact_updates(self, updates: List[Dict[str, Any]], max_items: int = 3):
        """Render compact version of updates for sidebar or small spaces."""
        st.markdown("#### 📱 Recent Activity")
        
        if not updates:
            st.info("No recent activity")
            return
        
        # Show only the most recent updates
        recent_updates = updates[-max_items:]
        
        for update in reversed(recent_updates):
            timestamp = update.get('timestamp', 'Unknown')
            message = update.get('message', 'Update')
            
            # Truncate long messages
            if len(message) > 50:
                message = message[:47] + "..."
            
            st.markdown(f"""
            <div style="
                font-size: 0.85em;
                padding: 0.4rem;
                margin: 0.2rem 0;
                background: #f8f9fa;
                border-radius: 5px;
                border-left: 3px solid #007bff;
            ">
                <strong>{timestamp}</strong><br>
                {message}
            </div>
            """, unsafe_allow_html=True)
    
    def get_update_statistics(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about updates."""
        if not updates:
            return {
                'total_updates': 0,
                'updates_by_agent': {},
                'recent_activity': 0
            }
        
        # Count updates by agent
        agent_counts = {}
        for update in updates:
            agent = update.get('agent', 'unknown')
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        # Count recent activity (last 10 updates)
        recent_activity = len(updates[-10:])
        
        return {
            'total_updates': len(updates),
            'updates_by_agent': agent_counts,
            'recent_activity': recent_activity
        }
    
    def clear_updates(self):
        """Clear all updates from the session state."""
        if 'real_time_updates' in st.session_state:
            st.session_state.real_time_updates = []
    
    def export_updates(self, updates: List[Dict[str, Any]]) -> str:
        """Export updates as formatted text."""
        if not updates:
            return "No updates to export"
        
        export_text = f"Real-time Updates Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        export_text += "=" * 50 + "\n\n"
        
        for update in updates:
            timestamp = update.get('timestamp', 'Unknown')
            message = update.get('message', 'Update')
            agent = update.get('agent', 'system')
            step = update.get('step', 'unknown')
            
            export_text += f"[{timestamp}] {agent.upper()} ({step})\n"
            export_text += f"    {message}\n\n"
        
        return export_text