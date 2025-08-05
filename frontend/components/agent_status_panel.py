import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

class AgentStatusPanel:
    """Enhanced component for displaying comprehensive agent status and performance monitoring."""
    
    def __init__(self, api_client):
        self.api_client = api_client
        self.status_history = []
        self.max_history = 100

    def render(self):
        """Render the complete agent status panel."""
        st.subheader("🤖 Multi-Agent System Status")
        
        try:
            # Get agent status data
            agent_status = self.api_client.get_agent_status()
            
            # Render agent overview cards
            self.render_agent_overview(agent_status)
            
            st.markdown("---")
            
            # Render detailed agent metrics
            col1, col2 = st.columns(2)
            
            with col1:
                self.render_agent_performance_metrics(agent_status)
            
            with col2:
                self.render_agent_activity_timeline()
                
        except Exception as e:
            st.error(f"Failed to load agent status: {str(e)}")

    def render_enhanced(self):
        """Render enhanced agent status panel with advanced monitoring."""
        st.markdown("### 🤖 Enhanced Agent Monitoring")
        
        try:
            agent_status = self.api_client.get_agent_status()
            
            # Enhanced monitoring tabs
            tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Performance", "🔧 Diagnostics"])
            
            with tab1:
                self._render_enhanced_overview(agent_status)
            
            with tab2:
                self._render_performance_dashboard(agent_status)
            
            with tab3:
                self._render_diagnostics_panel(agent_status)
                
        except Exception as e:
            st.error(f"❌ Enhanced monitoring unavailable: {str(e)}")
            self._render_fallback_status()

    def render_system_status(self):
        """Render comprehensive system status overview."""
        st.markdown("### 🖥️ System Status Dashboard")
        
        try:
            agent_status = self.api_client.get_agent_status()
            health_status = self.api_client.get_health_status()
            
            # System health overview
            self._render_system_health_overview(health_status, agent_status)
            
            # Agent grid view
            self._render_agent_grid(agent_status)
            
        except Exception as e:
            st.error(f"System status unavailable: {str(e)}")

    def render_agent_overview(self, agent_status: Dict[str, Any]):
        """Render overview cards for all agents."""
        if not agent_status:
            st.warning("No agent status data available")
            return
        
        # Create columns for agent cards
        cols = st.columns(min(len(agent_status), 5))  # Max 5 columns
        
        for i, (agent_id, status) in enumerate(agent_status.items()):
            with cols[i % 5]:
                self.render_agent_card(agent_id, status)

    def render_agent_card(self, agent_id: str, status: Dict[str, Any]):
        """Render individual agent status card."""
        agent_name = status.get('name', agent_id.replace('_', ' ').title())
        agent_status = status.get('status', 'unknown')
        uptime = status.get('uptime', 0)
        success_rate = status.get('metrics', {}).get('success_rate', 0)
        
        # Status color coding
        status_colors = {
            'idle': '#28a745',      # Green
            'working': '#ffc107',   # Yellow
            'error': '#dc3545',     # Red
            'thinking': '#17a2b8'   # Blue
        }
        status_color = status_colors.get(agent_status, '#6c757d')
        
        status_emoji = {
            'idle': '🟢',
            'working': '🟡', 
            'error': '🔴',
            'thinking': '🔵'
        }.get(agent_status, '⚪')
        
        # Render enhanced card
        st.markdown(f"""
        <div style='
            border: 2px solid {status_color}; 
            border-radius: 10px; 
            padding: 1rem; 
            margin: 0.5rem 0;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        '>
            <h4 style='color: {status_color}; margin: 0;'>{status_emoji} {agent_name}</h4>
            <hr style='margin: 0.5rem 0; border-color: {status_color};'>
            <p><strong>Status:</strong> {agent_status.title()}</p>
            <p><strong>Uptime:</strong> {uptime:.1f}s</p>
            <p><strong>Success Rate:</strong> {success_rate*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    def render_agent_performance_metrics(self, agent_status: Dict[str, Any]):
        """Render detailed performance metrics."""
        st.markdown("#### 📊 Performance Metrics")
        
        if not agent_status:
            st.info("No performance data available")
            return
        
        # Performance summary
        total_tasks = sum(status.get('metrics', {}).get('total_tasks_completed', 0) 
                         for status in agent_status.values())
        avg_success_rate = sum(status.get('metrics', {}).get('success_rate', 0) 
                              for status in agent_status.values()) / len(agent_status)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Tasks", total_tasks)
        with col2:
            st.metric("Avg Success Rate", f"{avg_success_rate*100:.1f}%")
        with col3:
            active_agents = len([s for s in agent_status.values() 
                               if s.get('status') != 'error'])
            st.metric("Active Agents", f"{active_agents}/{len(agent_status)}")

    def render_agent_activity_timeline(self):
        """Render agent activity timeline."""
        st.markdown("#### 📅 Recent Activity")
        
        # Mock activity data (in production, this would be real data)
        activities = [
            {"time": "14:32:15", "agent": "Content Strategist", "action": "Completed task", "status": "success"},
            {"time": "14:31:48", "agent": "Creative Generator", "action": "Started generation", "status": "working"},
            {"time": "14:31:22", "agent": "Persona Research", "action": "Analysis complete", "status": "success"},
            {"time": "14:30:55", "agent": "Quality Assurance", "action": "Validation passed", "status": "success"},
        ]
        
        for activity in activities:
            status_icon = "✅" if activity["status"] == "success" else "⚡"
            st.text(f"{activity['time']} - {status_icon} {activity['agent']}: {activity['action']}")

    def _render_enhanced_overview(self, agent_status: Dict[str, Any]):
        """Render enhanced overview with real-time updates."""
        st.markdown("#### 🎯 System Overview")
        
        # System metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_agents = len(agent_status)
            st.metric("Total Agents", total_agents)
        
        with col2:
            healthy_agents = len([s for s in agent_status.values() 
                                if s.get('status') not in ['error', 'unknown']])
            st.metric("Healthy Agents", f"{healthy_agents}/{total_agents}")
        
        with col3:
            working_agents = len([s for s in agent_status.values() 
                                if s.get('status') == 'working'])
            st.metric("Active Agents", working_agents)
        
        with col4:
            avg_uptime = sum(s.get('uptime', 0) for s in agent_status.values()) / len(agent_status)
            st.metric("Avg Uptime", f"{avg_uptime:.0f}s")

    def _render_performance_dashboard(self, agent_status: Dict[str, Any]):
        """Render performance dashboard with charts."""
        st.markdown("#### 📈 Performance Dashboard")
        
        # Performance chart
        agent_names = []
        success_rates = []
        response_times = []
        
        for agent_id, status in agent_status.items():
            agent_names.append(status.get('name', agent_id))
            success_rates.append(status.get('metrics', {}).get('success_rate', 0) * 100)
            response_times.append(status.get('metrics', {}).get('average_response_time', 0))
        
        # Success rate chart
        fig_success = px.bar(
            x=agent_names,
            y=success_rates,
            title="Agent Success Rates",
            labels={'x': 'Agents', 'y': 'Success Rate (%)'}
        )
        st.plotly_chart(fig_success, use_container_width=True)

    def _render_diagnostics_panel(self, agent_status: Dict[str, Any]):
        """Render diagnostics and troubleshooting panel."""
        st.markdown("#### 🔧 System Diagnostics")
        
        # Health checks
        for agent_id, status in agent_status.items():
            agent_name = status.get('name', agent_id)
            agent_health = status.get('status', 'unknown')
            
            if agent_health == 'error':
                st.error(f"❌ {agent_name}: System error detected")
            elif agent_health == 'working':
                st.info(f"⚡ {agent_name}: Currently processing")
            else:
                st.success(f"✅ {agent_name}: Operating normally")

    def _render_fallback_status(self):
        """Render fallback status when enhanced monitoring fails."""
        st.info("🔄 Enhanced monitoring temporarily unavailable - showing basic status")
        
        # Basic status indicators
        st.markdown("**System Components:**")
        st.write("🟢 API Server: Online")
        st.write("🟡 WebSocket: Connecting...")
        st.write("🟢 Agent Pool: Ready")

    def _render_system_health_overview(self, health_status: Dict[str, Any], agent_status: Dict[str, Any]):
        """Render system health overview."""
        st.markdown("#### 🏥 System Health")
        
        # Health indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            api_health = health_status.get('status', 'unknown')
            health_color = '🟢' if api_health == 'healthy' else '🔴'
            st.markdown(f"**API Status:** {health_color} {api_health.title()}")
        
        with col2:
            agent_health_score = len([s for s in agent_status.values() 
                                    if s.get('status') != 'error']) / len(agent_status) * 100
            st.markdown(f"**Agent Health:** {agent_health_score:.0f}%")
        
        with col3:
            st.markdown("**Last Updated:** Just now")

    def _render_agent_grid(self, agent_status: Dict[str, Any]):
        """Render agent status in grid layout."""
        st.markdown("#### 🔲 Agent Grid View")
        
        # Create grid
        cols = st.columns(3)
        
        for i, (agent_id, status) in enumerate(agent_status.items()):
            with cols[i % 3]:
                self._render_compact_agent_card(agent_id, status)

    def _render_compact_agent_card(self, agent_id: str, status: Dict[str, Any]):
        """Render compact agent status card for grid view."""
        agent_name = status.get('name', agent_id.replace('_', ' ').title())
        agent_status = status.get('status', 'unknown')
        success_rate = status.get('metrics', {}).get('success_rate', 0)
        
        status_colors = {
            'idle': '#28a745',
            'working': '#ffc107', 
            'error': '#dc3545',
            'thinking': '#17a2b8'
        }
        
        color = status_colors.get(agent_status, '#6c757d')
        
        st.markdown(f"""
        <div style='
            border-left: 4px solid {color};
            padding: 0.5rem;
            margin: 0.25rem 0;
            background: #f8f9fa;
        '>
            <strong>{agent_name}</strong><br>
            Status: {agent_status.title()}<br>
            Success: {success_rate*100:.0f}%
        </div>
        """, unsafe_allow_html=True)
