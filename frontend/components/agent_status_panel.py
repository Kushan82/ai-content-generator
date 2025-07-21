import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any, List

class AgentStatusPanel:
    """
    Component for displaying comprehensive agent status and performance monitoring.
    """
    
    def __init__(self, api_client):
        self.api_client = api_client
    
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
    
    def render_agent_overview(self, agent_status: Dict[str, Any]):
        """Render overview cards for all agents."""
        
        # Create columns for agent cards
        cols = st.columns(len(agent_status))
        
        for i, (agent_id, status) in enumerate(agent_status.items()):
            with cols[i]:
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
        
        # Render card
        st.markdown(f"""
        <div class="agent-card">
            <h4>{status_emoji} {agent_name}</h4>
            <p><strong>Status:</strong> <span style="color: {status_color};">{agent_status.title()}</span></p>
            <p><strong>Uptime:</strong> {uptime:.1f}s</p>
            <p><strong>Success Rate:</strong> {success_rate*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Agent capabilities
        capabilities = status.get('capabilities', [])
        if capabilities:
            with st.expander(f"🔧 {agent_name} Capabilities"):
                for capability in capabilities[:3]:  # Show top 3
                    cap_name = capability.get('name', 'Unknown')
                    proficiency = capability.get('proficiency_level', 0)
                    st.progress(proficiency, text=f"{cap_name}: {proficiency*100:.0f}%")
    
    def render_agent_performance_metrics(self, agent_status: Dict[str, Any]):
        """Render agent performance metrics charts."""
        st.subheader("📊 Agent Performance")
        
        # Prepare data for visualization
        agent_names = []
        success_rates = []
        response_times = []
        task_counts = []
        
        for agent_id, status in agent_status.items():
            agent_names.append(status.get('name', agent_id.replace('_', ' ').title()))
            metrics = status.get('metrics', {})
            success_rates.append(metrics.get('success_rate', 0))
            response_times.append(metrics.get('average_response_time', 0))
            task_counts.append(metrics.get('total_tasks_completed', 0))
        
        # Success rate chart
        fig_success = go.Figure(data=[
            go.Bar(
                x=agent_names,
                y=success_rates,
                marker_color=['#28a745' if rate > 0.8 else '#ffc107' if rate > 0.6 else '#dc3545' 
                             for rate in success_rates],
                text=[f'{rate*100:.1f}%' for rate in success_rates],
                textposition='auto',
            )
        ])
        
        fig_success.update_layout(
            title="Agent Success Rates",
            xaxis_title="Agent",
            yaxis_title="Success Rate",
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig_success, use_container_width=True)
        
        # Response time chart
        fig_time = go.Figure(data=[
            go.Scatter(
                x=agent_names,
                y=response_times,
                mode='markers+lines',
                marker=dict(size=10, color='#007bff'),
                line=dict(color='#007bff')
            )
        ])
        
        fig_time.update_layout(
            title="Average Response Times",
            xaxis_title="Agent",
            yaxis_title="Response Time (seconds)",
            height=300
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
    
    def render_agent_activity_timeline(self):
        """Render agent activity timeline."""
        st.subheader("⏱️ Recent Activity")
        
        # Mock activity data (in production, this would come from real logs)
        activities = [
            {
                'time': '14:35:22',
                'agent': 'Creative Generator',
                'activity': 'Completed content generation task',
                'status': 'success'
            },
            {
                'time': '14:34:15',
                'agent': 'Quality Assurance',
                'activity': 'Started quality review process',
                'status': 'working'
            },
            {
                'time': '14:33:08',
                'agent': 'Content Strategist',
                'activity': 'Generated strategic recommendations',
                'status': 'success'
            },
            {
                'time': '14:32:01',
                'agent': 'Persona Research',
                'activity': 'Completed demographic analysis',
                'status': 'success'
            }
        ]
        
        for activity in activities:
            status_icon = {
                'success': '✅',
                'working': '⚡',
                'error': '❌'
            }.get(activity['status'], '⚪')
            
            st.markdown(f"""
            <div class="workflow-step">
                <strong>{activity['time']}</strong> {status_icon} 
                <strong>{activity['agent']}:</strong> {activity['activity']}
            </div>
            """, unsafe_allow_html=True)
