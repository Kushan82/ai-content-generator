import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
from datetime import datetime

class WorkflowVisualizer:
    """Component for visualizing workflow execution steps."""
    
    def __init__(self):
        pass
    
    def render(self, workflow_data: Dict[str, Any] = None):
        """Render workflow visualization."""
        if not workflow_data:
            st.info("No active workflow to visualize")
            return
        
        # Simple workflow visualization
        st.subheader("🔄 Workflow Execution")
        
        steps = ["Persona Research", "Content Strategy", "Creative Generation", "Quality Assurance"]
        status = ["completed", "completed", "working", "pending"]
        
        for i, (step, stat) in enumerate(zip(steps, status)):
            if stat == "completed":
                st.success(f"✅ {step}")
            elif stat == "working":
                st.info(f"⚡ {step} (In Progress)")
            else:
                st.warning(f"⏳ {step} (Pending)")
    
    def render_workflow_progress(self, workflow_data: Dict[str, Any]):
        """Render workflow progress with enhanced visualization."""
        if not workflow_data:
            st.info("No active workflow to display")
            return
        
        # Progress overview
        progress = workflow_data.get('progress', 0)
        current_step = workflow_data.get('current_step', 'unknown')
        
        st.subheader("🔄 Workflow Progress")
        
        # Progress bar
        progress_col1, progress_col2 = st.columns([3, 1])
        
        with progress_col1:
            st.progress(progress / 100, text=f"Overall Progress: {progress}%")
        
        with progress_col2:
            st.metric("Current Step", current_step.replace('_', ' ').title())
        
        # Agent status indicators
        agents_status = workflow_data.get('agents_status', {})
        if agents_status:
            st.markdown("### 🤖 Agent Status")
            
            cols = st.columns(len(agents_status))
            
            for i, (agent_id, status) in enumerate(agents_status.items()):
                with cols[i]:
                    self._render_agent_status_indicator(agent_id, status)
        
        # Step-by-step progress
        self._render_step_progress(workflow_data)
    
    def render_detailed_workflow(self, workflow_data: Dict[str, Any]):
        """Render detailed workflow visualization with timeline."""
        if not workflow_data:
            st.info("No workflow data available")
            return
        
        st.subheader("📊 Detailed Workflow Analysis")
        
        # Workflow timeline
        self._render_workflow_timeline(workflow_data)
        
        # Performance metrics
        self._render_workflow_metrics(workflow_data)
    
    def render_enhanced(self):
        """Render enhanced workflow visualization."""
        st.subheader("🔄 Enhanced Workflow Monitoring")
        
        # Mock data for demonstration
        workflow_steps = [
            {"name": "Persona Research", "status": "completed", "duration": "15s", "quality": 0.92},
            {"name": "Content Strategy", "status": "completed", "duration": "12s", "quality": 0.89},
            {"name": "Creative Generation", "status": "working", "duration": "8s", "quality": None},
            {"name": "Quality Assurance", "status": "pending", "duration": None, "quality": None}
        ]
        
        # Enhanced step visualization
        for step in workflow_steps:
            self._render_enhanced_step(step)
    
    def _render_agent_status_indicator(self, agent_id: str, status: str):
        """Render individual agent status indicator."""
        agent_name = agent_id.replace('_', ' ').title()
        
        status_colors = {
            'pending': '#6c757d',
            'working': '#ffc107', 
            'completed': '#28a745',
            'error': '#dc3545'
        }
        
        status_icons = {
            'pending': '⏳',
            'working': '⚡',
            'completed': '✅',
            'error': '❌'
        }
        
        color = status_colors.get(status, '#6c757d')
        icon = status_icons.get(status, '⚪')
        
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; border-radius: 8px; background: #f8f9fa; border-left: 4px solid {color};">
            <div style="font-size: 1.5em; margin-bottom: 0.2rem;">{icon}</div>
            <div style="font-weight: bold; font-size: 0.9em;">{agent_name}</div>
            <div style="color: {color}; font-size: 0.8em; text-transform: capitalize;">{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_step_progress(self, workflow_data: Dict[str, Any]):
        """Render step-by-step progress visualization."""
        st.markdown("### 📋 Process Steps")
        
        steps = [
            {"name": "Persona Research", "key": "persona_research"},
            {"name": "Content Strategy", "key": "content_strategy"},
            {"name": "Creative Generation", "key": "creative_generation"},
            {"name": "Quality Assurance", "key": "quality_assurance"}
        ]
        
        current_step = workflow_data.get('current_step', '')
        progress = workflow_data.get('progress', 0)
        
        for i, step in enumerate(steps):
            step_progress = min(100, max(0, (progress - i * 25)))
            
            if step['key'] == current_step or step_progress > 0:
                if step_progress >= 100:
                    st.success(f"✅ {step['name']} - Complete")
                elif step_progress > 0:
                    st.info(f"⚡ {step['name']} - In Progress ({step_progress:.0f}%)")
                    st.progress(step_progress / 100)
                else:
                    st.warning(f"⏳ {step['name']} - Pending")
            else:
                st.warning(f"⏳ {step['name']} - Pending")
    
    def _render_workflow_timeline(self, workflow_data: Dict[str, Any]):
        """Render workflow execution timeline."""
        st.markdown("### ⏱️ Execution Timeline")
        
        # Mock timeline data
        timeline_events = [
            {"time": "14:30:00", "event": "Workflow Started", "agent": "Orchestrator"},
            {"time": "14:30:15", "event": "Persona Analysis Initiated", "agent": "Persona Researcher"},
            {"time": "14:30:45", "event": "Demographics Analysis Complete", "agent": "Persona Researcher"},
            {"time": "14:31:00", "event": "Strategy Planning Started", "agent": "Content Strategist"},
            {"time": "14:31:30", "event": "Messaging Framework Developed", "agent": "Content Strategist"}
        ]
        
        for event in timeline_events:
            st.markdown(f"""
            <div style="padding: 0.5rem; margin: 0.3rem 0; border-left: 3px solid #007bff; background: #f8f9fa; border-radius: 0 5px 5px 0;">
                <strong>{event['time']}</strong> - {event['event']} 
                <br><small style="color: #6c757d;">Agent: {event['agent']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_workflow_metrics(self, workflow_data: Dict[str, Any]):
        """Render workflow performance metrics."""
        st.markdown("### 📈 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Execution Time", f"{workflow_data.get('elapsed_time', 45)}s")
        
        with col2:
            st.metric("Quality Score", f"{workflow_data.get('quality_score', 0.87):.2f}")
        
        with col3:
            st.metric("Efficiency", f"{workflow_data.get('efficiency', 92)}%")
        
        with col4:
            st.metric("Agent Utilization", f"{workflow_data.get('agent_utilization', 78)}%")
    
    def _render_enhanced_step(self, step: Dict[str, Any]):
        """Render enhanced step visualization."""
        name = step['name']
        status = step['status']
        duration = step.get('duration', 'N/A')
        quality = step.get('quality')
        
        # Status styling
        if status == 'completed':
            status_color = '#28a745'
            status_icon = '✅'
        elif status == 'working':
            status_color = '#ffc107'
            status_icon = '⚡'
        elif status == 'error':
            status_color = '#dc3545'
            status_icon = '❌'
        else:
            status_color = '#6c757d'
            status_icon = '⏳'
        
        # Create expandable step card
        with st.expander(f"{status_icon} {name} - {status.title()}", expanded=(status == 'working')):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Duration", duration)
            
            with col2:
                if quality is not None:
                    st.metric("Quality Score", f"{quality:.2f}")
                else:
                    st.metric("Quality Score", "Pending")
            
            with col3:
                st.metric("Status", status.title())
            
            # Progress bar for working steps
            if status == 'working':
                st.progress(0.6, text="Processing...")
    
    def create_workflow_diagram(self, workflow_data: Dict[str, Any]):
        """Create an interactive workflow diagram using Plotly."""
        if not workflow_data:
            return None
        
        # Create nodes for workflow steps
        steps = ['Start', 'Persona Research', 'Content Strategy', 'Creative Generation', 'Quality Assurance', 'Complete']
        x_positions = [0, 1, 2, 3, 4, 5]
        y_positions = [0, 0, 0, 0, 0, 0]
        
        # Determine colors based on progress
        progress = workflow_data.get('progress', 0)
        colors = []
        for i, step in enumerate(steps):
            if i == 0 or i == len(steps) - 1:  # Start/End nodes
                colors.append('#007bff')
            elif progress >= (i - 1) * 25:
                colors.append('#28a745')  # Completed
            else:
                colors.append('#6c757d')  # Pending
        
        # Create the diagram
        fig = go.Figure()
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=y_positions,
            mode='markers+text',
            marker=dict(size=50, color=colors),
            text=steps,
            textposition='bottom center',
            textfont=dict(size=10),
            name='Workflow Steps'
        ))
        
        # Add connections
        for i in range(len(x_positions) - 1):
            fig.add_trace(go.Scatter(
                x=[x_positions[i], x_positions[i + 1]],
                y=[y_positions[i], y_positions[i + 1]],
                mode='lines',
                line=dict(color='#dee2e6', width=2),
                showlegend=False
            ))
        
        fig.update_layout(
            title="Workflow Execution Diagram",
            showlegend=False,
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            height=200,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig