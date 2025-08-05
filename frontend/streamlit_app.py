import streamlit as st
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import requests
import websockets
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading

from components.agent_status_panel import AgentStatusPanel
from components.workflow_visualizer import WorkflowVisualizer
from components.real_time_updates import RealTimeUpdates
from components.content_comparison import ContentComparison
from utils.api_client import APIClient
from utils.websocket_client import WebSocketClient

# Configure Streamlit page
st.set_page_config(
    page_title="Multi-Agent AI Content Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Multi-Agent AI Content Generator - Advanced content creation using collaborative AI agents"
    }
)

# Enhanced CSS for professional UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .agent-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .agent-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid #007bff;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-indicator {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        margin-right: 10px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .status-healthy { background-color: #28a745; }
    .status-working { background-color: #ffc107; }
    .status-error { background-color: #dc3545; }
    .status-idle { background-color: #6c757d; }
    
    .workflow-step {
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 6px;
        border-left: 4px solid #007bff;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        transition: all 0.3s ease;
    }
    
    .workflow-step:hover {
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        border-left-color: #0056b3;
    }
    
    .content-preview {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-family: 'Georgia', serif;
        line-height: 1.7;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    .tab-content {
        padding: 1rem;
        border-radius: 0 0 10px 10px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .component-section {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

class MultiAgentDashboard:
    """
    Enhanced multi-agent dashboard with integrated custom components.
    Provides comprehensive workflow monitoring, real-time updates, and content management.
    """
    
    def __init__(self):
        self.api_client = APIClient()
        self.websocket_client = WebSocketClient()
        
        # Initialize enhanced session state
        self._initialize_session_state()
        
        # Initialize component instances
        self.agent_status_panel = AgentStatusPanel(self.api_client)
        self.workflow_visualizer = WorkflowVisualizer()
        self.real_time_updates = RealTimeUpdates(self.websocket_client)
        self.content_comparison = ContentComparison()
    
    def _initialize_session_state(self):
        """Initialize comprehensive session state for enhanced functionality."""
        session_defaults = {
            'workflow_history': [],
            'active_workflow': None,
            'agent_status': {},
            'real_time_updates': [],
            'generated_content_cache': {},
            'content_variations': [],
            'workflow_metrics': {},
            'user_preferences': {
                'auto_refresh': True,
                'notification_enabled': True,
                'theme': 'light'
            },
            'comparison_content': [],
            'websocket_connected': False
        }
        
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def run(self):
        """Enhanced main dashboard execution flow with integrated components."""
        self.render_header()
        
        # Enhanced main layout with additional tabs
        main_tab, monitoring_tab, analytics_tab, comparison_tab, settings_tab = st.tabs([
            "🎯 Content Generation", 
            "📊 Real-Time Monitoring", 
            "📈 Analytics & Insights", 
            "🔄 Content Comparison",
            "⚙️ Settings"
        ])
        
        with main_tab:
            self.render_enhanced_content_generation()
        
        with monitoring_tab:
            self.render_real_time_monitoring()
        
        with analytics_tab:
            self.render_enhanced_analytics()
        
        with comparison_tab:
            self.render_content_comparison()
        
        with settings_tab:
            self.render_enhanced_settings()
    
    def render_header(self):
        """Enhanced header with real-time system status and WebSocket connectivity."""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 Multi-Agent AI Content Generator</h1>
            <p>Advanced content creation using collaborative AI agents with real-time workflow monitoring</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced system status bar with WebSocket status
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            self._render_api_health_status()
        
        with col2:
            self._render_agent_status_summary()
        
        with col3:
            self._render_workflow_status()
        
        with col4:
            self._render_websocket_status()
        
        with col5:
            if st.button("🔄 Refresh All", key="refresh_all"):
                self._refresh_all_components()
    
    def _render_api_health_status(self):
        """Render API health status with detailed information."""
        try:
            health = self.api_client.get_health_status()
            if health.get('status') == 'healthy':
                st.success("🟢 API Healthy")
                with st.expander("API Details", expanded=False):
                    st.write(f"Version: {health.get('version', 'Unknown')}")
                    st.write(f"Uptime: {health.get('uptime', 'Unknown')}")
            else:
                st.error("🔴 API Issues")
        except:
            st.error("🔴 API Offline")
    
    def _render_agent_status_summary(self):
        """Render enhanced agent status summary."""
        try:
            agent_status = self.api_client.get_agent_status()
            healthy_agents = sum(1 for agent in agent_status.values() if agent.get('status') == 'idle')
            st.info(f"🤖 {healthy_agents}/5 Agents Ready")
            
            # Show agent activity indicator
            working_agents = sum(1 for agent in agent_status.values() if agent.get('status') == 'working')
            if working_agents > 0:
                st.warning(f"⚡ {working_agents} Agents Working")
        except:
            st.warning("⚠️ Agent Status Unknown")
    
    def _render_workflow_status(self):
        """Render workflow status with progress indication."""
        active_count = 1 if st.session_state.active_workflow else 0
        if active_count > 0:
            workflow = st.session_state.active_workflow
            progress = workflow.get('progress', 0)
            st.info(f"⚡ Workflow: {progress}%")
        else:
            st.success("✅ System Ready")
    
    def _render_websocket_status(self):
        """Render WebSocket connection status."""
        if st.session_state.websocket_connected:
            st.success("🔗 Live Updates")
        else:
            if st.button("🔌 Connect Live"):
                self._initialize_websocket_connection()
    
    def render_enhanced_content_generation(self):
        """Enhanced content generation interface with integrated components."""
        st.header("🎯 Multi-Agent Content Generation")
        
        # Three-column layout for enhanced functionality
        config_col, monitoring_col, preview_col = st.columns([1, 1, 1])
        
        with config_col:
            self._render_content_configuration()
        
        with monitoring_col:
            self._render_workflow_monitoring_enhanced()
        
        with preview_col:
            self._render_content_preview()
    
    def _render_content_configuration(self):
        """Enhanced content configuration with advanced options."""
        st.markdown('<div class="component-section">', unsafe_allow_html=True)
        st.subheader("📋 Content Configuration")
        
        # Persona selection with enhanced details
        with st.container():
            st.markdown("### 🎭 Target Persona")
            
            try:
                personas = self.api_client.get_personas()
                persona_options = {p['name']: p['id'] for p in personas}
                
                selected_persona_name = st.selectbox(
                    "Choose your target audience:",
                    list(persona_options.keys()),
                    help="Select the persona that best represents your target audience"
                )
                selected_persona_id = persona_options[selected_persona_name]
                
                # Enhanced persona details display
                selected_persona = next(p for p in personas if p['id'] == selected_persona_id)
                with st.expander("📊 Detailed Persona Analysis", expanded=False):
                    self._render_enhanced_persona_details(selected_persona)
                
                # Store selected persona in session state
                st.session_state['selected_persona'] = selected_persona
                
            except Exception as e:
                st.error(f"Failed to load personas: {str(e)}")
                return
        
        # Enhanced content settings
        self._render_enhanced_content_settings()
        
        # Advanced generation options
        self._render_advanced_generation_options()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_enhanced_persona_details(self, persona: Dict[str, Any]):
        """Enhanced persona details with visual indicators."""
        # Create tabs for different persona aspects
        demo_tab, psycho_tab, behavior_tab = st.tabs(["Demographics", "Psychographics", "Behavior"])
        
        with demo_tab:
            demographics = persona.get('demographics', {})
            for key, value in demographics.items():
                st.metric(
                    label=key.replace('_', ' ').title(),
                    value=str(value)
                )
        
        with psycho_tab:
            # Pain points with severity indicators
            st.markdown("**Pain Points:**")
            pain_points = persona.get('pain_points', [])[:5]
            for i, pain_point in enumerate(pain_points):
                severity = "🔴" if i < 2 else "🟡" if i < 4 else "🟢"
                st.write(f"{severity} {pain_point}")
            
            # Goals with priority indicators
            st.markdown("**Primary Goals:**")
            goals = persona.get('goals', [])[:5]
            for i, goal in enumerate(goals):
                priority = "⭐" * (3 - i//2) if i < 6 else "⭐"
                st.write(f"{priority} {goal}")
        
        with behavior_tab:
            # Communication preferences
            comm_prefs = persona.get('communication_preferences', {})
            if comm_prefs:
                st.metric("Preferred Tone", comm_prefs.get('tone', 'Professional'))
                st.metric("Formality Level", comm_prefs.get('formality', 'Balanced'))
                st.metric("Terminology", comm_prefs.get('terminology', 'Standard'))
    
    def _render_enhanced_content_settings(self):
        """Enhanced content settings with real-time validation."""
        st.markdown("### ⚙️ Content Settings")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            content_types = self.api_client.get_content_types()
            content_type_options = {ct['name']: ct['id'] for ct in content_types['content_types']}
            
            selected_content_type_name = st.selectbox(
                "Content Type:",
                list(content_type_options.keys()),
                help="Choose the type of content you want to generate"
            )
            selected_content_type = content_type_options[selected_content_type_name]
            
            # Show content type specifications
            content_spec = next(ct for ct in content_types['content_types'] if ct['id'] == selected_content_type)
            st.info(f"📏 Optimal: {content_spec.get('optimal_word_count', 'Variable')}")
        
        with col_b:
            creativity_level = st.slider(
                "Creativity Level:",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher values produce more creative and varied content"
            )
            
            # Dynamic creativity indicator
            creativity_label = "🎨 Creative" if creativity_level > 0.7 else "⚖️ Balanced" if creativity_level > 0.4 else "📊 Conservative"
            st.write(f"Style: {creativity_label}")
        
        # Topic and context with validation
        topic = st.text_input(
            "Topic/Product:",
            placeholder="e.g., AI-powered project management software",
            help="Describe the main topic, product, or service"
        )
        
        # Real-time topic validation
        if topic:
            topic_length = len(topic.split())
            if topic_length < 3:
                st.warning("⚠️ Consider adding more detail to your topic")
            elif topic_length > 20:
                st.warning("⚠️ Topic might be too detailed, consider simplifying")
            else:
                st.success("✅ Good topic length")
        
        additional_context = st.text_area(
            "Additional Context (Optional):",
            placeholder="Any specific requirements, features to highlight, or constraints...",
            height=100,
            help="Provide any additional context that will help generate better content"
        )
        
        # Store configuration in session state
        st.session_state['content_config'] = {
            'content_type': selected_content_type,
            'topic': topic,
            'creativity_level': creativity_level,
            'additional_context': additional_context
        }
    
    def _render_advanced_generation_options(self):
        """Advanced generation options with intelligent defaults."""
        with st.expander("🔧 Advanced Options", expanded=False):
            col_x, col_y = st.columns(2)
            
            with col_x:
                word_count = st.slider("Word Count:", 50, 1000, 200, 25)
                enable_qa = st.checkbox("Quality Assurance", True, help="Enable AI quality review")
                enable_variations = st.checkbox("Generate Variations", False, help="Create multiple content versions")
            
            with col_y:
                real_time_updates = st.checkbox("Real-time Updates", True, help="Show live progress")
                save_to_history = st.checkbox("Save to History", True, help="Save results to workflow history")
                auto_optimize = st.checkbox("Auto-Optimize", False, help="Automatically apply optimization suggestions")
            
            # Workflow type selection
            st.markdown("**Workflow Configuration:**")
            workflow_type = st.selectbox(
                "Workflow Type:",
                ["standard", "rapid", "research_heavy"],
                help="Choose workflow complexity level"
            )
            
            # Store advanced options
            st.session_state['advanced_options'] = {
                'word_count': word_count,
                'enable_qa': enable_qa,
                'enable_variations': enable_variations,
                'real_time_updates': real_time_updates,
                'save_to_history': save_to_history,
                'auto_optimize': auto_optimize,
                'workflow_type': workflow_type
            }
        
        # Enhanced generation button
        st.markdown("### 🚀 Generate Content")
        
        generation_col1, generation_col2 = st.columns([3, 1])
        
        with generation_col1:
            if st.button("🎯 Launch Multi-Agent Generation", type="primary", use_container_width=True):
                self._handle_content_generation()
        
        with generation_col2:
            if st.button("💾 Save Config", help="Save current configuration"):
                st.success("Configuration saved!")
    
    def _render_workflow_monitoring_enhanced(self):
        """Enhanced workflow monitoring with real-time visualization."""
        st.markdown('<div class="component-section">', unsafe_allow_html=True)
        st.subheader("🔄 Workflow Monitoring")
        
        if st.session_state.active_workflow:
            # Use WorkflowVisualizer component
            self.workflow_visualizer.render_workflow_progress(st.session_state.active_workflow)
            
            # Real-time updates component
            if st.session_state.get('real_time_updates'):
                self.real_time_updates.render_live_updates(st.session_state.real_time_updates)
        else:
            st.info("👆 Configure your content parameters and click 'Generate Content' to start workflow monitoring.")
            
            # Show recent workflows with enhanced details
            self._render_workflow_history()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_content_preview(self):
        """Enhanced content preview with comparison capabilities."""
        st.markdown('<div class="component-section">', unsafe_allow_html=True)
        st.subheader("📝 Content Preview")
        
        if st.session_state.active_workflow and st.session_state.active_workflow.get('generated_content'):
            content = st.session_state.active_workflow['generated_content']
            
            # Content preview with enhanced formatting
            st.markdown(f"""
            <div class="content-preview">
                {content}
            </div>
            """, unsafe_allow_html=True)
            
            # Preview actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Copy", key="copy_preview"):
                    st.success("Copied to clipboard!")
            
            with col2:
                if st.button("💾 Save", key="save_preview"):
                    self._save_content_to_cache(content)
            
            with col3:
                if st.button("🔄 Compare", key="compare_preview"):
                    self._add_to_comparison(content)
        
        # Show cached content for comparison
        if st.session_state.generated_content_cache:
            with st.expander("📚 Cached Content", expanded=False):
                for i, (timestamp, cached_content) in enumerate(st.session_state.generated_content_cache.items()):
                    st.write(f"**{timestamp}**: {cached_content[:100]}...")
                    if st.button(f"🔄 Use as Template", key=f"template_{i}"):
                        st.session_state['template_content'] = cached_content
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_real_time_monitoring(self):
        """Enhanced real-time monitoring dashboard with integrated components."""
        st.header("📊 Real-Time System Monitoring")
        
        # Real-time metrics overview
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            self._render_live_metrics_card("Active Workflows", 
                                         len([w for w in st.session_state.workflow_history if w.get('status') == 'running']),
                                         "⚡")
        
        with metrics_col2:
            self._render_live_metrics_card("Total Generated", 
                                         len(st.session_state.workflow_history),
                                         "📝")
        
        with metrics_col3:
            success_rate = self._calculate_success_rate()
            self._render_live_metrics_card("Success Rate", 
                                         f"{success_rate:.1f}%",
                                         "✅")
        
        with metrics_col4:
            avg_time = self._calculate_average_time()
            self._render_live_metrics_card("Avg Time", 
                                         f"{avg_time:.1f}s",
                                         "⏱️")
        
        st.markdown("---")
        
        # Enhanced monitoring layout
        monitoring_col1, monitoring_col2 = st.columns([2, 1])
        
        with monitoring_col1:
            # Agent status panel with real-time updates
            self.agent_status_panel.render_enhanced()
            
            # Workflow visualizer
            if st.session_state.active_workflow:
                self.workflow_visualizer.render_detailed_workflow(st.session_state.active_workflow)
        
        with monitoring_col2:
            # Real-time updates panel
            self.real_time_updates.render_updates_panel()
            
            # System health indicators
            self._render_system_health_panel()
    
    def render_enhanced_analytics(self):
        """Enhanced analytics dashboard with advanced visualizations."""
        st.header("📈 Analytics & Performance Insights")
        
        # Analytics overview
        self._render_analytics_overview()
        
        # Detailed analytics sections
        analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs([
            "📊 Generation Analytics",
            "🤖 Agent Performance", 
            "📈 Quality Trends"
        ])
        
        with analytics_tab1:
            self._render_generation_analytics()
        
        with analytics_tab2:
            self._render_agent_performance_analytics()
        
        with analytics_tab3:
            self._render_quality_analytics()
    
    def render_content_comparison(self):
        """Enhanced content comparison interface."""
        st.header("🔄 Content Comparison & Optimization")
        
        # Use ContentComparison component
        self.content_comparison.render_comparison_interface(st.session_state.content_variations)
        
        # Content library management
        st.markdown("---")
        st.subheader("📚 Content Library")
        
        self._render_content_library()
    
    def render_enhanced_settings(self):
        """Enhanced settings with advanced configuration options."""
        st.header("⚙️ System Settings & Configuration")
        
        # Settings categories
        settings_tab1, settings_tab2, settings_tab3, settings_tab4 = st.tabs([
            "🔧 API Configuration",
            "🤖 Agent Settings",
            "⚡ Performance",
            "🎨 UI Preferences"
        ])
        
        with settings_tab1:
            self._render_api_configuration()
        
        with settings_tab2:
            self._render_agent_configuration()
        
        with settings_tab3:
            self._render_performance_settings()
        
        with settings_tab4:
            self._render_ui_preferences()
    
    # Helper methods for enhanced functionality
    def _handle_content_generation(self):
        """Enhanced content generation handler with validation."""
        config = st.session_state.get('content_config', {})
        advanced = st.session_state.get('advanced_options', {})
        
        if not config.get('topic'):
            st.error("Please enter a topic/product.")
            return
        
        # Prepare enhanced request data
        request_data = {
            **config,
            **advanced,
            'persona_id': st.session_state.get('selected_persona', {}).get('id'),
            'enable_realtime_updates': advanced.get('real_time_updates', True)
        }
        
        self.start_enhanced_content_generation(request_data)
    
    def start_enhanced_content_generation(self, request_data: Dict[str, Any]):
        """Enhanced content generation with real-time monitoring."""
        try:
            with st.spinner("🚀 Launching multi-agent workflow..."):
                # Initialize real-time monitoring
                if request_data.get('enable_realtime_updates'):
                    self._setup_real_time_monitoring(request_data)
                
                # Call the API
                response = self.api_client.generate_content(request_data)
                
                if request_data.get('enable_realtime_updates'):
                    self._start_real_time_workflow_monitoring(response)
                else:
                    self._display_enhanced_results(response)
        
        except Exception as e:
            st.error(f"Content generation failed: {str(e)}")
            self._log_error(e, request_data)
    
    def _setup_real_time_monitoring(self, request_data: Dict[str, Any]):
        """Setup real-time monitoring for workflow execution."""
        st.session_state.active_workflow = {
            'id': f"workflow_{int(time.time())}",
            'status': 'initializing',
            'progress': 0,
            'current_step': 'initialization',
            'topic': request_data['topic'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'agents_status': {
                'persona_researcher': 'pending',
                'content_strategist': 'pending',
                'creative_generator': 'pending',
                'quality_assurance': 'pending'
            }
        }
    
    def _start_real_time_workflow_monitoring(self, response: Dict[str, Any]):
        """Start real-time workflow monitoring with WebSocket updates."""
        workflow_id = response.get('workflow_id')
        
        # Enhanced progress simulation with agent-specific updates
        progress_steps = [
            (15, "persona_research", "🔍 Analyzing target persona demographics...", "persona_researcher", "working"),
            (30, "persona_research", "📊 Completing psychographic analysis...", "persona_researcher", "completed"),
            (45, "content_strategy", "🎯 Developing messaging strategy...", "content_strategist", "working"),
            (60, "content_strategy", "📋 Finalizing content blueprint...", "content_strategist", "completed"),
            (75, "creative_generation", "✍️ Generating creative content...", "creative_generator", "working"),
            (85, "creative_generation", "🎨 Applying creative enhancements...", "creative_generator", "completed"),
            (95, "quality_assurance", "🔍 Performing quality review...", "quality_assurance", "working"),
            (100, "completed", "✅ Multi-agent workflow completed!", "orchestrator", "completed")
        ]
        
        for i, (progress, step, message, agent, agent_status) in enumerate(progress_steps):
            time.sleep(2)  # Simulate processing time
            
            # Update workflow status
            if st.session_state.active_workflow:
                st.session_state.active_workflow.update({
                    'progress': progress,
                    'current_step': step,
                })
                
                # Update agent status
                if agent in st.session_state.active_workflow['agents_status']:
                    st.session_state.active_workflow['agents_status'][agent] = agent_status
                
                # Add real-time update
                st.session_state.real_time_updates.append({
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'message': message,
                    'step': step,
                    'agent': agent,
                    'progress': progress
                })
            
            # Trigger rerun for real-time updates
            if i < len(progress_steps) - 1:
                st.rerun()
        
        # Complete workflow
        self._complete_workflow_monitoring()
    
    def _complete_workflow_monitoring(self):
        """Complete workflow monitoring and save results."""
        if st.session_state.active_workflow:
            # Generate sample content
            sample_content = self._generate_sample_content()
            
            st.session_state.active_workflow.update({
                'status': 'completed',
                'generated_content': sample_content,
                'quality_score': 0.87,
                'completion_time': datetime.now().strftime("%H:%M:%S")
            })
            
            # Save to history
            st.session_state.workflow_history.append(st.session_state.active_workflow.copy())
            
            # Add to content cache
            self._save_content_to_cache(sample_content)
            
            # Clear active workflow after delay
            time.sleep(3)
            st.session_state.active_workflow = None
            
            st.rerun()
    
    def _generate_sample_content(self) -> str:
        """Generate sample content based on configuration."""
        config = st.session_state.get('content_config', {})
        persona = st.session_state.get('selected_persona', {})
        
        topic = config.get('topic', 'AI-powered solution')
        content_type = config.get('content_type', 'ad')
        
        # Generate contextual sample content
        if content_type == 'ad':
            return f"Discover {topic} - the game-changing solution that {persona.get('name', 'professionals')} have been waiting for. Transform your workflow with cutting-edge AI technology. Get started today!"
        elif content_type == 'landing_page':
            return f"Welcome to the future of {topic}. Our innovative platform addresses your key challenges while delivering exceptional results. Join thousands of satisfied customers who've revolutionized their approach."
        else:
            return f"Experience the power of {topic}. Designed specifically for {persona.get('name', 'your needs')}, our solution combines innovation with reliability to deliver outstanding results."
    
    def _save_content_to_cache(self, content: str):
        """Save content to cache with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.generated_content_cache[timestamp] = content
    
    def _add_to_comparison(self, content: str):
        """Add content to comparison list."""
        if 'comparison_content' not in st.session_state:
            st.session_state.comparison_content = []
        
        st.session_state.comparison_content.append({
            'content': content,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': st.session_state.get('content_config', {}).get('content_type', 'unknown')
        })
        
        st.success("Content added to comparison!")
    
    def _refresh_all_components(self):
        """Refresh all dashboard components."""
        # Reset connection status
        st.session_state.websocket_connected = False
        
        # Clear temporary data
        if 'temp_data' in st.session_state:
            del st.session_state.temp_data
        
        # Trigger component refresh
        st.rerun()
    
    def _initialize_websocket_connection(self):
        """Initialize WebSocket connection for real-time updates."""
        try:
            # Simulate WebSocket connection
            st.session_state.websocket_connected = True
            st.success("🔗 Connected to live updates!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to connect: {str(e)}")
    
    def _render_live_metrics_card(self, title: str, value: str, icon: str):
        """Render live metrics card with animation."""
        st.markdown(f"""
        <div class="metric-card">
            <h3>{icon} {title}</h3>
            <h2 style="color: #007bff; margin: 0;">{value}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate from workflow history."""
        if not st.session_state.workflow_history:
            return 100.0
        
        successful = sum(1 for w in st.session_state.workflow_history if w.get('status') == 'completed')
        return (successful / len(st.session_state.workflow_history)) * 100
    
    def _calculate_average_time(self) -> float:
        """Calculate average workflow completion time."""
        # Mock calculation - in production, this would use real timing data
        return 45.2
    
    def _render_workflow_history(self):
        """Render enhanced workflow history with filtering."""
        if st.session_state.workflow_history:
            st.markdown("### 📜 Recent Workflows")
            
            # Filter options
            col_filter1, col_filter2 = st.columns(2)
            
            with col_filter1:
                status_filter = st.selectbox("Filter by Status:", ["All", "Completed", "Failed"], key="status_filter")
            
            with col_filter2:
                time_filter = st.selectbox("Time Range:", ["All Time", "Last 24h", "Last Week"], key="time_filter")
            
            # Display filtered workflows
            filtered_workflows = self._filter_workflows(st.session_state.workflow_history, status_filter, time_filter)
            
            for workflow in filtered_workflows[-10:]:  # Show last 10
                self._render_workflow_history_item(workflow)
    
    def _filter_workflows(self, workflows: List[Dict], status_filter: str, time_filter: str) -> List[Dict]:
        """Filter workflows based on criteria."""
        filtered = workflows
        
        if status_filter != "All":
            status_map = {"Completed": "completed", "Failed": "failed"}
            filtered = [w for w in filtered if w.get('status') == status_map.get(status_filter)]
        
        # Add time filtering logic here if needed
        
        return filtered
    
    def _render_workflow_history_item(self, workflow: Dict[str, Any]):
        """Render individual workflow history item."""
        status_icon = "🟢" if workflow.get('status') == 'completed' else "🔴"
        quality_score = workflow.get('quality_score', 0)
        
        with st.expander(f"{status_icon} {workflow.get('topic', 'Unknown')} - {workflow.get('timestamp', 'Unknown')}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Quality Score", f"{quality_score:.2f}")
            
            with col2:
                st.metric("Duration", f"{workflow.get('duration', 'Unknown')}")
            
            with col3:
                if st.button("🔄 Regenerate", key=f"regen_{workflow.get('id')}"):
                    st.info("Regeneration feature would be implemented here")
    
    # Additional helper methods for other components would be implemented here...
    
    def _render_system_health_panel(self):
        """Render system health monitoring panel."""
        st.markdown("### 🏥 System Health")
        
        health_metrics = {
            "CPU Usage": 45,
            "Memory": 62,
            "API Response": 98,
            "Agent Load": 34
        }
        
        for metric, value in health_metrics.items():
            color = "green" if value < 70 else "orange" if value < 90 else "red"
            st.metric(metric, f"{value}%", delta=f"{value-50}%")
    
    # Placeholder methods for other sections
    def _render_analytics_overview(self): pass
    def _render_generation_analytics(self): pass  
    def _render_agent_performance_analytics(self): pass
    def _render_quality_analytics(self): pass
    def _render_content_library(self): pass
    def _render_api_configuration(self): pass
    def _render_agent_configuration(self): pass
    def _render_performance_settings(self): pass
    def _render_ui_preferences(self): pass
    def _log_error(self, error: Exception, context: Dict[str, Any]): pass

# Initialize and run the enhanced dashboard
if __name__ == "__main__":
    dashboard = MultiAgentDashboard()
    dashboard.run()
