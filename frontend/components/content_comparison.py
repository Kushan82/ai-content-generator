import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime
import difflib

class ContentComparison:
    """Enhanced component for comparing different content variations."""
    
    def __init__(self):
        self.comparison_cache = {}
        self.max_variations = 10

    def render(self, content_variations: List[str] = None):
        """Render basic content comparison interface."""
        st.subheader("📊 Content Variations")
        
        if not content_variations:
            st.info("No content variations available for comparison")
            return
        
        # Simple comparison display
        for i, content in enumerate(content_variations):
            with st.expander(f"Variation {i+1}"):
                st.write(content)

    def render_comparison_interface(self, content_variations: List[str] = None):
        """Render enhanced comparison interface with detailed analysis."""
        st.markdown("### 🔍 Content Comparison & Analysis")
        
        if not content_variations or len(content_variations) < 2:
            self._render_no_comparison_state()
            return
        
        # Comparison controls
        self._render_comparison_controls(content_variations)
        
        # Side-by-side comparison
        self._render_side_by_side_comparison(content_variations)
        
        # Analysis metrics
        self._render_comparison_metrics(content_variations)

    def render_detailed_comparison(self, content_variations: List[str] = None):
        """Render detailed comparison with diff analysis."""
        if not content_variations:
            st.info("No content variations to analyze")
            return
        
        st.subheader("📋 Detailed Content Analysis")
        
        # Content selection for comparison
        if len(content_variations) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_1 = st.selectbox(
                    "Select First Content:",
                    range(len(content_variations)),
                    format_func=lambda x: f"Variation {x+1}"
                )
            
            with col2:
                selected_2 = st.selectbox(
                    "Select Second Content:",
                    range(len(content_variations)),
                    format_func=lambda x: f"Variation {x+1}",
                    index=1 if len(content_variations) > 1 else 0
                )
            
            if selected_1 != selected_2:
                self._render_diff_analysis(
                    content_variations[selected_1], 
                    content_variations[selected_2],
                    selected_1,
                    selected_2
                )

    def render_enhanced_comparison(self):
        """Render enhanced comparison with advanced features."""
        st.markdown("### 🎯 Advanced Content Comparison")
        
        # Get content from session state
        content_variations = st.session_state.get('content_variations', [])
        
        if not content_variations:
            self._render_empty_comparison_state()
            return
        
        # Enhanced comparison features
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Detailed Analysis", "📈 Performance"])
        
        with tab1:
            self._render_comparison_overview(content_variations)
        
        with tab2:
            self._render_detailed_analysis(content_variations)
        
        with tab3:
            self._render_performance_comparison(content_variations)

    def _render_no_comparison_state(self):
        """Render state when no content is available for comparison."""
        st.info("🔍 Generate multiple content variations to see comparison analysis")
        st.markdown("""
        **To enable comparison:**
        1. Enable "Generate Variations" in Content Generation
        2. Run content generation workflow
        3. Multiple variations will appear here for analysis
        """)

    def _render_empty_comparison_state(self):
        """Render empty state for enhanced comparison."""
        st.markdown("""
        <div style='text-align: center; padding: 2rem; border: 2px dashed #cccccc; border-radius: 10px;'>
            <h3>🎯 Advanced Content Comparison</h3>
            <p>Generate content variations to unlock powerful comparison features:</p>
            <ul style='text-align: left; display: inline-block;'>
                <li>📊 Side-by-side content analysis</li>
                <li>🔍 Word-level difference highlighting</li>
                <li>📈 Performance metrics comparison</li>
                <li>🎨 Style and tone analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    def _render_comparison_controls(self, content_variations: List[str]):
        """Render comparison control options."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_metrics = st.checkbox("📊 Show Metrics", value=True)
        
        with col2:
            highlight_diff = st.checkbox("🔍 Highlight Differences", value=True)
        
        with col3:
            show_stats = st.checkbox("📈 Show Statistics", value=False)
        
        return show_metrics, highlight_diff, show_stats

    def _render_side_by_side_comparison(self, content_variations: List[str]):
        """Render side-by-side content comparison."""
        if len(content_variations) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Variation 1**")
                st.markdown(f"``````")
                st.caption(f"Length: {len(content_variations[0])} characters")
            
            with col2:
                st.markdown("**Variation 2**")
                if len(content_variations) > 1:
                    st.markdown(f"``````")
                    st.caption(f"Length: {len(content_variations[1])} characters")

    def _render_comparison_metrics(self, content_variations: List[str]):
        """Render comparison metrics."""
        st.markdown("### 📊 Comparison Metrics")
        
        metrics_cols = st.columns(len(content_variations))
        
        for i, (content, col) in enumerate(zip(content_variations, metrics_cols)):
            with col:
                st.metric(
                    f"Variation {i+1}",
                    f"{len(content.split())} words",
                    delta=f"{len(content)} chars"
                )

    def _render_diff_analysis(self, content1: str, content2: str, idx1: int, idx2: int):
        """Render detailed diff analysis between two content pieces."""
        st.markdown(f"### 🔍 Difference Analysis: Variation {idx1+1} vs Variation {idx2+1}")
        
        # Calculate diff
        diff = list(difflib.unified_diff(
            content1.splitlines(keepends=True),
            content2.splitlines(keepends=True),
            fromfile=f'Variation {idx1+1}',
            tofile=f'Variation {idx2+1}',
            lineterm=''
        ))
        
        if diff:
            st.code(''.join(diff), language='diff')
        else:
            st.success("✅ No differences found between selected variations")

    def _render_comparison_overview(self, content_variations: List[str]):
        """Render comparison overview."""
        st.markdown("#### 📋 Content Overview")
        
        for i, content in enumerate(content_variations[:5]):  # Show max 5
            with st.expander(f"📄 Variation {i+1} - {len(content.split())} words"):
                st.write(content[:200] + "..." if len(content) > 200 else content)
                
                # Quick stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"Words: {len(content.split())}")
                with col2:
                    st.caption(f"Characters: {len(content)}")
                with col3:
                    st.caption(f"Sentences: {content.count('.')}")

    def _render_detailed_analysis(self, content_variations: List[str]):
        """Render detailed content analysis."""
        if len(content_variations) >= 2:
            st.markdown("#### 🔬 Detailed Analysis")
            
            # Content selection
            selected_content = st.selectbox(
                "Select content to analyze:",
                range(len(content_variations)),
                format_func=lambda x: f"Variation {x+1}"
            )
            
            content = content_variations[selected_content]
            
            # Analysis results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Content Statistics:**")
                st.write(f"- Word count: {len(content.split())}")
                st.write(f"- Character count: {len(content)}")
                st.write(f"- Sentence count: {content.count('.')}")
                st.write(f"- Average word length: {sum(len(word) for word in content.split()) / len(content.split()):.1f}")
            
            with col2:
                st.markdown("**Content Preview:**")
                st.write(content[:300] + "..." if len(content) > 300 else content)

    def _render_performance_comparison(self, content_variations: List[str]):
        """Render performance comparison metrics."""
        st.markdown("#### 📈 Performance Comparison")
        
        # Mock performance data (in production, this would be real metrics)
        performance_data = []
        for i, content in enumerate(content_variations):
            performance_data.append({
                'Variation': f'Variation {i+1}',
                'Engagement Score': 0.7 + (i * 0.05),
                'Clarity Score': 0.8 - (i * 0.03),
                'Persuasion Score': 0.75 + (i * 0.02),
                'Word Count': len(content.split())
            })
        
        # Display as metrics
        for data in performance_data:
            st.markdown(f"**{data['Variation']}**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Engagement", f"{data['Engagement Score']:.2f}")
            with col2:
                st.metric("Clarity", f"{data['Clarity Score']:.2f}")
            with col3:
                st.metric("Persuasion", f"{data['Persuasion Score']:.2f}")
