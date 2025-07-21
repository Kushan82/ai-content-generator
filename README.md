# Multi-Agent AI Content Generator

> Advanced content generation platform using collaborative AI agents with LangGraph orchestration

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-FF4B4B.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.16-purple.svg)](https://github.com/langchain-ai/langgraph)

## 🚀 Overview

The Multi-Agent AI Content Generator is a sophisticated content creation platform that leverages specialized AI agents working collaboratively to produce high-quality, persona-tailored marketing content. Built with modern Python frameworks and enterprise-grade architecture patterns.

### ✨ Key Features

- **🤖 Multi-Agent Architecture**: Five specialized AI agents working in orchestrated workflows
- **🎯 Persona-Driven Content**: Deep persona research drives personalized content creation
- **📊 Real-Time Monitoring**: Live workflow tracking with WebSocket updates
- **⚡ High Performance**: Optimized for speed with caching and parallel processing
- **🔧 Production Ready**: Comprehensive testing, monitoring, and deployment tools
- **📈 Analytics Dashboard**: Performance metrics and quality assessment

### 🏗️ Architecture




## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Groq API Key 

### 1. Clone Repository

git clone 
cd ai-content-generator

### 2. Environment Setup
Create virtual environment
    -uv venv
    -.venv\Scripts\activate.ps1

Install dependencies
    -uv pip install -r requirements.txt

### 3. Configuration
Create a .env file to store the api key


### 4. Run Development Server

Start FastAPI backend
uvicorn app.main:app --reload --port 8000

Start Streamlit frontend (new terminal)
streamlit run frontend/streamlit_app.py 

### 5. Access Application

- **Frontend Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🐳 Docker Deployment

### Development
docker-compose up -d

### Production

Build and deploy
docker-compose -f docker/docker-compose.prod.yml up -d

Monitor logs
docker-compose -f docker/docker-compose.prod.yml logs -f


## 🤖 Agent System

### Specialized Agents

| Agent | Role | Capabilities |
|-------|------|-------------|
| **Persona Research** | Market Research & Demographics | Demographic analysis, psychographic profiling, competitive research |
| **Content Strategy** | Strategic Planning & Messaging | Messaging strategy, content planning, persuasion optimization |
| **Creative Generation** | Content Creation & Copywriting | Creative copywriting, brand voice adaptation, multi-format content |
| **Quality Assurance** | Quality Control & Optimization | Content assessment, strategic alignment, brand consistency |
| **Orchestrator** | Workflow Coordination | Task delegation, agent coordination, result synthesis |

### Workflow Patterns

- **Sequential**: Step-by-step agent execution with data accumulation
- **Parallel**: Concurrent agent execution for faster processing
- **Conditional**: Dynamic routing based on intermediate results
- **Iterative**: Quality-driven refinement loops

## 📊 API Reference

### Content Generation

