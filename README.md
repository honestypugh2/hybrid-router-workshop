# Hybrid LLM Model Router Workshop

A hands-on workshop for building hybrid AI chatbots that intelligently route between on-device and cloud models

## 🎯 Overview

This workshop guides you through building a **hybrid AI chatbot** that combines the best of both worlds:

- **On-device models** (Azure Foundry Local) for fast, private responses
- **Cloud models** (Azure AI Foundry) for complex, sophisticated analysis
- **Intelligent routing** that automatically selects the best model for each query
- **Seamless context sharing** across model transitions

## 🖼️ Frontend Preview

![Chat Interface](images/advanced_hybrid_example1.png)
*Interactive Streamlit chat interface with model routing transparency*

## 🏗️ Architecture

```
User Query → Router Logic → [Local Model] or [Cloud Model] → Unified Response
                ↓                ↓              ↓
            Analytics     Fast Response   Smart Analysis
```

The system maintains conversation context across both local and cloud models, providing transparency about where each response was processed while ensuring a seamless user experience.

## 🚀 Key Features

- **⚡ Low-Latency Local Responses**: Simple queries answered instantly on-device
- **🧠 Intelligent Cloud Escalation**: Complex tasks automatically routed to Azure
- **🔄 Seamless Context Sharing**: Unified conversation across model switches
- **👁️ Full Transparency**: Clear indication of processing location
- **📊 Comprehensive Observability**: Telemetry and performance monitoring
- **🎨 Interactive Frontend**: Streamlit-based chat interface

## 📋 Prerequisites

- Python 3.10+
- Azure subscription with AI Foundry access
- Azure Foundry Local installation
- Basic knowledge of Python and Azure services
- Azure API Management (optional)

### Assumption

This workshop uses a Windows machine for on-device.

## 🛠️ Setup

1. **Clone the workshop materials**:

   ```bash
   git clone <repository-url>
   cd hybrid-router-workshop
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:

   ```bash
   cp .env.example .env
   # Edit .env with your Azure credentials and endpoints
   ```

5. **Login into Azure**:

   ```bash
   az login
   ```

## 📚 Workshop Structure

### 🔧 Core Modules (`modules/`)

- **`router.py`**: Rule-based intelligent query routing logic
- **`bert_router.py`**: BERT-based query classification router
- **`phi_router.py`**: Phi SLM-based query routing with fine-tuning
- **`hybrid_router.py`**: Multi-tier hybrid routing orchestration
- **`azure_ai_manager.py`**: Azure AI Foundry integration and management
- **`context_manager.py`**: Conversation history and session management
- **`telemetry.py`**: Observability, performance tracking, and analytics
- **`config.py`**: Configuration management and environment settings
- **`helper.py`**: Utility functions and common operations

### 📓 Lab Notebooks (`notebooks/`)

#### **Lab 1: Environment Setup** (`lab1_environment_setup.ipynb`)

- Configure Azure Foundry Local and Cloud environments
- Test connectivity and model availability
- Verify authentication and basic functionality

#### **Lab 2: Local Model Testing** (`lab2_local_model_testing.ipynb`)

- Explore on-device model capabilities
- Test response times and quality
- Understand local model limitations and strengths

#### **Lab 3: Azure Model Testing** (`lab3_azure_model_testing.ipynb`)

- Configure Azure OpenAI integration
- Test cloud model capabilities
- Compare performance with local models

#### **Lab 3 (Alternative): Azure AI Foundry Agents** (`lab3_azure_ai_foundry_agents_testing.ipynb`)

- Work with Azure AI Foundry agent framework
- Implement agent-based conversation patterns
- Explore advanced agent capabilities and orchestration

#### **Lab 4: Model Routing Logic** (`lab4_model_routing.ipynb`)

- Implement intelligent query analysis
- Build routing decision engine
- Test routing accuracy with various query types

#### **Lab 4 (Extended): BERT Query Router** (`lab4_bert_query_router.ipynb`)

- Deep dive into BERT-based query classification
- Fine-tune models for specific routing scenarios
- Evaluate routing accuracy and performance

#### **Lab 4 (SLM): Phi Small Language Model Routing** (`lab4_phi_slm_routing.ipynb`)

- Implement routing using Phi small language models
- Fine-tune Phi models for routing decisions
- Compare SLM vs traditional ML approaches

#### **Lab 4 (Alternative): Foundry Agent Routing** (`lab4_foundry_agent_routing.ipynb`)

- Route between different Azure Foundry agents
- Implement multi-agent coordination
- Build agent-specific routing logic

#### **Lab 4 (Integration): API Management Router** (`lab4_apim_model_router.ipynb`)

- Integrate with Azure API Management
- Implement enterprise-grade routing patterns
- Add rate limiting and authentication

#### **Lab 5: Hybrid Orchestration** (`lab5_hybrid_orchestration.ipynb`)

- Combine local and cloud models seamlessly
- Implement conversation context sharing
- Build multi-turn conversation management

#### **Lab 6: Observability & Telemetry** (`lab6_observability_telemetry.ipynb`)

- Add comprehensive logging and monitoring
- Implement performance metrics collection
- Create analytics dashboard for insights

#### **Lab 7: Frontend Chat Interface** (`lab7_frontend_chat_interface.ipynb`)

- Build Streamlit-based chat UI
- Implement real-time conversation interface
- Add visual indicators for model routing

#### **Lab 7 (Advanced): Frontend Chat Interface** (`lab7_advanced_routing_agents.ipynb`)

- Implementing sophisticated multi-agent routing systems
- Building complex decision trees for agent selection
- Creating advanced conversation orchestration patterns

## 🎯 Success Criteria

By completing this workshop, you will have built a system that demonstrates:

✅ **Low-Latency Local Responses**: Simple queries answered in <0.5s on-device  
✅ **Seamless Cloud Escalation**: Complex tasks automatically routed without user friction  
✅ **Context Continuity**: Unified conversation memory across model switches  
✅ **Transparency & Control**: Clear indication of processing location  
✅ **Performance Optimization**: Measurable speed and efficiency gains  
✅ **Stakeholder-Ready Demo**: Interactive interface for business validation  

### 📱 User Interface Examples

![Local Response Example](images/advanced_hybrid_example1.png)
*Fast local model response with performance indicators*

![Cloud Response Example](images/advanced_hybrid_example2.png)
*Complex cloud model response with routing explanation*

## 📊 Expected Outcomes

### Performance Metrics

- **Local queries**: 70-80% of simple queries, <0.5s response time
- **Cloud queries**: 20-30% of complex queries, 1-3s response time
- **Context preservation**: 100% accuracy across model transitions
- **User satisfaction**: Seamless experience with transparency

### Technical Achievements

- **Hybrid Architecture**: Production-ready routing system
- **Scalable Design**: Modular components for easy enhancement
- **Comprehensive Monitoring**: Full observability stack
- **Business Value**: Clear ROI demonstration through metrics

*Complete hybrid routing system architecture diagram (COMING SOON)*

## 🛟 Troubleshooting

### Common Issues

**Connection Problems**:

- Verify Azure credentials in `.env`
- Check Azure Foundry Local service status
- Confirm network connectivity

**Model Loading Issues**:

- Ensure sufficient local resources
- Verify model deployment in Azure
- Check API quotas and limits

**Routing Problems**:

- Review query classification logic
- Adjust complexity thresholds
- Test with various query types

## 🔗 Additional Resources

- [Azure AI Foundry Documentation](https://docs.microsoft.com/azure/ai-foundry)
- [Azure Foundry Local Setup Guide](https://github.com/guygregory/FoundryLocal)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Streamlit Documentation](https://docs.streamlit.io)
- [MobileBERT Reference](https://huggingface.co/docs/transformers/model_doc/mobilebert#mobilebert)
- [Phi CookBook Reference](https://github.com/microsoft/PhiCookBook)

## 🤝 Contributing

This workshop is designed for educational purposes. Feedback and improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## 📄 License

This workshop is provided under the MIT License. See `LICENSE` file for details.

## 🆘 Support

For workshop-specific questions:

- Create an issue in this repository
- Contact the workshop facilitators
- Join the discussion in our community forum

---

## Ready to build the future of hybrid AI? Let's get started with Lab 1! 🚀
