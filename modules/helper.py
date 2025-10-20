import os
import sys
import uuid
import time
import requests
from dotenv import find_dotenv, load_dotenv
from openai import AzureOpenAI, OpenAI
import re

# Load configuration from Lab 1
load_dotenv(find_dotenv(".env"))

# Add parent directory for module imports
sys.path.append(os.path.dirname(os.getcwd()))
# Add modules to path
sys.path.append('../modules')


# Import our custom modules
from modules.router import HybridRouter, ModelTarget, QueryAnalysis
from modules.context_manager import ConversationManager, ModelSource
from modules.telemetry import TelemetryCollector, EventType, MetricType

# Local model configuration
LOCAL_ENDPOINT = os.environ["LOCAL_MODEL_ENDPOINT"] 
LOCAL_MODEL_ALIAS = os.environ["LOCAL_MODEL_NAME"]
AZURE_OPENAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
LOCAL_MODEL_ID = os.environ["LOCAL_MODEL_ID"]

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY')
AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_DEPLOYMENT_NAME')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')

# ============================================================================
# TELEMETRY
# ============================================================================

LOCAL_MODEL = LOCAL_MODEL_ID
AZURE_DEPLOYMENT = AZURE_OPENAI_DEPLOYMENT

# Initialize router and conversation manager with telemetry
router = HybridRouter(complexity_threshold=0.5)
conversation_manager = ConversationManager(max_history_length=20)

def answer_with_telemetry(user_message: str, conversation_manager: ConversationManager, 
                         session_id: str, show_reasoning: bool = False, telemetry: TelemetryCollector = None,):
    """
    Answer a question using the hybrid routing system with comprehensive telemetry.
    
    Args:
        user_message: The user's input
        conversation_manager: ConversationManager instance
        session_id: Unique session identifier
        show_reasoning: Whether to include routing reasoning in response
    
    Returns:
        tuple: (response_text, response_time, source, success, query_id)
    """
    # Generate unique query ID
    query_id = str(uuid.uuid4())[:8]
    
    # Log query received
    telemetry.log_query_received(user_message, session_id, query_id)
    
    # Add user message to conversation history
    conversation_manager.add_user_message(user_message)
    
    # Start telemetry trace
    with telemetry.trace_operation("hybrid_query_processing", session_id, query_id, 
                                 query_preview=user_message[:50]) as span:
        
        try:
            # Analyze query characteristics
            analysis_start = time.time()
            analysis = router.analyze_query_characteristics(user_message)
            analysis_time = time.time() - analysis_start
            
            # Make routing decision
            target, reason = router.route_query(user_message, analysis)
            
            # Log routing decision
            telemetry.log_routing_decision(
                user_message, target.value, reason, 
                analysis.complexity_score, session_id, query_id
            )
            
            # Track model switches
            last_source = getattr(conversation_manager, '_last_model_used', None)
            if last_source and last_source != target.value:
                telemetry.log_model_switch(last_source, target.value, session_id, query_id)
            conversation_manager._last_model_used = target.value
            
            # Get appropriate conversation history for the target model
            messages = conversation_manager.get_messages_for_model(target.value)
            
            # Prepare request details
            request_details = {
                "messages_count": len(messages),
                "analysis_time": analysis_time,
                "complexity_score": analysis.complexity_score,
                "estimated_tokens": analysis.estimated_tokens
            }
            
            # Log model request
            telemetry.log_model_request(target.value, session_id, query_id, request_details)
            
            # Make API call
            start_time = time.time()
            
            if target == ModelTarget.LOCAL:
                # Simulate local model call (replace with actual call when available)
                if local_client:
                    response = local_client.chat.completions.create(
                        model=LOCAL_MODEL,
                        messages=messages,
                        max_tokens=200,
                        temperature=0.7
                    )
                    content = response.choices[0].message.content
                else:
                    # Mock response for demonstration
                    time.sleep(0.1)  # Simulate fast local response
                    content = "This is a simulated local model response."
                
                source_tag = "[LOCAL]"
                actual_source = ModelSource.LOCAL
                
            else:  # target == ModelTarget.CLOUD
                # Simulate cloud model call (replace with actual call when available)
                if azure_client:
                    response = azure_client.chat.completions.create(
                        model=AZURE_DEPLOYMENT,
                        messages=messages,
                        max_tokens=400,
                        temperature=0.7
                    )
                    content = response.choices[0].message.content
                else:
                    # Mock response for demonstration
                    time.sleep(1.5)  # Simulate slower cloud response
                    content = "This is a simulated cloud model response with more detailed analysis and comprehensive information."
                
                source_tag = "[CLOUD]"
                actual_source = ModelSource.CLOUD
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Format response with source tag
            if show_reasoning:
                formatted_response = f"{source_tag} {content}\n\n[Routing: {reason}]"
            else:
                formatted_response = f"{source_tag} {content}"
            
            # Log successful model response
            response_details = {
                "content_length": len(content),
                "reasoning_shown": show_reasoning,
                "total_processing_time": end_time - analysis_start
            }
            
            telemetry.log_model_response(
                target.value, response_time, True, session_id, query_id, response_details
            )
            
            # Add assistant response to conversation history
            conversation_manager.add_assistant_message(
                formatted_response, actual_source, response_time
            )
            
            return formatted_response, response_time, actual_source.value, True, query_id
            
        except Exception as e:
            error_time = time.time() - start_time if 'start_time' in locals() else 0
            
            # Log error
            telemetry.log_error(
                e, "answer_with_telemetry", session_id, query_id,
                {"processing_stage": "model_call", "target_model": target.value if 'target' in locals() else "unknown"}
            )
            
            # Log failed model response
            if 'target' in locals():
                telemetry.log_model_response(
                    target.value, error_time, False, session_id, query_id,
                    {"error_message": str(e)}
                )
            
            error_msg = f"[ERROR] {str(e)}"
            conversation_manager.add_assistant_message(error_msg, ModelSource.ERROR, error_time)
            
            return error_msg, error_time, "error", False, query_id

# ============================================================================
# LOCAL
# ============================================================================
def get_local_client():
    """Get configured OpenAI client for local model."""
    return OpenAI(
        base_url=f"{LOCAL_ENDPOINT}/v1",
        api_key="not-needed"
    )

def query_local_model_simple(prompt, local_available=True):
    """Simple local model query function."""
    if not local_available:
        return "Local model not available", 0, False
    
    try:
        start_time = time.time()
        response = get_local_client().chat.completions.create(
            model=get_local_client()[0].id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        end_time = time.time()
        
        content = response.choices[0].message.content
        return content, end_time - start_time, True
    except Exception as e:
        return f"Error: {str(e)}", 0, False

def query_local_model(prompt):
    """Send a query to the local model and return the response with timing."""
    try:
        start_time = time.time()
        
        # Ensure we're using the properly configured client
        response = get_local_client().chat.completions.create(
            model=get_local_client()[0].id,  # Use the alias directly
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
            stream=False
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        content = response.choices[0].message.content
        return content, response_time
        
    except Exception as e:
        return f"Error: {str(e)}", 0

def query_local_with_history(prompt, chat_history=None):
    """Query local model with optional chat history."""
    if chat_history is None:
        chat_history = []
    
    # Add current prompt to history
    messages = chat_history + [{"role": "user", "content": prompt}]
    
    try:
        start_time = time.time()
        
        response = get_local_client().chat.completions.create(
            model=LOCAL_MODEL_ALIAS,
            messages=messages,
            max_tokens=150,
            temperature=0.7,
            stream=False
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        content = response.choices[0].message.content
        return content, response_time, True
        
    except Exception as e:
        return f"Error: {str(e)}", 0, False
    
# ============================================================================
# AZURE OPENAI - DIRECT
# ============================================================================
def get_azure_client():
    """Get configured Azure OpenAI client."""
    return AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )


def query_azure_with_history(prompt, chat_history=None, max_tokens=500):
    """Query Azure model with optional chat history."""
    if chat_history is None:
        chat_history = []
    
    # Add current prompt to history
    messages = chat_history + [{"role": "user", "content": prompt}]
    
    try:
        start_time = time.time()
        
        response = get_azure_client().chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        content = response.choices[0].message.content
        return content, response_time, True
        
    except Exception as e:
        return f"Error: {str(e)}", 0, False

def query_with_direct_openai(prompt: str, chat_history: list = None, max_tokens: int = 500, azure_client=None) -> tuple:
    """Query using direct Azure OpenAI client."""
    if not azure_client:
        return "Azure OpenAI client not available", 0, False
    
    try:
        start_time = time.time()
        
        # Prepare messages
        if chat_history is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = chat_history + [{"role": "user", "content": prompt}]
        
        # Make API call
        response = azure_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        end_time = time.time()
        content = response.choices[0].message.content
        return content, end_time - start_time, True
        
    except Exception as e:
        return f"Direct OpenAI error: {str(e)}", 0, False

def analyze_query_characteristics(query):
    """Analyze various characteristics of a query to inform routing decisions."""
    analysis = {
        'original_query': query,
        'length': len(query),
        'word_count': len(query.split()),
        'has_complex_keywords': False,
        'is_greeting': False,
        'is_simple_question': False,
        'requires_analysis': False,
        'requires_creativity': False,
        'is_calculation': False
    }
    
    query_lower = query.lower().strip()
    
    # Complex task keywords that typically require cloud processing
    complex_keywords = [
        'summarize', 'analyze', 'explain in detail', 'comprehensive',
        'write a report', 'business plan', 'strategy', 'compare and contrast',
        'evaluate', 'assess', 'research', 'investigate', 'elaborate',
        'pros and cons', 'advantages and disadvantages', 'implications',
        'create a plan', 'develop a', 'design a', 'write an essay'
    ]
    
    # Creative task keywords
    creative_keywords = [
        'write a poem', 'write a story', 'create a character',
        'creative writing', 'brainstorm', 'imagine', 'invent',
        'compose', 'draft a letter', 'write a script'
    ]
    
    # Simple greeting patterns
    greeting_patterns = [
        r'^(hi|hello|hey|good morning|good afternoon|good evening)',
        r'^(how are you|what\'s up|greetings)'
    ]
    
    # Simple question patterns
    simple_patterns = [
        r'^what is',
        r'^who is',
        r'^where is',
        r'^when is',
        r'^how much is',
        r'^what time',
        r'^what day'
    ]
    
    # Math/calculation patterns
    calculation_patterns = [
        r'\d+\s*[+\-*/]\s*\d+',  # Basic math operations
        r'calculate|compute|solve|convert',
        r'\d+\s*(degrees|celsius|fahrenheit)',  # Temperature conversion
        r'what is \d+',  # "What is 2+2" type questions
    ]
    
    # Check for complex keywords
    for keyword in complex_keywords:
        if keyword in query_lower:
            analysis['has_complex_keywords'] = True
            analysis['requires_analysis'] = True
            break
    
    # Check for creative keywords
    for keyword in creative_keywords:
        if keyword in query_lower:
            analysis['requires_creativity'] = True
            break
    
    # Check for greetings
    for pattern in greeting_patterns:
        if re.match(pattern, query_lower):
            analysis['is_greeting'] = True
            break
    
    # Check for simple questions
    for pattern in simple_patterns:
        if re.match(pattern, query_lower):
            analysis['is_simple_question'] = True
            break
    
    # Check for calculations
    for pattern in calculation_patterns:
        if re.search(pattern, query_lower):
            analysis['is_calculation'] = True
            break
    
    return analysis