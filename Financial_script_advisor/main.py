# config.py
from dataclasses import dataclass
from typing import Dict, Optional
from camel.configs import ChatGPTConfig
import os
from dotenv import load_dotenv
from camel.types import TaskType, ModelType, ModelPlatformType

# load_dotenv()

# @dataclass
# class ModelConfig:
#     """Configuration for model endpoints"""
#     model_platform: ModelPlatformType = ModelPlatformType.VLLM
#     model_type: ModelType = ModelType.GPT_3_5_TURBO  # or your local model
#     api_key: str = os.getenv("API_KEY", "EMPTY")
#     url: str = os.getenv("API_BASE", "http://localhost:8088/v1")
#     temperature: float = 0.7
#     top_p: float = 0.95
#     max_tokens: int = 8192

# main.py
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import json
from camel.types import RoleType

from camel.societies import RolePlaying
from camel.types import TaskType, ModelType, ModelPlatformType
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.messages import BaseMessage


    
def as_dict(self) -> dict:
    """Convert config to dictionary for ModelFactory"""
    return {
        "api_key": self.api_key,
        "url": self.base_url,
        "temperature": self.temperature,
        "top_p": self.top_p,
        "max_tokens": self.max_tokens
    }
    
model = ModelFactory.create(
    model_platform=ModelPlatformType.VLLM,
    # model_platform = ModelPlatformType.VLLM,
    # model_type=ModelType.STUB,  # Use STUB for custom models
    model_type="Qwen/Qwen3-4B-Instruct-2507",
    api_key="Not_Used",
    url="http://localhost:8090/v1",
    model_config_dict={
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 8192
    }
)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecommendationType(Enum):
    """Standardized recommendation types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class AnalystRole(Enum):
    """Defined analyst roles"""
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    RISK = "risk"
    MANAGER = "manager"

class StockAnalysisDebate:
    """Main class for conducting multi-agent stock analysis debates"""
    
    # Enhanced role definitions with structured output requirements
    AGENT_ROLES = {
        AnalystRole.FUNDAMENTAL: """You are a Senior Fundamental Analyst with CFA certification. 
        Analyze the stock using:
        1. Financial metrics (P/E, P/B, EV/EBITDA, PEG ratio)
        2. Revenue and earnings growth trends
        3. Debt-to-equity ratio and interest coverage
        4. Free cash flow analysis
        5. Competitive positioning and moat analysis
        
        Provide your analysis in this format:
        - Key Metrics: [list key financial ratios]
        - Strengths: [bullet points]
        - Concerns: [bullet points]
        - Score (0-10): [your score]
        - Rationale: [brief explanation]""",
        
        AnalystRole.TECHNICAL: """You are a Certified Market Technician (CMT). 
        Analyze the stock using:
        1. Price action and trend analysis
        2. Moving averages (20, 50, 200 DMA)
        3. RSI, MACD, and momentum indicators
        4. Support and resistance levels
        5. Volume analysis and accumulation/distribution
        
        Provide your analysis in this format:
        - Current Trend: [bullish/bearish/neutral]
        - Key Levels: [support and resistance]
        - Technical Indicators: [status of key indicators]
        - Score (0-10): [your score]
        - Rationale: [brief explanation]""",
        
        AnalystRole.SENTIMENT: """You are a Market Sentiment and Behavioral Finance Expert. 
        Analyze the stock using:
        1. Recent news sentiment and media coverage
        2. Analyst ratings and target price changes
        3. Insider trading activity
        4. Social media sentiment and retail investor interest
        5. Options flow and put/call ratios
        
        Provide your analysis in this format:
        - News Sentiment: [positive/negative/neutral]
        - Analyst Consensus: [summary]
        - Market Positioning: [bullish/bearish]
        - Score (0-10): [your score]
        - Rationale: [brief explanation]""",
        
        AnalystRole.RISK: """You are a Risk Management Expert specializing in equity markets.
        Analyze the stock's risk profile including:
        1. Market risk (beta, correlation with indices)
        2. Company-specific risks (regulatory, operational)
        3. Sector/Industry risks
        4. Liquidity risk
        5. Event risks (earnings, product launches, litigation)
        
        Provide your analysis in this format:
        - Risk Level: [low/medium/high]
        - Key Risks: [bullet points]
        - Risk Mitigation: [suggestions]
        - Risk-Adjusted Score (0-10): [your score]
        - Rationale: [brief explanation]""",
        
        AnalystRole.MANAGER: """You are the Chief Investment Officer responsible for final investment decisions.
        
        Evaluate each analyst's arguments considering:
        1. Quality and depth of analysis
        2. Data accuracy and relevance
        3. Logical consistency
        4. Risk-reward balance
        
        Provide your decision in this format:
        - Analyst Scores:
          * Fundamental: [0-10]
          * Technical: [0-10]
          * Sentiment: [0-10]
          * Risk: [0-10]
        - Weighted Score: [calculated score]
        - Recommendation: [STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL]
        - Confidence Level: [HIGH/MEDIUM/LOW]
        - Key Factors: [top 3 decision factors]
        - Risk Warnings: [any important risks to note]"""
    }
    
    def __init__(self, 
                 model=None,
                 model_config: Optional[ChatGPTConfig] = None,
                 weights: Optional[Dict[AnalystRole, float]] = None,
                 enable_risk_analyst: bool = True):
        """
        Initialize the debate system
        
        Args:
            model_config: Configuration for the language model
            weights: Custom weights for each analyst (must sum to 1.0)
            enable_risk_analyst: Whether to include risk analyst in the debate
        """
        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.VLLM,
            model_type="Qwen/Qwen3-4B-Instruct-2507",
            api_key="Not_Used",
            url="http://localhost:8090/v1",
            model_config_dict={
                # "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 8192
            }
        )
        self.model_config = model_config or self._get_default_config()
        self.enable_risk_analyst = enable_risk_analyst
        
        # Default weights including risk analyst
        default_weights = {
            AnalystRole.FUNDAMENTAL: 0.35,
            AnalystRole.TECHNICAL: 0.25,
            AnalystRole.SENTIMENT: 0.20,
            AnalystRole.RISK: 0.20
        }
        
        if not enable_risk_analyst:
            # Redistribute weights if risk analyst is disabled
            default_weights = {
                AnalystRole.FUNDAMENTAL: 0.40,
                AnalystRole.TECHNICAL: 0.30,
                AnalystRole.SENTIMENT: 0.30
            }
            default_weights.pop(AnalystRole.RISK, None)
        
        self.weights = weights or default_weights
        self._validate_weights()
        
    def _get_default_config(self) -> ChatGPTConfig:
        """Get default model configuration"""
        config = ModelConfig()
        return ChatGPTConfig(
            api_key=config.api_key,
            url=config.url,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            model=config.model_type.value
        )
        
    def _validate_weights(self):
        """Validate that weights sum to 1.0"""
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def create_agent(self, role: AnalystRole, stock_symbol: str) -> RolePlaying:
        """Create an agent with specific role"""
        try:
            import os
            os.environ["OPENAI_API_KEY"] = "EMPTY"
            role_prompt = self.AGENT_ROLES[role]
            
            # Create model using ModelFactory
            task_prompt = f"{role_prompt}\n\nThe stock symbol being analyzed is {stock_symbol}."
            from camel.configs import BaseConfig
            
            class VLLMConfig(BaseConfig):
                api_key: str = "Not_Used"
                url: str = "http://localhost:8090/v1"
                model: str = "Qwen/Qwen3-4B-Instruct-2507"
            
            vllm_config = VLLMConfig()
        
            return RolePlaying(
                assistant_role_name=role.value,
                user_role_name="investor",
                task_prompt=task_prompt,
                task_type=TaskType.AI_SOCIETY,
                with_task_specify=False,
                    assistant_agent_kwargs={
                "model": self.model,  # Pass the model instance directly
                    },
                    user_agent_kwargs={
                        "model": self.model,  # Pass the model instance directly
                    }
                )
        except Exception as e:
            logger.error(f"Failed to create agent for role {role}: {str(e)}")
            raise
    
    def run_analyst_debate(self, stock_symbol: str) -> Dict:
        """
        Run the complete debate process
        
        Args:
            stock_symbol: Stock ticker symbol to analyze
            
        Returns:
            Dictionary containing analysis results and recommendation
        """
        logger.info(f"Starting analysis debate for {stock_symbol}")
        
        # Validate stock symbol
        if not stock_symbol or not isinstance(stock_symbol, str):
            raise ValueError("Invalid stock symbol provided")
        
        stock_symbol = stock_symbol.upper().strip()
        
        # Create analysts
        analyst_roles = [AnalystRole.FUNDAMENTAL, AnalystRole.TECHNICAL, AnalystRole.SENTIMENT]
        if self.enable_risk_analyst:
            analyst_roles.append(AnalystRole.RISK)
        
        agents = {}
        for role in analyst_roles:
            try:
                agents[role] = self.create_agent(role, stock_symbol)
            except Exception as e:
                logger.error(f"Failed to create {role} agent: {str(e)}")
                raise
        
        # Collect analyst responses
        all_responses = {}
        analysis_prompt = f"""Provide a comprehensive analysis of {stock_symbol} stock in Indian Market
        based on your expertise. Use current market data and follow the specified format."""
        
        for role, agent in agents.items():
            try:
                logger.info(f"Getting analysis from {role.value} analyst")
                msg = BaseMessage(
                    role_name="investor",
                    role_type=RoleType.USER,
                    meta_dict={},
                    content=analysis_prompt
                )
                assistant_msg, _ = agent.step(msg)
                print("manager","\n",assistant_msg,"\n",assistant_msg.msgs[0].content)
                all_responses[role] = assistant_msg.msgs[0].content
                logger.info(f"Received response from {role.value} analyst")
            except Exception as e:
                logger.error(f"Error getting response from {role}: {str(e)}")
                all_responses[role] = f"Error: Unable to complete analysis - {str(e)}"
        
        # Manager evaluation
        manager_decision = self._get_manager_decision(stock_symbol, all_responses)
        
        # Compile results
        results = {
            "stock_symbol": stock_symbol,
            "timestamp": datetime.now().isoformat(),
            "analyst_reports": {role.value: response for role, response in all_responses.items()},
            "weights": {role.value: weight for role, weight in self.weights.items()},
            "manager_decision": manager_decision,
            "metadata": {
                # "model": self.model_config.model_type,
                "risk_analyst_enabled": self.enable_risk_analyst
            }
        }
        
        return results
    
    def _get_manager_decision(self, stock_symbol: str, analyst_responses: Dict[AnalystRole, str]) -> Dict:
        """Get final decision from manager"""
        try:
            manager = self.create_agent(AnalystRole.MANAGER, stock_symbol)
            
            # Format analyst responses for manager
            formatted_responses = "\n\n".join([
                f"{'='*50}\n{role.value.upper()} ANALYST REPORT:\n{'='*50}\n{content}"
                for role, content in analyst_responses.items()
            ])
            
            # Add weights information
            weights_info = "\n".join([
                f"- {role.value.capitalize()}: {weight*100:.0f}%"
                for role, weight in self.weights.items()
            ])
            
            manager_prompt = f"""
            Review the following analyst reports for {stock_symbol} and provide your final investment decision.
            
            ANALYST WEIGHTS:
            {weights_info}
            
            ANALYST REPORTS:
            {formatted_responses}
            
            Based on these analyses and the assigned weights, provide your final recommendation 
            following the specified format.
            """
            
            msg = BaseMessage(
                role_name="investor",
                role_type=RoleType.USER,
                meta_dict={},
                content=manager_prompt
            )
            
            # response = manager.step(msg)
            response, _ = manager.step(msg)
            print("manager","\n",response,"\n",response.msgs[0].content)
            return {
                "decision": response.msgs[0].content,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Manager decision failed: {str(e)}")
            return {
                "decision": "Error: Unable to reach final decision",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def save_results(self, results: Dict, output_path: str = None):
        """Save analysis results to JSON file"""
        if output_path is None:
            output_path = f"stock_analysis_{results['stock_symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise

# Example usage with error handling
def main():
    """Main execution function"""
    
    model = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type="Qwen/Qwen3-4B-Instruct-2507",
        api_key="Not_Used",
        url="http://localhost:8090/v1",
        model_config_dict={
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 15000
            }
        )
    try:
        # Initialize debate system
        debate_system = StockAnalysisDebate(
            enable_risk_analyst=True,
            weights={
                AnalystRole.FUNDAMENTAL: 0.35,
                AnalystRole.TECHNICAL: 0.25,
                AnalystRole.SENTIMENT: 0.20,
                AnalystRole.RISK: 0.20
            },
            model=model,
            model_config=model.model_config_dict
        )
        
        # Run analysis
        stock_symbol = "RIL"
        results = debate_system.run_analyst_debate(stock_symbol)
        
        # Save results
        debate_system.save_results(results)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"STOCK ANALYSIS COMPLETE: {stock_symbol}")
        print(f"{'='*60}")
        print("\nMANAGER'S FINAL DECISION:")
        print(results["manager_decision"]["decision"])
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
