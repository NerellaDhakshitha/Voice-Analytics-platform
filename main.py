"""
Voice-Controlled Data Analytics Platform
Modern implementation with web interface and enhanced error handling
"""
import pandas as pd
import streamlit as st
import logging
import time
import yaml
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from analysis import DataAnalyzer
from stt import SpeechRecognizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_analytics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VoiceAnalyticsApp:
    """
    Modern Voice-Controlled Data Analytics Platform
    Features: Real-time speech processing, interactive web interface, session analytics
    """
    
    def __init__(self):
        self.config = self._load_config()
        self.df = self._load_dataset()
        self.analyzer = DataAnalyzer()
        self.speech = SpeechRecognizer()
        self.session_metrics = {
            'commands_processed': 0,
            'successful_queries': 0,
            'total_processing_time': 0.0,
            'session_start': time.time()
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with comprehensive error handling"""
        config_path = Path("config.yaml")
        
        if not config_path.exists():
            logger.error("config.yaml not found")
            # Create default config if running in Streamlit
            if 'streamlit' in sys.modules:
                st.error("⚠️ Configuration file missing. Please create config.yaml with your dataset path.")
                st.stop()
            else:
                print("Error: config.yaml not found. Please create it with your dataset path.")
                sys.exit(1)
        
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                logger.info("Configuration loaded successfully")
                return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config.yaml: {e}")
            if 'streamlit' in sys.modules:
                st.error(f"Configuration file error: {e}")
                st.stop()
            else:
                sys.exit(1)
    
    def _load_dataset(self) -> pd.DataFrame:
        """Load dataset with multiple encoding support and validation"""
        dataset_path = self.config.get('dataset_path')
        
        if not dataset_path:
            logger.error("Dataset path not specified in config")
            if 'streamlit' in sys.modules:
                st.error("Dataset path not found in configuration")
                st.stop()
            else:
                sys.exit(1)
        
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            logger.error(f"Dataset not found at {dataset_path}")
            if 'streamlit' in sys.modules:
                st.error(f"Dataset file not found: {dataset_path}")
                st.stop()
            else:
                sys.exit(1)
        
        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(dataset_path, encoding=encoding)
                logger.info(f"Dataset loaded successfully with {encoding} encoding")
                logger.info(f"Dataset shape: {df.shape}, Columns: {list(df.columns)}")
                
                # Basic data validation
                if df.empty:
                    raise ValueError("Dataset is empty")
                
                return df
                
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error loading dataset with {encoding}: {e}")
                continue
        
        logger.error("Failed to load dataset with any encoding")
        if 'streamlit' in sys.modules:
            st.error("Unable to load dataset. Please check file format and encoding.")
            st.stop()
        else:
            sys.exit(1)
    
    def run_streamlit_app(self):
        """Modern Streamlit web interface with enhanced UX"""
        st.set_page_config(
            page_title="Voice Analytics Platform",
            page_icon="🎤📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            text-align: center;
            color: #666;
            font-style: italic;
            margin-bottom: 2rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">🎤 Voice Analytics Platform</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Speak your data queries and get instant visualizations</p>', unsafe_allow_html=True)
        
        # Initialize session state
        if 'commands_history' not in st.session_state:
            st.session_state.commands_history = []
        
        # Sidebar with enhanced information
        with st.sidebar:
            st.header("📊 Dataset Overview")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", f"{len(self.df):,}")
            with col2:
                st.metric("Columns", len(self.df.columns))
            
            # Dataset preview
            with st.expander("Preview Data"):
                st.dataframe(self.df.head())
            
            # Available columns
            st.subheader("Available Columns")
            for col in self.df.columns:
                col_type = str(self.df[col].dtype)
                unique_vals = self.df[col].nunique()
                st.write(f"• **{col}** ({col_type}) - {unique_vals} unique values")
            
            st.divider()
            
            # Session statistics
            st.header("📈 Session Stats")
            success_rate = (self.session_metrics['successful_queries'] / 
                          max(1, self.session_metrics['commands_processed']) * 100)
            
            st.metric("Commands Processed", self.session_metrics['commands_processed'])
            st.metric("Success Rate", f"{success_rate:.1f}%")
            
            if self.session_metrics['commands_processed'] > 0:
                avg_time = (self.session_metrics['total_processing_time'] / 
                           self.session_metrics['commands_processed'])
                st.metric("Avg Response Time", f"{avg_time:.2f}s")
            
            # Example commands
            st.header("💡 Example Commands")
            example_commands = [
                "Show total sales by region",
                "Plot average profit by category as pie chart",
                "Calculate maximum quantity",
                "Display sales by state as bar chart",
                "Show profit distribution by segment"
            ]
            
            for cmd in example_commands:
                if st.button(cmd, key=f"example_{cmd}"):
                    st.session_state['current_command'] = cmd
        
        # Main interface
        st.header("🎯 Query Interface")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Text input with better UX
            text_input = st.text_input(
                "Type your command:",
                placeholder="e.g., 'Show total sales by region as bar chart'",
                help="You can ask for visualizations, calculations, or data filtering"
            )
            
            if st.button("🚀 Process Command", type="primary", use_container_width=True):
                if text_input.strip():
                    st.session_state['current_command'] = text_input.strip()
        
        with col2:
            st.write("**Or use voice:**")
            if st.button("🎤 Voice Input", type="secondary", use_container_width=True):
                with st.spinner("🎧 Listening... Please speak clearly"):
                    command = self.speech.recognize_speech()
                    if command:
                        st.session_state['current_command'] = command
                        st.success(f"Heard: {command}")
                    else:
                        st.warning("No speech detected or recognition failed")
        
        # Process command if available
        if 'current_command' in st.session_state and st.session_state['current_command']:
            command = st.session_state['current_command']
            
            st.subheader("🔄 Processing Command")
            st.info(f"**Command:** {command}")
            
            # Add to history
            if command not in [cmd['text'] for cmd in st.session_state.commands_history[-5:]]:
                st.session_state.commands_history.append({
                    'text': command,
                    'timestamp': time.strftime("%H:%M:%S")
                })
            
            try:
                with st.spinner("Processing your request..."):
                    start_time = time.time()
                    result = self.analyzer.handle_command(command, self.df)
                    processing_time = time.time() - start_time
                
                # Update metrics
                self.session_metrics['commands_processed'] += 1
                self.session_metrics['total_processing_time'] += processing_time
                
                if result['success']:
                    self.session_metrics['successful_queries'] += 1
                    
                    st.success(f"✅ {result['message']} (Processed in {processing_time:.2f}s)")
                    
                    if result['type'] == 'visualization':
                        st.plotly_chart(result['figure'], use_container_width=True)
                    elif result['type'] == 'calculation':
                        st.metric("Result", result['value'])
                    elif result['type'] == 'data':
                        st.subheader("📋 Filtered Results")
                        st.dataframe(result['data'], use_container_width=True)
                        st.info(f"Showing {len(result['data'])} rows")
                else:
                    st.error(f"❌ {result['message']}")
                    
                    # Provide helpful suggestions
                    st.subheader("💡 Suggestions")
                    suggestions = [
                        "Try specifying both a metric (sales, profit, quantity) and dimension (region, category)",
                        "Use words like 'show', 'plot', or 'display' for visualizations",
                        "For calculations, try 'calculate total sales' or 'find average profit'"
                    ]
                    for suggestion in suggestions:
                        st.write(f"• {suggestion}")
            
            except Exception as e:
                logger.error(f"Unexpected error processing command: {e}")
                st.error("An unexpected error occurred. Please try again or check the logs.")
            
            # Clear the command
            del st.session_state['current_command']
        
        # Command history
        if st.session_state.commands_history:
            st.subheader("📜 Recent Commands")
            for i, cmd in enumerate(reversed(st.session_state.commands_history[-10:])):
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"{cmd['text']}", key=f"history_{i}"):
                        st.session_state['current_command'] = cmd['text']
                        st.rerun()
                with col2:
                    st.write(f"*{cmd['timestamp']}*")
    
    def run_cli(self):
        """Enhanced CLI mode with better user experience"""
        print("🎤 Voice-Controlled Data Analytics Platform")
        print("=" * 60)
        print(f"📊 Dataset loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
        print(f"📋 Available columns: {', '.join(self.df.columns)}")
        print("\n💡 Example commands:")
        print("  • 'Show total sales by region'")
        print("  • 'Plot average profit by category as pie chart'")
        print("  • 'Calculate maximum quantity'")
        print("  • 'Filter data where region is East'")
        print("\n🎯 Say 'stop', 'exit', or press Ctrl+C to quit\n")
        
        while True:
            try:
                print("\n🎤 Listening for your command...")
                command = self.speech.recognize_speech()
                
                if command:
                    print(f"📝 Processing: '{command}'")
                    start_time = time.time()
                    
                    result = self.analyzer.handle_command(command, self.df)
                    processing_time = time.time() - start_time
                    
                    # Update metrics
                    self.session_metrics['commands_processed'] += 1
                    self.session_metrics['total_processing_time'] += processing_time
                    
                    if result['success']:
                        self.session_metrics['successful_queries'] += 1
                        print(f"✅ {result['message']} (⏱️ {processing_time:.2f}s)")
                        
                        if result['type'] == 'visualization':
                            result['figure'].show()
                        elif result['type'] == 'calculation':
                            print(f"🔢 Result: {result['value']}")
                        elif result['type'] == 'data':
                            print(f"📋 Filtered Data ({len(result['data'])} rows):")
                            print(result['data'].to_string())
                    else:
                        print(f"❌ {result['message']}")
                        print("💡 Try commands like 'show sales by region' or 'calculate total profit'")
                else:
                    print("🔇 No speech detected or recognition failed")
                    
            except KeyboardInterrupt:
                # Print session summary
                session_time = time.time() - self.session_metrics['session_start']
                success_rate = (self.session_metrics['successful_queries'] / 
                               max(1, self.session_metrics['commands_processed']) * 100)
                
                print(f"\n📊 Session Summary:")
                print(f"   ⏱️  Duration: {session_time:.1f} seconds")
                print(f"   🎯 Commands processed: {self.session_metrics['commands_processed']}")
                print(f"   ✅ Success rate: {success_rate:.1f}%")
                print(f"   👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error in CLI: {e}")
                print("❌ An unexpected error occurred. Continuing...")
                continue

def main():
    """Main entry point with argument parsing"""
    app = VoiceAnalyticsApp()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        logger.info("Starting Streamlit web interface")
        app.run_streamlit_app()
    else:
        logger.info("Starting CLI interface")
        app.run_cli()

if __name__ == "__main__":
    main()