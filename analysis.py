"""
Enhanced Data Analysis Engine
Modern implementation with comprehensive error handling and performance optimization
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from nlu import NLUProcessor

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """
    Advanced data analysis engine with intelligent visualization and robust error handling
    Features: Smart chart selection, performance optimization, comprehensive analytics
    """
    
    def __init__(self):
        self.nlu = NLUProcessor()
        self.supported_aggregations = {
            'sum': 'sum', 'total': 'sum', 'add': 'sum',
            'average': 'mean', 'mean': 'mean', 'avg': 'mean',
            'maximum': 'max', 'max': 'max', 'highest': 'max',
            'minimum': 'min', 'min': 'min', 'lowest': 'min',
            'count': 'count', 'number': 'count',
            'median': 'median', 'middle': 'median',
            'std': 'std', 'deviation': 'std'
        }
        
        self.chart_themes = {
            'professional': 'plotly_white',
            'dark': 'plotly_dark',
            'colorful': 'plotly',
            'minimal': 'simple_white'
        }
        
        self.performance_stats = {
            'total_queries': 0,
            'avg_processing_time': 0.0,
            'cache_hits': 0
        }
    
    def handle_command(self, command: str, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Process user commands with comprehensive error handling and performance tracking
        Returns structured response for better integration with UI components
        """
        start_time = time.time()
        command_lower = command.lower().strip()
        
        try:
            # Extract intent and entities using NLU
            intent, entities = self.nlu.extract_intent_and_entities(command)
            logger.info(f"Command: '{command}' | Intent: {intent} | Entities: {entities}")
            
            # Route to appropriate handler
            if intent == "show_data":
                result = self._handle_visualization(entities, dataframe, command)
            elif intent == "calculate_data":
                result = self._handle_calculation(entities, dataframe)
            elif intent == "filter_data":
                result = self._handle_filtering(entities, dataframe)
            elif intent == "sort_data":
                result = self._handle_sorting(entities, dataframe)
            elif intent == "describe_data":
                result = self._handle_data_description(entities, dataframe)
            elif intent == "compare_data":
                result = self._handle_comparison(entities, dataframe)
            else:
                result = self._handle_unknown_command(command, dataframe)
            
            # Add performance metrics
            processing_time = time.time() - start_time
            result['processing_time'] = round(processing_time, 3)
            result['command'] = command
            
            # Update performance stats
            self.performance_stats['total_queries'] += 1
            self.performance_stats['avg_processing_time'] = (
                (self.performance_stats['avg_processing_time'] * (self.performance_stats['total_queries'] - 1) + 
                 processing_time) / self.performance_stats['total_queries']
            )
            
            logger.info(f"Command processed successfully in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing command '{command}': {str(e)}")
            return {
                'success': False,
                'message': f"Error processing command: {str(e)}",
                'type': 'error',
                'processing_time': round(processing_time, 3),
                'command': command,
                'suggestions': self._get_error_suggestions(str(e))
            }
    
    def _handle_visualization(self, entities: Dict, df: pd.DataFrame, original_command: str) -> Dict[str, Any]:
        """Enhanced visualization with smart defaults and advanced chart options"""
        try:
            metrics = entities.get("metric", [])
            dimensions = entities.get("dimension", [])
            chart_type = entities.get("chart_type")
            agg_func = entities.get("aggregation_type", "sum")
            
            # Input validation
            validation_result = self._validate_visualization_inputs(metrics, dimensions, df)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'message': validation_result['message'],
                    'type': 'error',
                    'suggestions': validation_result['suggestions']
                }
            
            metrics = validation_result['metrics']
            dimensions = validation_result['dimensions']
            
            # Smart chart type selection
            if not chart_type:
                chart_type = self._smart_chart_selection(dimensions, metrics, df)
                logger.info(f"Auto-selected chart type: {chart_type}")
            
            # Data aggregation with performance optimization
            try:
                if len(metrics) == 1:
                    if agg_func in ['count', 'nunique']:
                        grouped_data = df.groupby(dimensions)[metrics[0]].agg(agg_func).reset_index()
                    else:
                        grouped_data = df.groupby(dimensions)[metrics[0]].agg(agg_func).reset_index()
                else:
                    # Multiple metrics
                    grouped_data = df.groupby(dimensions)[metrics].agg(agg_func).reset_index()
                
                # Handle empty results
                if grouped_data.empty:
                    return {
                        'success': False,
                        'message': "No data found matching your criteria",
                        'type': 'error'
                    }
                
            except Exception as e:
                return {
                    'success': False,
                    'message': f"Data aggregation failed: {str(e)}",
                    'type': 'error'
                }
            
            # Create visualization
            fig = self._create_chart(grouped_data, chart_type, metrics, dimensions, agg_func)
            
            if fig is None:
                return {
                    'success': False,
                    'message': f"Failed to create {chart_type} chart",
                    'type': 'error'
                }
            
            # Enhanced chart styling
            self._apply_advanced_styling(fig, chart_type)
            
            return {
                'success': True,
                'figure': fig,
                'type': 'visualization',
                'message': f"Generated {chart_type} chart showing {agg_func} of {', '.join(metrics)} by {', '.join(dimensions)}",
                'data_points': len(grouped_data),
                'chart_type': chart_type
            }
            
        except Exception as e:
            logger.error(f"Visualization error: {str(e)}")
            return {
                'success': False,
                'message': f"Visualization failed: {str(e)}",
                'type': 'error'
            }
    
    def _handle_calculation(self, entities: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced calculation with support for advanced statistical operations"""
        try:
            metrics = entities.get("metric", [])
            agg_func = entities.get("aggregation_type", "sum")
            dimensions = entities.get("dimension", [])
            
            if not metrics:
                return {
                    'success': False,
                    'message': "Please specify a metric to calculate (e.g., sales, profit, quantity)",
                    'type': 'error'
                }
            
            metric = metrics[0]
            
            # Check if metric exists
            if metric not in df.columns:
                similar_cols = [col for col in df.columns if metric.lower() in col.lower()]
                suggestion = f" Did you mean: {', '.join(similar_cols)}?" if similar_cols else ""
                return {
                    'success': False,
                    'message': f"Column '{metric}' not found in dataset.{suggestion}",
                    'type': 'error'
                }
            
            # Perform calculation
            try:
                if dimensions:
                    # Group by dimensions and calculate
                    grouped = df.groupby(dimensions)[metric].agg(agg_func)
                    result_value = grouped.sum() if agg_func in ['sum', 'total'] else grouped.mean()
                    breakdown = grouped.to_dict()
                    
                    return {
                        'success': True,
                        'value': f"{result_value:,.2f}",
                        'type': 'calculation',
                        'message': f"The {agg_func} of {metric} is {result_value:,.2f}",
                        'breakdown': breakdown,
                        'grouped_by': dimensions
                    }
                else:
                    # Simple calculation
                    if agg_func in ['sum', 'total']:
                        result = df[metric].sum()
                    elif agg_func in ['mean', 'average', 'avg']:
                        result = df[metric].mean()
                    elif agg_func in ['max', 'maximum']:
                        result = df[metric].max()
                    elif agg_func in ['min', 'minimum']:
                        result = df[metric].min()
                    elif agg_func == 'median':
                        result = df[metric].median()
                    elif agg_func == 'count':
                        result = df[metric].count()
                    elif agg_func == 'std':
                        result = df[metric].std()
                    else:
                        result = getattr(df[metric], agg_func)()
                    
                    return {
                        'success': True,
                        'value': f"{result:,.2f}" if isinstance(result, (int, float)) else str(result),
                        'type': 'calculation',
                        'message': f"The {agg_func} of {metric} is {result:,.2f}" if isinstance(result, (int, float)) else f"The {agg_func} of {metric} is {result}",
                        'raw_value': result
                    }
                    
            except Exception as e:
                return {
                    'success': False,
                    'message': f"Calculation failed: {str(e)}",
                    'type': 'error'
                }
            
        except Exception as e:
            logger.error(f"Calculation error: {str(e)}")
            return {
                'success': False,
                'message': f"Calculation error: {str(e)}",
                'type': 'error'
            }
    
    def _handle_filtering(self, entities: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced data filtering with multiple conditions support"""
        try:
            filter_column = entities.get("filter_column")
            filter_value = entities.get("filter_value")
            
            if not filter_column or not filter_value:
                return {
                    'success': False,
                    'message': "Please specify both a column and value to filter by (e.g., 'filter by region East')",
                    'type': 'error'
                }
            
            # Check if column exists
            if filter_column not in df.columns:
                similar_cols = [col for col in df.columns if filter_column.lower() in col.lower()]
                suggestion = f" Did you mean: {', '.join(similar_cols)}?" if similar_cols else ""
                return {
                    'success': False,
                    'message': f"Column '{filter_column}' not found.{suggestion}",
                    'type': 'error'
                }
            
            # Apply filter with case-insensitive matching
            try:
                if df[filter_column].dtype == 'object':
                    # String filtering - case insensitive
                    mask = df[filter_column].astype(str).str.lower().str.contains(filter_value.lower(), na=False)
                else:
                    # Numeric filtering - exact match or range
                    try:
                        numeric_value = float(filter_value)
                        mask = df[filter_column] == numeric_value
                    except ValueError:
                        mask = df[filter_column].astype(str).str.lower() == filter_value.lower()
                
                filtered_df = df[mask]
                
                if filtered_df.empty:
                    return {
                        'success': False,
                        'message': f"No records found where {filter_column} matches '{filter_value}'",
                        'type': 'error'
                    }
                
                return {
                    'success': True,
                    'data': filtered_df.head(100),  # Limit to first 100 rows for display
                    'type': 'data',
                    'message': f"Found {len(filtered_df)} records where {filter_column} contains '{filter_value}'",
                    'total_rows': len(filtered_df),
                    'showing_rows': min(100, len(filtered_df))
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'message': f"Filtering failed: {str(e)}",
                    'type': 'error'
                }
            
        except Exception as e:
            logger.error(f"Filtering error: {str(e)}")
            return {
                'success': False,
                'message': f"Filtering error: {str(e)}",
                'type': 'error'
            }
    
    def _handle_sorting(self, entities: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced data sorting with multiple column support"""
        try:
            sort_column = entities.get("sort_column")
            sort_order = entities.get("sort_order", "ascending")
            
            if not sort_column:
                return {
                    'success': False,
                    'message': "Please specify a column to sort by (e.g., 'sort by sales')",
                    'type': 'error'
                }
            
            # Check if column exists
            if sort_column not in df.columns:
                similar_cols = [col for col in df.columns if sort_column.lower() in col.lower()]
                suggestion = f" Did you mean: {', '.join(similar_cols)}?" if similar_cols else ""
                return {
                    'success': False,
                    'message': f"Column '{sort_column}' not found.{suggestion}",
                    'type': 'error'
                }
            
            try:
                ascending = sort_order.lower() != "descending"
                sorted_df = df.sort_values(by=sort_column, ascending=ascending)
                
                return {
                    'success': True,
                    'data': sorted_df.head(50),  # Show top 50 rows
                    'type': 'data',
                    'message': f"Data sorted by {sort_column} in {sort_order} order",
                    'sort_column': sort_column,
                    'sort_order': sort_order,
                    'total_rows': len(sorted_df),
                    'showing_rows': min(50, len(sorted_df))
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'message': f"Sorting failed: {str(e)}",
                    'type': 'error'
                }
            
        except Exception as e:
            logger.error(f"Sorting error: {str(e)}")
            return {
                'success': False,
                'message': f"Sorting error: {str(e)}",
                'type': 'error'
            }
    
    def _handle_data_description(self, entities: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Provide statistical description of data"""
        try:
            metrics = entities.get("metric", [])
            
            if metrics:
                # Describe specific columns
                metric = metrics[0]
                if metric not in df.columns:
                    return {
                        'success': False,
                        'message': f"Column '{metric}' not found",
                        'type': 'error'
                    }
                
                stats = df[metric].describe()
                return {
                    'success': True,
                    'data': stats.to_frame().T,
                    'type': 'data',
                    'message': f"Statistical summary for {metric}",
                    'stats': stats.to_dict()
                }
            else:
                # Describe entire dataset
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) == 0:
                    return {
                        'success': False,
                        'message': "No numeric columns found for statistical description",
                        'type': 'error'
                    }
                
                stats = df[numeric_cols].describe()
                return {
                    'success': True,
                    'data': stats,
                    'type': 'data',
                    'message': f"Statistical summary for all numeric columns"
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"Description error: {str(e)}",
                'type': 'error'
            }
    
    def _handle_comparison(self, entities: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Handle comparison queries between different segments"""
        try:
            metrics = entities.get("metric", [])
            dimensions = entities.get("dimension", [])
            
            if not metrics or not dimensions:
                return {
                    'success': False,
                    'message': "Please specify metrics and dimensions to compare",
                    'type': 'error'
                }
            
            metric = metrics[0]
            dimension = dimensions[0]
            
            # Create comparison visualization
            grouped_data = df.groupby(dimension)[metric].agg(['sum', 'mean', 'count']).reset_index()
            
            # Create subplot with multiple metrics
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Total', 'Average', 'Count'),
                specs=[[{'secondary_y': False}, {'secondary_y': False}, {'secondary_y': False}]]
            )
            
            fig.add_trace(
                go.Bar(x=grouped_data[dimension], y=grouped_data['sum'], name='Total'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=grouped_data[dimension], y=grouped_data['mean'], name='Average'),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=grouped_data[dimension], y=grouped_data['count'], name='Count'),
                row=1, col=3
            )
            
            fig.update_layout(
                title=f"Comparison of {metric} across {dimension}",
                showlegend=False,
                template='plotly_white'
            )
            
            return {
                'success': True,
                'figure': fig,
                'type': 'visualization',
                'message': f"Generated comparison chart for {metric} by {dimension}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Comparison error: {str(e)}",
                'type': 'error'
            }
    
    def _handle_unknown_command(self, command: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Handle unrecognized commands with intelligent suggestions"""
        suggestions = []
        
        # Analyze the command for potential intent
        command_lower = command.lower()
        
        if any(word in command_lower for word in ['show', 'display', 'plot', 'chart', 'graph']):
            suggestions.append("For visualizations, try: 'Show total sales by region as bar chart'")
        
        if any(word in command_lower for word in ['calculate', 'sum', 'total', 'average', 'max', 'min']):
            suggestions.append("For calculations, try: 'Calculate total profit' or 'Find average sales'")
        
        if any(word in command_lower for word in ['filter', 'where', 'find']):
            suggestions.append("For filtering, try: 'Filter data where region is East'")
        
        # Check for column name mentions
        mentioned_cols = [col for col in df.columns if col.lower() in command_lower]
        if mentioned_cols:
            suggestions.append(f"I detected column(s): {', '.join(mentioned_cols)}. Try using them in your query.")
        
        if not suggestions:
            suggestions = [
                "Try: 'Show sales by region'",
                "Try: 'Calculate total profit'",
                "Try: 'Filter by category Furniture'"
            ]
        
        return {
            'success': False,
            'message': "I couldn't understand your command. Here are some suggestions:",
            'type': 'error',
            'suggestions': suggestions,
            'available_columns': list(df.columns)
        }
    
    def _validate_visualization_inputs(self, metrics: List, dimensions: List, df: pd.DataFrame) -> Dict:
        """Validate and correct visualization inputs"""
        validated_metrics = []
        validated_dimensions = []
        suggestions = []
        
        # Validate metrics
        if not metrics:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            suggestions.append(f"Available numeric columns: {', '.join(numeric_cols[:5])}")
            return {
                'valid': False,
                'message': "Please specify a metric (numeric column) for visualization",
                'suggestions': suggestions
            }
        
        for metric in metrics:
            if metric in df.columns:
                validated_metrics.append(metric)
            else:
                # Try to find similar column names
                similar = [col for col in df.columns if metric.lower() in col.lower()]
                if similar:
                    suggestions.append(f"Did you mean '{similar[0]}' instead of '{metric}'?")
                    validated_metrics.append(similar[0])
                else:
                    return {
                        'valid': False,
                        'message': f"Column '{metric}' not found in dataset",
                        'suggestions': [f"Available columns: {', '.join(df.columns)}"]
                    }
        
        # Validate dimensions
        if not dimensions:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            suggestions.append(f"Available categorical columns: {', '.join(categorical_cols[:5])}")
            return {
                'valid': False,
                'message': "Please specify a dimension (categorical column) for grouping",
                'suggestions': suggestions
            }
        
        for dimension in dimensions:
            if dimension in df.columns:
                validated_dimensions.append(dimension)
            else:
                similar = [col for col in df.columns if dimension.lower() in col.lower()]
                if similar:
                    suggestions.append(f"Did you mean '{similar[0]}' instead of '{dimension}'?")
                    validated_dimensions.append(similar[0])
                else:
                    return {
                        'valid': False,
                        'message': f"Column '{dimension}' not found in dataset",
                        'suggestions': [f"Available columns: {', '.join(df.columns)}"]
                    }
        
        return {
            'valid': True,
            'metrics': validated_metrics,
            'dimensions': validated_dimensions,
            'suggestions': suggestions
        }
    
    def _smart_chart_selection(self, dimensions: List, metrics: List, df: pd.DataFrame) -> str:
        """Intelligently select appropriate chart type based on data characteristics"""
        if len(dimensions) == 1:
            unique_vals = df[dimensions[0]].nunique()
            
            # Check if dimension is datetime
            if df[dimensions[0]].dtype in ['datetime64[ns]', 'datetime']:
                return "line"
            
            # For categorical data
            if unique_vals <= 8:
                return "pie"
            elif unique_vals <= 20:
                return "bar"
            else:
                return "hist"
        
        elif len(dimensions) == 2:
            # Two dimensions - use grouped bar or heatmap
            if df[dimensions[0]].nunique() <= 10 and df[dimensions[1]].nunique() <= 10:
                return "bar"  # Grouped bar chart
            else:
                return "scatter"
        
        else:
            # Multiple dimensions - default to bar
            return "bar"
    
    def _create_chart(self, data: pd.DataFrame, chart_type: str, metrics: List, dimensions: List, agg_func: str) -> Optional[go.Figure]:
        """Create chart with enhanced styling and interactivity"""
        try:
            title = f"{agg_func.title()} of {', '.join(metrics)} by {', '.join(dimensions)}"
            
            if chart_type == "bar":
                if len(dimensions) == 1:
                    fig = px.bar(
                        data, 
                        x=dimensions[0], 
                        y=metrics[0],
                        title=title,
                        template="plotly_white"
                    )
                else:
                    fig = px.bar(
                        data, 
                        x=dimensions[0], 
                        y=metrics[0],
                        color=dimensions[1],
                        title=title,
                        template="plotly_white"
                    )
            
            elif chart_type == "pie":
                if len(dimensions) > 1:
                    return None  # Pie charts don't support multiple dimensions
                
                fig = px.pie(
                    data, 
                    names=dimensions[0], 
                    values=metrics[0],
                    title=title
                )
            
            elif chart_type == "line":
                fig = px.line(
                    data, 
                    x=dimensions[0], 
                    y=metrics[0],
                    color=dimensions[1] if len(dimensions) > 1 else None,
                    title=title,
                    template="plotly_white",
                    markers=True
                )
            
            elif chart_type == "scatter":
                if len(dimensions) >= 2:
                    fig = px.scatter(
                        data, 
                        x=dimensions[0], 
                        y=metrics[0],
                        color=dimensions[1],
                        size=metrics[0],
                        title=title,
                        template="plotly_white"
                    )
                else:
                    fig = px.scatter(
                        data, 
                        x=dimensions[0], 
                        y=metrics[0],
                        title=title,
                        template="plotly_white"
                    )
            
            elif chart_type == "hist":
                fig = px.histogram(
                    data, 
                    x=metrics[0],
                    title=f"Distribution of {metrics[0]}",
                    template="plotly_white"
                )
            
            else:
                return None
            
            return fig
            
        except Exception as e:
            logger.error(f"Chart creation error: {str(e)}")
            return None
    
    def _apply_advanced_styling(self, fig: go.Figure, chart_type: str):
        """Apply advanced styling and interactivity to charts"""
        # Enhanced layout
        fig.update_layout(
            font=dict(size=12, family="Arial, sans-serif"),
            showlegend=True,
            height=500,
            hovermode='x unified' if chart_type in ['bar', 'line'] else 'closest',
            margin=dict(t=80, b=60, l=60, r=40)
        )
        
        # Add hover information
        if chart_type == "bar":
            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>Value: %{y:,.0f}<extra></extra>'
            )
        elif chart_type == "pie":
            fig.update_traces(
                hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
            )
        
        # Add subtle animations
        if chart_type in ["bar", "line"]:
            fig.update_layout(transition_duration=500)
    
    def _get_error_suggestions(self, error_message: str) -> List[str]:
        """Generate helpful suggestions based on error type"""
        suggestions = []
        
        if "not found" in error_message.lower():
            suggestions.append("Check column names - they are case sensitive")
            suggestions.append("Use 'describe data' to see available columns")
        
        elif "aggregation" in error_message.lower():
            suggestions.append("Try aggregations like: sum, average, max, min, count")
        
        elif "chart" in error_message.lower():
            suggestions.append("Supported chart types: bar, pie, line, scatter, histogram")
        
        else:
            suggestions.extend([
                "Try: 'Show sales by region'",
                "Try: 'Calculate total profit'",
                "Try: 'Filter by category Furniture'"
            ])
        
        return suggestions

    def get_performance_stats(self) -> Dict[str, Any]:
        """Return current performance statistics"""
        return self.performance_stats.copy()