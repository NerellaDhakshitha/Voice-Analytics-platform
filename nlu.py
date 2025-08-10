"""
Advanced Natural Language Understanding (NLU) Module
Enhanced intent detection and entity extraction with improved accuracy and flexibility
"""
import spacy
from spacy.matcher import Matcher, PhraseMatcher
import re
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class NLUProcessor:
    """
    Advanced NLP processor with enhanced intent detection and entity extraction
    Features: Multi-pattern matching, fuzzy matching, context understanding
    """
    
    def __init__(self):
        self.nlp = self._load_spacy_model()
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        # Enhanced vocabulary mappings
        self.column_mapping = {
            # Metrics (numerical columns)
            "sales": "Sales", "sale": "Sales", "revenue": "Sales", 
            "profit": "Profit", "profits": "Profit", "earnings": "Profit",
            "quantity": "Quantity", "qty": "Quantity", "amount": "Quantity", "units": "Quantity",
            "discount": "Discount", "discounts": "Discount",
            "price": "Price", "cost": "Cost", "value": "Value",
            
            # Dimensions (categorical columns)
            "region": "Region", "regions": "Region", "area": "Region", "zone": "Region",
            "category": "Category", "categories": "Category", "type": "Category", "kind": "Category",
            "segment": "Segment", "segments": "Segment", "market": "Segment",
            "state": "State", "states": "State", "location": "State", "place": "State",
            "city": "City", "cities": "City",
            "country": "Country", "countries": "Country", "nation": "Country",
            "customer": "Customer", "customers": "Customer", "client": "Customer",
            "product": "Product", "products": "Product", "item": "Product",
            
            # Aggregation functions
            "total": "sum", "sum": "sum", "add": "sum", "addition": "sum",
            "average": "mean", "mean": "mean", "avg": "mean", "typical": "mean",
            "maximum": "max", "max": "max", "highest": "max", "peak": "max", "top": "max",
            "minimum": "min", "min": "min", "lowest": "min", "bottom": "min", "least": "min",
            "count": "count", "number": "count", "quantity": "count", "how many": "count",
            "distinct": "nunique", "unique": "nunique", "different": "nunique",
            "median": "median", "middle": "median", "mid": "median",
            "standard deviation": "std", "std": "std", "deviation": "std", "variance": "std",
            
            # Chart types
            "bar": "bar", "bar chart": "bar", "bar graph": "bar", "column": "bar", "columns": "bar",
            "line": "line", "line chart": "line", "line graph": "line", "trend": "line",
            "pie": "pie", "pie chart": "pie", "donut": "pie", "circle": "pie",
            "scatter": "scatter", "scatter plot": "scatter", "scatter chart": "scatter", "point": "scatter",
            "histogram": "hist", "hist": "hist", "distribution": "hist", "frequency": "hist"
        }
        
        # Intent patterns using spaCy's pattern matching
        self._setup_intent_patterns()
        
        # Statistical terms for better recognition
        self.stats_terms = {
            "describe": "describe_data", "summary": "describe_data", "statistics": "describe_data",
            "stats": "describe_data", "overview": "describe_data", "info": "describe_data",
            "compare": "compare_data", "comparison": "compare_data", "versus": "compare_data",
            "vs": "compare_data", "against": "compare_data", "between": "compare_data"
        }
        
        # Performance tracking
        self.processing_stats = {
            'total_queries': 0,
            'successful_extractions': 0,
            'failed_extractions': 0
        }
    
    def _load_spacy_model(self):
        """Load spaCy model with error handling and auto-download"""
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully")
            return nlp
        except OSError:
            logger.warning("SpaCy model not found, attempting to download...")
            try:
                spacy.cli.download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
                logger.info("SpaCy model downloaded and loaded successfully")
                return nlp
            except Exception as e:
                logger.error(f"Failed to download SpaCy model: {e}")
                # Fallback to basic processing
                return None
    
    def _setup_intent_patterns(self):
        """Setup advanced pattern matching for intent detection"""
        
        # Visualization patterns
        visualization_patterns = [
            [{"LOWER": {"IN": ["show", "display", "plot", "draw", "chart", "graph", "visualize"]}},
             {"IS_ALPHA": True, "OP": "*"},
             {"LOWER": {"IN": ["by", "per", "across", "for"]}, "OP": "?"},
             {"IS_ALPHA": True, "OP": "*"}],
            
            [{"LOWER": {"IN": ["create", "make", "generate", "build"]}},
             {"IS_ALPHA": True, "OP": "*"},
             {"LOWER": {"IN": ["chart", "graph", "plot", "visualization"]}},
             {"IS_ALPHA": True, "OP": "*"}]
        ]
        
        # Calculation patterns
        calculation_patterns = [
            [{"LOWER": {"IN": ["calculate", "compute", "find", "get", "what", "how"]}},
             {"LOWER": {"IN": ["is", "are", "much", "many"]}, "OP": "?"},
             {"LOWER": {"IN": ["the", "total", "sum", "average", "max", "min"]}},
             {"IS_ALPHA": True, "OP": "+"}],
            
            [{"LOWER": {"IN": ["sum", "total", "add"]}},
             {"LOWER": {"IN": ["of", "up"]}, "OP": "?"},
             {"IS_ALPHA": True, "OP": "+"}]
        ]
        
        # Filter patterns
        filter_patterns = [
            [{"LOWER": {"IN": ["filter", "where", "select", "find"]}},
             {"IS_ALPHA": True, "OP": "*"},
             {"LOWER": {"IN": ["where", "by", "is", "equals", "="]}},
             {"IS_ALPHA": True, "OP": "+"}]
        ]
        
        # Sort patterns
        sort_patterns = [
            [{"LOWER": {"IN": ["sort", "order", "arrange", "rank"]}},
             {"IS_ALPHA": True, "OP": "*"},
             {"LOWER": {"IN": ["by", "on"]}, "OP": "?"},
             {"IS_ALPHA": True, "OP": "+"}]
        ]
        
        # Add patterns to matcher
        self.matcher.add("VISUALIZATION", visualization_patterns)
        self.matcher.add("CALCULATION", calculation_patterns)
        self.matcher.add("FILTER", filter_patterns)
        self.matcher.add("SORT", sort_patterns)
    
    def extract_intent_and_entities(self, text: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Enhanced intent and entity extraction with improved accuracy
        Returns: (intent, entities_dict)
        """
        try:
            self.processing_stats['total_queries'] += 1
            
            # Preprocess text
            text_clean = self._preprocess_text(text)
            doc = self.nlp(text_clean.lower()) if self.nlp else None
            
            # Initialize entities structure
            entities = {
                "metric": [],
                "dimension": [],
                "aggregation_type": None,
                "chart_type": None,
                "filter_column": None,
                "filter_value": None,
                "sort_column": None,
                "sort_order": "ascending"
            }
            
            # Extract entities using multiple methods
            self._extract_entities_by_mapping(text_clean, entities)
            
            if doc:
                self._extract_entities_by_pos(doc, entities)
                self._extract_entities_by_dependencies(doc, entities)
            
            # Detect intent using pattern matching and keywords
            intent = self._detect_intent_advanced(text_clean, entities, doc)
            
            # Post-process and validate entities
            self._post_process_entities(entities, intent)
            
            # Log successful extraction
            if intent:
                self.processing_stats['successful_extractions'] += 1
                logger.debug(f"Extracted intent: {intent}, entities: {entities}")
            else:
                self.processing_stats['failed_extractions'] += 1
                logger.warning(f"Failed to extract intent from: {text}")
            
            return intent, entities
            
        except Exception as e:
            self.processing_stats['failed_extractions'] += 1
            logger.error(f"Error in NLU processing: {e}")
            return None, {
                "metric": [], "dimension": [], "aggregation_type": None,
                "chart_type": None, "filter_column": None, "filter_value": None,
                "sort_column": None, "sort_order": "ascending"
            }
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize input text"""
        # Handle common contractions and variations
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"show me", "show", text)
        text = re.sub(r"can you", "", text)
        text = re.sub(r"please", "", text)
        text = re.sub(r"i want to", "", text)
        text = re.sub(r"i need", "", text)
        
        # Normalize chart type mentions
        text = re.sub(r"as a (.+) chart", r"as \1", text)
        text = re.sub(r"in (.+) format", r"as \1", text)
        
        # Handle multi-word phrases
        text = re.sub(r"bar chart", "bar", text)
        text = re.sub(r"pie chart", "pie", text)
        text = re.sub(r"line chart", "line", text)
        
        return text.strip()
    
    def _extract_entities_by_mapping(self, text: str, entities: Dict):
        """Extract entities using direct vocabulary mapping"""
        text_lower = text.lower()
        
        # Extract using exact phrase matching first
        for phrase, mapped_value in self.column_mapping.items():
            if phrase in text_lower:
                if mapped_value in ["Sales", "Profit", "Quantity", "Discount", "Price", "Cost", "Value"]:
                    if mapped_value not in entities["metric"]:
                        entities["metric"].append(mapped_value)
                elif mapped_value in ["Region", "Category", "Segment", "State", "City", "Country", "Customer", "Product"]:
                    if mapped_value not in entities["dimension"]:
                        entities["dimension"].append(mapped_value)
                elif mapped_value in ["sum", "mean", "max", "min", "count", "nunique", "median", "std"]:
                    if entities["aggregation_type"] is None:
                        entities["aggregation_type"] = mapped_value
                elif mapped_value in ["bar", "line", "pie", "scatter", "hist"]:
                    entities["chart_type"] = mapped_value
        
        # Extract statistical operations
        for term, intent_type in self.stats_terms.items():
            if term in text_lower:
                entities["_special_intent"] = intent_type
    
    def _extract_entities_by_pos(self, doc, entities: Dict):
        """Extract entities using Part-of-Speech tagging"""
        if not doc:
            return
            
        for token in doc:
            # Look for potential column names (proper nouns, nouns)
            if token.pos_ in ["PROPN", "NOUN"] and len(token.text) > 2:
                token_title = token.text.title()
                
                # Check if it matches known columns
                if token_title in ["Sales", "Profit", "Quantity", "Region", "Category", "Segment", "State"]:
                    if token_title in ["Sales", "Profit", "Quantity"]:
                        if token_title not in entities["metric"]:
                            entities["metric"].append(token_title)
                    else:
                        if token_title not in entities["dimension"]:
                            entities["dimension"].append(token_title)
            
            # Extract numerical values for filtering
            if token.like_num or token.pos_ == "NUM":
                entities["_numeric_value"] = token.text
    
    def _extract_entities_by_dependencies(self, doc, entities: Dict):
        """Extract entities using dependency parsing"""
        if not doc:
            return
            
        # Look for filter patterns using dependency relations
        for token in doc:
            if token.lemma_ in ["filter", "where", "select"]:
                # Look for objects of filtering
                for child in token.children:
                    if child.dep_ in ["dobj", "pobj", "attr"]:
                        child_title = child.text.title()
                        if child_title in ["Region", "Category", "Segment", "State"]:
                            entities["filter_column"] = child_title
            
            # Look for sorting patterns
            if token.lemma_ in ["sort", "order"]:
                for child in token.children:
                    if child.dep_ in ["agent", "pobj"]:
                        child_title = child.text.title()
                        if child_title in ["Sales", "Profit", "Quantity"]:
                            entities["sort_column"] = child_title
            
            # Detect descending order
            if token.lemma_ in ["descending", "desc", "down", "reverse"]:
                entities["sort_order"] = "descending"
    
    def _detect_intent_advanced(self, text: str, entities: Dict, doc) -> Optional[str]:
        """Advanced intent detection using multiple strategies"""
        
        # Check for special statistical intents
        if entities.get("_special_intent"):
            return entities["_special_intent"]
        
        # Use pattern matching if available
        if doc and self.matcher:
            matches = self.matcher(doc)
            if matches:
                match_id, start, end = matches[0]
                intent_label = self.nlp.vocab.strings[match_id]
                
                if intent_label == "VISUALIZATION":
                    return "show_data"
                elif intent_label == "CALCULATION":
                    return "calculate_data"
                elif intent_label == "FILTER":
                    return "filter_data"
                elif intent_label == "SORT":
                    return "sort_data"
        
        # Fallback to keyword-based intent detection
        text_lower = text.lower()
        
        # Visualization intent keywords
        viz_keywords = ["show", "display", "plot", "chart", "graph", "visualize", "draw", "create"]
        if any(keyword in text_lower for keyword in viz_keywords):
            if entities["metric"] and entities["dimension"]:
                return "show_data"
        
        # Calculation intent keywords  
        calc_keywords = ["calculate", "compute", "find", "get", "total", "sum", "average", "max", "min", "count"]
        if any(keyword in text_lower for keyword in calc_keywords):
            if entities["aggregation_type"] and entities["metric"]:
                return "calculate_data"
        
        # Filter intent keywords
        filter_keywords = ["filter", "where", "select", "find records", "show records"]
        if any(keyword in text_lower for keyword in filter_keywords):
            return "filter_data"
        
        # Sort intent keywords
        sort_keywords = ["sort", "order", "arrange", "rank"]
        if any(keyword in text_lower for keyword in sort_keywords):
            return "sort_data"
        
        # Heuristic-based intent detection
        if entities["metric"] and entities["dimension"]:
            # If we have both metric and dimension, likely visualization
            if entities["chart_type"] or any(viz in text_lower for viz in ["by", "across", "per"]):
                return "show_data"
            elif entities["aggregation_type"]:
                return "calculate_data"
        
        elif entities["aggregation_type"] and entities["metric"]:
            # Clear calculation intent
            return "calculate_data"
        
        elif entities["filter_column"] or "where" in text_lower:
            # Clear filtering intent
            return "filter_data"
        
        elif entities["sort_column"] or any(sort in text_lower for sort in ["sort", "order"]):
            # Clear sorting intent
            return "sort_data"
        
        # Default fallback - try to infer from available entities
        if entities["metric"] and not entities["dimension"]:
            return "calculate_data"  # Likely wants to calculate something
        elif entities["dimension"] and not entities["metric"]:
            return "show_data"  # Likely wants to see data breakdown
        
        return None
    
    def _post_process_entities(self, entities: Dict, intent: Optional[str]):
        """Post-process and validate extracted entities"""
        
        # Set default aggregation for visualizations if missing
        if intent == "show_data" and not entities["aggregation_type"]:
            if entities["metric"]:
                entities["aggregation_type"] = "sum"  # Default to sum for visualizations
        
        # Handle filter value extraction
        if intent == "filter_data" and entities["filter_column"] and not entities["filter_value"]:
            # Try to extract filter value from numeric values found
            if entities.get("_numeric_value"):
                entities["filter_value"] = entities["_numeric_value"]
        
        # Clean up temporary entities
        entities.pop("_special_intent", None)
        entities.pop("_numeric_value", None)
        
        # Validate combinations
        if intent == "show_data":
            if not entities["metric"]:
                # Try to infer metric from context
                common_metrics = ["Sales", "Profit", "Quantity"]
                entities["metric"] = common_metrics[:1]  # Default to Sales
            
            if not entities["dimension"]:
                # Try to infer dimension from context
                common_dimensions = ["Region", "Category", "Segment"]
                entities["dimension"] = common_dimensions[:1]  # Default to Region
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Return current processing statistics"""
        success_rate = 0.0
        if self.processing_stats['total_queries'] > 0:
            success_rate = (self.processing_stats['successful_extractions'] / 
                          self.processing_stats['total_queries']) * 100
        
        return {
            **self.processing_stats,
            'success_rate': round(success_rate, 2),
            'model_loaded': self.nlp is not None
        }
    
    def add_custom_mapping(self, term: str, mapped_value: str, category: str):
        """Add custom term mapping for domain-specific vocabulary"""
        if category == "metric":
            # Validate that it's a metric-type mapping
            self.column_mapping[term.lower()] = mapped_value
        elif category == "dimension":
            self.column_mapping[term.lower()] = mapped_value
        elif category == "aggregation":
            self.column_mapping[term.lower()] = mapped_value
        elif category == "chart":
            self.column_mapping[term.lower()] = mapped_value
        
        logger.info(f"Added custom mapping: {term} -> {mapped_value} ({category})")
    
    def fuzzy_match_column(self, term: str, available_columns: List[str], threshold: float = 0.7) -> Optional[str]:
        """Fuzzy matching for column names when exact match fails"""
        try:
            from difflib import SequenceMatcher
            
            best_match = None
            best_score = 0.0
            
            term_lower = term.lower()
            
            for col in available_columns:
                # Direct substring match gets high priority
                if term_lower in col.lower() or col.lower() in term_lower:
                    if len(term_lower) > 2:  # Avoid matching very short terms
                        return col
                
                # Sequence matching
                score = SequenceMatcher(None, term_lower, col.lower()).ratio()
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = col
            
            if best_match:
                logger.info(f"Fuzzy matched '{term}' to '{best_match}' (score: {best_score:.2f})")
            
            return best_match
            
        except ImportError:
            logger.warning("difflib not available for fuzzy matching")
            return None
    
    def explain_extraction(self, text: str) -> Dict[str, Any]:
        """Provide detailed explanation of the extraction process for debugging"""
        intent, entities = self.extract_intent_and_entities(text)
        
        explanation = {
            'input_text': text,
            'detected_intent': intent,
            'extracted_entities': entities,
            'processing_steps': [],
            'confidence_scores': {}
        }
        
        # Add processing steps explanation
        if self.nlp:
            doc = self.nlp(text.lower())
            
            # POS tagging results
            pos_info = [(token.text, token.pos_, token.lemma_) for token in doc]
            explanation['pos_analysis'] = pos_info
            
            # Dependency parsing results
            dep_info = [(token.text, token.dep_, token.head.text) for token in doc]
            explanation['dependency_analysis'] = dep_info
            
            # Pattern matches
            if self.matcher:
                matches = self.matcher(doc)
                match_info = []
                for match_id, start, end in matches:
                    label = self.nlp.vocab.strings[match_id]
                    span = doc[start:end]
                    match_info.append((label, span.text))
                explanation['pattern_matches'] = match_info
        
        # Confidence scoring
        confidence = self._calculate_confidence(intent, entities)
        explanation['confidence_scores'] = confidence
        
        return explanation
    
    def _calculate_confidence(self, intent: Optional[str], entities: Dict) -> Dict[str, float]:
        """Calculate confidence scores for the extraction"""
        scores = {
            'overall': 0.0,
            'intent': 0.0,
            'entities': 0.0
        }
        
        # Intent confidence
        if intent:
            scores['intent'] = 0.8  # Base confidence
            # Boost if entities support the intent
            if intent == "show_data" and entities["metric"] and entities["dimension"]:
                scores['intent'] = 0.95
            elif intent == "calculate_data" and entities["metric"] and entities["aggregation_type"]:
                scores['intent'] = 0.9
        else:
            scores['intent'] = 0.1
        
        # Entity confidence
        entity_count = 0
        filled_entities = 0
        
        for key, value in entities.items():
            if key.startswith('_'):
                continue
            entity_count += 1
            if value:
                if isinstance(value, list):
                    filled_entities += 1 if value else 0
                else:
                    filled_entities += 1
        
        scores['entities'] = filled_entities / entity_count if entity_count > 0 else 0.0
        
        # Overall confidence
        scores['overall'] = (scores['intent'] + scores['entities']) / 2
        
        return scores