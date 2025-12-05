"""
LangGraph workflow agent for biomedical knowledge graphs.

5-step workflow: Classify → Extract → Generate → Execute → Format
"""

import json
import os
from typing import Any, Dict, List, Optional, TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from .graph_interface import GraphInterface
from .workflow_history import WorkflowHistory


class WorkflowState(TypedDict):
    """State that flows through the workflow steps."""

    user_question: str
    conversation_history: Optional[List[Dict]]
    question_type: Optional[str]
    entities: Optional[List[str]]
    cypher_query: Optional[str]
    results: Optional[List[Dict]]
    final_answer: Optional[str]
    error: Optional[str]


class WorkflowAgent:
    """LangGraph workflow agent for biomedical knowledge graphs."""

    # Class constants
    MODEL_NAME = "claude-sonnet-4-20250514"
    DEFAULT_MAX_TOKENS = 200

    # Default schema query
    SCHEMA_QUERY = (
        "MATCH (n) RETURN labels(n) as node_type, count(n) as count "
        "ORDER BY count DESC LIMIT 10"
    )

    def __init__(self, graph_interface: GraphInterface, anthropic_api_key: str):
        self.graph_db = graph_interface
        self.anthropic = Anthropic(api_key=anthropic_api_key)
        self.schema = self.graph_db.get_schema_info()
        self.property_values = self._get_key_property_values()
        self.workflow = self._create_workflow()
        #memory addition
        self.history = WorkflowHistory()


    def _get_key_property_values(self) -> Dict[str, List[Any]]:
        """Get property values dynamically from all nodes and relationships.

        This method discovers all available properties in the database schema and
        collects sample values for each property. This enables the LLM to generate
        more accurate queries by knowing what property values actually exist.

        Returns:
            Dict mapping property names to lists of sample values from the database
        """
        values = {}
        try:
            # Discover and collect property values from all node types in the database
            # This replaces hardcoded property lists with dynamic schema discovery
            for node_label in self.schema.get("node_labels", []):
                # Get all properties that exist for this node type
                node_props = self.schema.get("node_properties", {}).get(node_label, [])
                for prop_name in node_props:
                    # Avoid duplicate property names (same property might exist
                    # on multiple node types)
                    if prop_name not in values:
                        # Query the database for actual property values
                        # (limited to 20 for performance)
                        prop_values = self.graph_db.get_property_values(
                            node_label, prop_name
                        )
                        # Only store properties that have actual values in the database
                        if prop_values:
                            values[prop_name] = prop_values

            # Discover and collect property values from all relationship types
            # This ensures we capture relationship-specific properties like
            # confidence, weight, etc.
            for rel_type in self.schema.get("relationship_types", []):
                # GraphInterface expects 'REL_' prefix for relationship queries
                rel_label = f"REL_{rel_type}"
                # Get all properties that exist for this relationship type
                rel_props = self.schema.get("relationship_properties", {}).get(
                    rel_type, []
                )
                for prop_name in rel_props:
                    # Skip if we already have this property from a node type
                    if prop_name not in values:
                        try:
                            # Query relationship properties using the REL_ prefix
                            # convention
                            prop_values = self.graph_db.get_property_values(
                                rel_label, prop_name
                            )
                            # Only store if the relationship actually has values
                            # for this property
                            if prop_values:
                                values[prop_name] = prop_values
                        except Exception:
                            # Some relationships might not have certain properties -
                            # skip gracefully
                            continue

        except Exception:
            pass
        return values

    def _get_llm_response(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Get response from LLM and handle content extraction."""
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS

        try:
            response = self.anthropic.messages.create(
                model=self.MODEL_NAME,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0]
            return content.text.strip() if hasattr(content, "text") else str(content)
        except Exception as e:
            raise RuntimeError(f"LLM response failed: {str(e)}")

    def _create_workflow(self) -> Any:
        """Create the LangGraph workflow."""
        workflow = StateGraph(WorkflowState)

        workflow.add_node("classify", self.classify_question)
        workflow.add_node("extract", self.extract_entities)
        workflow.add_node("generate", self.generate_query)
        workflow.add_node("execute", self.execute_query)
        workflow.add_node("format", self.format_answer)

        workflow.add_edge("classify", "extract")
        workflow.add_edge("extract", "generate")
        workflow.add_edge("generate", "execute")
        workflow.add_edge("execute", "format")
        workflow.add_edge("format", END)

        workflow.set_entry_point("classify")
        return workflow.compile()

    def _build_classification_prompt(self, question: str) -> str:
        """Build classification prompt with consistent formatting."""
        return f"""Classify this biomedical question. Choose one:
- gene_disease: genes and diseases
- drug_treatment: drugs and treatments
- protein_function: proteins and functions
- general_db: database exploration
- general_knowledge: biomedical concepts

Question: {question}

Respond with just the type."""

    def classify_question(self, state: WorkflowState) -> WorkflowState:
        """Classify the biomedical question type using an LLM.

        Uses LLM-based classification instead of hardcoded keyword matching for
        more flexible and accurate question type detection. This enables the agent
        to handle nuanced questions that don't fit simple keyword patterns.
        """
        try:
            # Build classification prompt with available question types
            prompt = self._build_classification_prompt(state["user_question"])
            if state.get("conversation_history"):
                prompt += "\n\nRecent conversation context:\n"
                for i, turn in enumerate(state["conversation_history"], 1):
                    prompt += f"{i}. Q: {turn['question']}\n"
                    classify_step = turn.get('steps', {}).get('classify', '')
                    if classify_step:
                        # Extract type
                        if 'question_type=' in classify_step:
                            qtype = classify_step.split('=')[1]
                            prompt += f"   Type: {qtype}\n"
                prompt += "\nIf the current question is a follow-up (uses 'what about', 'and', 'also', 'how about' etc.), use the SAME classification as the previous question.\n"
            # Use minimal tokens since we only need a single classification word
            state["question_type"] = self._get_llm_response(prompt, max_tokens=20)
            # log classification to memory
            self.history.set_question_type(state["question_type"])
            self.history.add_step("classify", f"question_type={state['question_type']}")
        except Exception as e:
            # If classification fails, record error but continue with safe fallback
            state["error"] = f"Classification failed: {str(e)}"
            # Default to general knowledge to avoid database queries with
            # malformed inputs
            state["question_type"] = "general_knowledge"
        return state

    def extract_entities(self, state: WorkflowState) -> WorkflowState:
        """Extract biomedical entities from the question."""
        question_type = state.get("question_type")
        if question_type in ["general_db", "general_knowledge"]:
            state["entities"] = []
            self.history.set_entities([])
            self.history.add_step("extract", "entities=[]")
            return state

        # Build property info from database
        property_info = []
        for prop_name, values in self.property_values.items():
            if values:
                sample_values = ", ".join(str(v) for v in values[:3])
                property_info.append(f"- {prop_name}: {sample_values}")

        entity_types_str = ", ".join(self.schema.get("node_labels", []))
        relationship_types_str = ", ".join(self.schema.get("relationship_types", []))

        # Get recent entities for pronoun resolution
        recent_entities = self.history.get_recent_entities(k=2)
        last_question = self.history.get_last_question()

        prompt = f"""Extract biomedical entities from this question.

    Available entity types: {entity_types_str}
    Available relationships: {relationship_types_str}

    Property values in database:
    {chr(10).join(property_info) if property_info else "- No property values available"}

    Question: {state['user_question']}
    """

        # Add pronoun resolution context if we have history
        if recent_entities:
            prompt += f"""
    CRITICAL - PRONOUN RESOLUTION:
    Previous question: "{last_question}"
    Previous entities: {recent_entities}

    The current question contains a pronoun ("they", "it", "those", etc.).
    You MUST include these previous entities in your output: {recent_entities}

    Output MUST include: {recent_entities}
    """

        prompt += """
    Extract ALL relevant terms including:
    - Specific entity names (genes, proteins, diseases, drugs)
    - Property values or constraints mentioned
    - Any entities that pronouns refer to (from previous turns)

    Return ONLY a JSON list: ["term1", "term2"] or [] if none found."""

        try:
            response_text = self._get_llm_response(prompt, max_tokens=100)
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
            
            entities = json.loads(cleaned_text)
            state["entities"] = entities
        except (json.JSONDecodeError, AttributeError):
            state["entities"] = []

        # Store parsed entities in history (the actual list, not string)
        self.history.set_entities(state["entities"])
        self.history.add_step("extract", f"entities={state['entities']}")

        return state

    def generate_query(self, state: WorkflowState) -> WorkflowState:
        """Generate Cypher query based on question type and conversation context."""
        question_type = state.get("question_type", "general")
        if question_type == "general_db":
            state["cypher_query"] = self.SCHEMA_QUERY
            self.history.add_step("generate", f"cypher_query={state['cypher_query']}")
            return state

        if question_type == "general_knowledge":
            state["cypher_query"] = None
            self.history.add_step("generate", "cypher_query=None")
            return state

        # Build relationship guide
        relationship_guide = """
    Relationship directions (use exactly as shown):
    - Gene -[:ENCODES]-> Protein
    - Gene -[:LINKED_TO]-> Disease
    - Protein -[:ASSOCIATED_WITH]-> Disease
    - Drug -[:TREATS]-> Disease
    - Drug -[:TARGETS]-> Protein

    For multi-hop queries, chain relationships. Example:
    "What proteins do drugs that treat cardiovascular diseases target?"
    MATCH (d:Drug)-[:TREATS]->(dis:Disease), (d)-[:TARGETS]->(p:Protein)
    WHERE dis.category = 'cardiovascular'
    RETURN p.protein_name, d.drug_name
    """

        # Build property info
        property_details = []
        for prop_name, values in self.property_values.items():
            if values:
                value_type = "text" if isinstance(values[0], str) else "numeric"
                property_details.append(f"- {prop_name}: {values[:5]} ({value_type})")

        property_info = f"""Property values:
    {chr(10).join(property_details) if property_details else "- No values available"}"""

        prompt = f"""Create a Cypher query for this biomedical question.

    Question: {state['user_question']}
    Question type: {question_type}
    Extracted entities: {state.get('entities', [])}

    Schema:
    Nodes: {', '.join(self.schema['node_labels'])}
    Relationships: {', '.join(self.schema['relationship_types'])}

    {property_info}

    {relationship_guide}
    """

        # Add previous query context for follow-up questions
        recent_turns = self.history.get_summary(k=2)
        if recent_turns:
            last_turn = recent_turns[-1]
            prev_query = last_turn.get('steps', {}).get('generate', '')
            prev_question = last_turn.get('question', '')
            prev_entities = last_turn.get('entities', [])
            
            if prev_query and 'cypher_query=' in prev_query:
                actual_query = prev_query.split('cypher_query=')[1]
                prompt += f"""
    CONVERSATION CONTEXT:
    Previous question: "{prev_question}"
    Previous entities: {prev_entities}
    Previous query:
    {actual_query}

    The current question is a FOLLOW-UP. You must:
    1. Use entities/filters from the previous query as starting point
    2. Chain relationships to answer the new question
    3. Keep relevant WHERE clauses from previous query

    Example: If previous query found "approved drugs treating cardiovascular diseases"
    and new question asks "what proteins do they target", your query should:
    MATCH (d:Drug)-[:TREATS]->(dis:Disease), (d)-[:TARGETS]->(p:Protein)
    WHERE d.approval_status = 'approved' AND dis.category = 'cardiovascular'
    RETURN p.protein_name, d.drug_name
    LIMIT 10
    """

        prompt += """
    Return ONLY the Cypher query, no explanation."""

        cypher_query = self._get_llm_response(prompt, max_tokens=200)

        # Clean markdown formatting
        if cypher_query.startswith("```"):
            cypher_query = "\n".join(
                line for line in cypher_query.split("\n")
                if not line.startswith("```") and not line.startswith("cypher")
            ).strip()

        state["cypher_query"] = cypher_query
        self.history.add_step("generate", f"cypher_query={state['cypher_query']}")

        return state
    
    def execute_query(self, state: WorkflowState) -> WorkflowState:
        """Execute the generated Cypher query against the database.

        Safely executes the LLM-generated query with error handling to prevent
        crashes from malformed queries while capturing useful error information.
        """
        try:
            query = state.get("cypher_query")
            # Execute query only if one was generated (some question types
            # skip this step)
            state["results"] = self.graph_db.execute_query(query) if query else []
        except Exception as e:
            # Capture query execution errors but continue workflow to provide
            # helpful feedback
            state["error"] = f"Query failed: {str(e)}"
            # Set empty results so the format step can handle the error gracefully
            state["results"] = []

        #memory logging
        summary = (
            f"{len(state['results'])} results" 
            if state.get("results") 
            else "no results"
        )
        self.history.add_step("execute", summary)
        return state

    def format_answer(self, state: WorkflowState) -> WorkflowState:
        """Format database results into human-readable answer.

        Takes raw database results and converts them into natural language
        responses, handling different question types and error conditions.
        """
        # Handle any errors that occurred during the workflow
        if state.get("error"):
            state["final_answer"] = (
                f"Sorry, I had trouble with that question: {state['error']}"
            )
            return state

        question_type = state.get("question_type")

        # General knowledge questions use LLM knowledge instead of database results
        if question_type == "general_knowledge":
            # Generate answer from LLM's training knowledge rather than database lookup
            state["final_answer"] = self._get_llm_response(
                f"""Answer this general biomedical question using your knowledge:

Question: {state['user_question']}

Provide a clear, informative answer about biomedical concepts.""",
                max_tokens=300,  # Allow more tokens for explanatory content
            )
            #memory
            self.history.add_step("format", f"final_answer={state['final_answer'][:120]}...")
            return state

        # Handle database-based answers using query results
        results = state.get("results", [])
        if not results:
            # No results found - provide helpful guidance for next steps
            state["final_answer"] = (
                "I didn't find any information for that question. Try asking about "
                "genes, diseases, or drugs in our database."
            )
            self.history.add_step("format", f"final_answer={state['final_answer'][:120]}...")
            return state

        # Convert raw database results into natural language using LLM
        state["final_answer"] = self._get_llm_response(
            f"""Convert these database results into a clear answer:

Question: {state['user_question']}
Results: {json.dumps(results[:5], indent=2)}
Total found: {len(results)}

Make it concise and informative.""",
            max_tokens=250,  # Balanced token limit for informative but concise
            # responses
        )
        self.history.add_step("format", f"final_answer={state['final_answer'][:120]}...")

        return state

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a biomedical question using the LangGraph workflow."""
        self.history.start_turn(question)
        recent_history = self.history.get_summary(k=3)

        initial_state = WorkflowState(
            user_question=question,
            conversation_history=recent_history,
            question_type=None,
            entities=None,
            cypher_query=None,
            results=None,
            final_answer=None,
            error=None,
        )

        final_state = self.workflow.invoke(initial_state)
        self.history.finalize_turn()

        return {
            "answer": final_state.get("final_answer", "No answer generated"),
            "question_type": final_state.get("question_type"),
            "entities": final_state.get("entities", []),
            "cypher_query": final_state.get("cypher_query"),
            "results_count": len(final_state.get("results", [])),
            "raw_results": final_state.get("results", [])[:3],
            "error": final_state.get("error"),
        }


def create_workflow_graph() -> Any:
    """Factory function for LangGraph Studio."""
    load_dotenv()

    graph_interface = GraphInterface(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
    )

    agent = WorkflowAgent(
        graph_interface=graph_interface,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    )

    return agent.workflow
