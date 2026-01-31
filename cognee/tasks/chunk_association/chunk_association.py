from typing import List, Dict, Any, Tuple, Union  
from pydantic import BaseModel  
from cognee.infrastructure.llm.LLMGateway import LLMGateway  
from cognee.infrastructure.llm.prompts import render_prompt, read_query_prompt  
from cognee.infrastructure.databases.vector import get_vector_engine  
from types import SimpleNamespace  
import asyncio  
  
class AssociationDecision(BaseModel):  
    should_associate: bool  
    confidence: float  
    reason: str  
  
async def should_create_association(chunk1: SimpleNamespace, chunk2: SimpleNamespace) -> AssociationDecision:  
    """  
    Uses LLM to determine if two chunks should be semantically associated.  
    """  
    user_context = {  
        "chunk1_text": chunk1.text,  
        "chunk2_text": chunk2.text,  
        "chunk1_id": str(chunk1.id),  
        "chunk2_id": str(chunk2.id)  
    } 
    print(user_context)
      
    user_prompt = render_prompt("chunk_association_user.txt", context=user_context)  
    system_prompt = read_query_prompt("association_system_prompt.txt")  
      
    decision = await LLMGateway.acreate_structured_output(  
        text_input=user_prompt,  
        system_prompt=system_prompt,  
        response_model=AssociationDecision  
    ) 
    print(decision)
    return decision  
  
async def get_all_chunks_with_scores(  
    chunk_text: str,  
    data_chunks: List[Dict],  
    max_associations_per_chunk: int = 5  
) -> List[Tuple[Dict, float]]:  
    """  
    Get chunks using vector search and match with original chunk data.  
    No similarity filtering - returns all results for LLM to decide.  
      
    Returns: List of (chunk_dict, similarity_score) tuples  
    """  
    vector_engine = get_vector_engine()  
      
    # Get ScoredResult objects from vector search  
    scored_results = await vector_engine.search(  
        "DocumentChunk_text",  
        query_text=chunk_text,  
        limit=max_associations_per_chunk  
    )  
      
    # Create mapping from chunk_id to similarity_score (no filtering)  
    score_mapping = {  
        str(result.id): result.score   
        for result in scored_results  
    }  
      
    # Match with original chunk data  
    similar_chunks = []  
    for chunk in data_chunks:  
        chunk_id = str(chunk.get('id', ''))  
        if chunk_id in score_mapping:  
            similar_chunks.append((chunk, score_mapping[chunk_id]))  
      
    # Sort by similarity score (highest first)  
    similar_chunks.sort(key=lambda x: x[1], reverse=True)  
      
    return similar_chunks  
  
async def create_chunk_associations(  
    data_chunks: List[Dict],  
    max_associations_per_chunk: int = 5  
) -> List[Tuple[str, str, str, Dict[str, Any]]]:  
    """  
    Creates semantic association edges between document chunks using LLM validation only.  
      
    Returns: List of edge tuples in format (source_id, target_id, relationship_name, properties)  
    """  
    association_edges = []  
    data_chunks = [SimpleNamespace(**chunk) for chunk in data_chunks]  
      
    for chunk in data_chunks:  
        # Get chunks with scores (no similarity filtering)  
        chunks_with_scores = await get_all_chunks_with_scores(  
            chunk_text=chunk.text,  
            data_chunks=[chunk.__dict__ for chunk in data_chunks],  
            max_associations_per_chunk=max_associations_per_chunk  
        )  
          
        # Process each chunk  
        for similar_chunk_dict, vector_score in chunks_with_scores:  
            if similar_chunk_dict.get('id') != chunk.id:  
                similar_chunk_ns = SimpleNamespace(**similar_chunk_dict)  
                  
                # LLM validation (only decision mechanism)  
                decision = await should_create_association(chunk, similar_chunk_ns)  
                  
                if decision.should_associate:  
                    edge_data = (  
                        str(chunk.id),  
                        str(similar_chunk_dict.get('id')),  
                        "semantically_associated_with",  
                        {  
                            "relationship_name": "semantically_associated_with",  
                            "weight": decision.confidence,  
                            "vector_score": vector_score,  
                            "reason": decision.reason,  
                            "source_node_id": str(chunk.id),  
                            "target_node_id": str(similar_chunk_dict.get('id')),  
                            "association_type": "semantic_similarity"  
                        }  
                    )  
                    association_edges.append(edge_data)  
      
    return association_edges
