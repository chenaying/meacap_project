# -*- coding: utf-8 -*-
"""
IFCap EF (Frequency-based Entity Filtering) Module Integration for MeaCap
"""

import nltk
import numpy as np
from typing import List, Tuple, Optional
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


def log_normal_filter(freq_entities: List[Tuple[int, str]], alpha: float = 1.0) -> List[str]:
    """
    Filter entities using log-normal distribution adaptive threshold
    
    Args:
        freq_entities: List[Tuple[int, str]], [(frequency, entity), ...]
        alpha: float, threshold coefficient
    
    Returns:
        List[str], filtered entity list
    """
    if not freq_entities:
        return []
    
    freq, _ = zip(*freq_entities)
    mean = np.mean(np.log(freq))
    var = np.mean(np.square(np.log(freq) - mean))
    std = var ** 0.5
    threshold = mean + std * alpha

    filtered_entities = list(filter(lambda x: np.log(x[0]) > threshold, freq_entities))
    if len(filtered_entities):
        _, filtered_entities = zip(*filtered_entities)
    else:
        filtered_entities = []

    return list(filtered_entities)


def normal_filter(freq_entities: List[Tuple[int, str]], alpha: float = 1.0) -> List[str]:
    """
    Filter entities using normal distribution adaptive threshold
    
    Args:
        freq_entities: List[Tuple[int, str]], [(frequency, entity), ...]
        alpha: float, threshold coefficient
    
    Returns:
        List[str], filtered entity list
    """
    if not freq_entities:
        return []
    
    freq, _ = zip(*freq_entities)
    mean = np.mean(freq)
    std = np.std(freq)
    threshold = mean + std * alpha

    filtered_entities = list(filter(lambda x: x[0] > threshold, freq_entities))
    if len(filtered_entities):
        _, filtered_entities = zip(*filtered_entities)
    else:
        filtered_entities = []

    return list(filtered_entities)


def extract_entities_from_captions(captions: List[str]) -> List[Tuple[int, str]]:
    """
    Extract entities from caption list and count frequency
    
    Args:
        captions: List[str], caption list
    
    Returns:
        List[Tuple[int, str]], [(frequency, entity), ...], sorted by frequency descending
    """
    lemmatizer = WordNetLemmatizer()
    detected_entities = {}
    
    for caption in captions:
        try:
            # POS tagging
            pos_tags = nltk.pos_tag(nltk.word_tokenize(caption))
            
            # Filter nouns and count frequency
            for word, pos_tag in pos_tags:
                if pos_tag == 'NN' or pos_tag == 'NNS':  # singular or plural noun
                    # Lemmatization and normalization
                    entity = lemmatizer.lemmatize(word.lower().strip())
                    
                    # Filter out too short words
                    if len(entity) > 1:
                        if entity not in detected_entities:
                            detected_entities[entity] = 0
                        detected_entities[entity] += 1
        except Exception as e:
            # Skip this caption if processing fails
            continue
    
    # Convert to list and sort by frequency descending
    freq_entities = [(count, entity) for entity, count in detected_entities.items()]
    freq_entities.sort(reverse=True)
    
    return freq_entities


def retrieve_concepts_with_ef(
    select_memory_captions: List[str],
    filter_method: str = 'log_normal',
    alpha: float = 1.0,
    fixed_threshold: Optional[int] = None,
    max_concepts: int = 10,
    min_freq: int = 1
) -> List[str]:
    """
    Extract key concepts from retrieved captions using IFCap EF module
    
    Args:
        select_memory_captions: List[str], retrieved memory caption list
        filter_method: str, filtering method ('log_normal', 'normal', 'fixed')
        alpha: float, adaptive filtering alpha coefficient (for log_normal and normal)
        fixed_threshold: Optional[int], fixed threshold (for fixed method)
        max_concepts: int, maximum number of concepts to return
        min_freq: int, minimum frequency requirement
    
    Returns:
        List[str], filtered key concept list
    """
    if not select_memory_captions:
        return ['person', 'scene']
    
    # Step 1: Extract entities and count frequency
    freq_entities = extract_entities_from_captions(select_memory_captions)
    
    if not freq_entities:
        return ['person', 'scene']
    
    # Step 2: Apply minimum frequency filter
    freq_entities = [(freq, entity) for freq, entity in freq_entities if freq >= min_freq]
    
    if not freq_entities:
        return ['person', 'scene']
    
    # Step 3: Filter according to filter method
    if filter_method == 'log_normal':
        filtered_entities = log_normal_filter(freq_entities, alpha=alpha)
    elif filter_method == 'normal':
        filtered_entities = normal_filter(freq_entities, alpha=alpha)
    elif filter_method == 'fixed':
        if fixed_threshold is None:
            fixed_threshold = 2  # default threshold
        filtered_entities = [entity for freq, entity in freq_entities if freq >= fixed_threshold]
    else:
        # Default to log_normal
        filtered_entities = log_normal_filter(freq_entities, alpha=alpha)
    
    # Step 4: Limit return count
    filtered_entities = filtered_entities[:max_concepts]
    
    # If no results after filtering, return top frequency entities
    if not filtered_entities:
        filtered_entities = [entity for _, entity in freq_entities[:max_concepts]]
    
    return filtered_entities if filtered_entities else ['person', 'scene']


def retrieve_concepts_with_ef_hybrid(
    select_memory_captions: List[str],
    parser_model=None,
    parser_tokenizer=None,
    wte_model=None,
    image_embeds=None,
    device=None,
    ef_filter_method: str = 'log_normal',
    ef_alpha: float = 1.0,
    ef_fixed_threshold: Optional[int] = None,
    use_ef: bool = True,
    use_parser: bool = True,
    max_concepts: int = 10,
    logger=None
) -> List[str]:
    """
    Hybrid method: combine IFCap EF module with original Parser method
    
    Args:
        select_memory_captions: List[str], retrieved memory caption list
        parser_model: original Parser model (optional)
        parser_tokenizer: Parser tokenizer (optional)
        wte_model: WTE model (optional)
        image_embeds: image embeddings (optional)
        device: device (optional)
        ef_filter_method: str, EF filtering method
        ef_alpha: float, EF alpha coefficient
        ef_fixed_threshold: Optional[int], EF fixed threshold
        use_ef: bool, whether to use EF module
        use_parser: bool, whether to use Parser method
        max_concepts: int, maximum number of concepts
        logger: logger object (optional)
    
    Returns:
        List[str], merged key concept list
    """
    ef_concepts = []
    parser_concepts = []
    
    # Method 1: Extract using EF module
    if use_ef:
        try:
            ef_concepts = retrieve_concepts_with_ef(
                select_memory_captions,
                filter_method=ef_filter_method,
                alpha=ef_alpha,
                fixed_threshold=ef_fixed_threshold,
                max_concepts=max_concepts
            )
            if logger:
                logger.logger.info(f"EF extracted concepts: {ef_concepts}")
        except Exception as e:
            if logger:
                logger.logger.warning(f"EF extraction failed: {e}")
    
    # Method 2: Extract using original Parser method
    if use_parser and parser_model is not None:
        try:
            from .detect_utils import retrieve_concepts
            parser_concepts = retrieve_concepts(
                parser_model=parser_model,
                parser_tokenizer=parser_tokenizer,
                wte_model=wte_model,
                select_memory_captions=select_memory_captions,
                image_embeds=image_embeds,
                device=device,
                logger=logger,
                args=None
            )
            if logger:
                logger.logger.info(f"Parser extracted concepts: {parser_concepts}")
        except Exception as e:
            if logger:
                logger.logger.warning(f"Parser extraction failed: {e}")
    
    # Merge results from both methods
    if ef_concepts and parser_concepts:
        # Merge and deduplicate, prioritize EF results (frequency-based)
        all_concepts = []
        seen = set()
        
        # Add EF results first
        for concept in ef_concepts:
            if concept.lower() not in seen:
                all_concepts.append(concept)
                seen.add(concept.lower())
        
        # Add Parser results (if not in EF results)
        for concept in parser_concepts:
            if concept.lower() not in seen:
                all_concepts.append(concept)
                seen.add(concept.lower())
        
        final_concepts = all_concepts[:max_concepts]
    elif ef_concepts:
        final_concepts = ef_concepts[:max_concepts]
    elif parser_concepts:
        final_concepts = parser_concepts[:max_concepts]
    else:
        final_concepts = ['person', 'scene']
    
    if logger:
        logger.logger.info(f"Final merged concepts: {final_concepts}")
    
    return final_concepts
