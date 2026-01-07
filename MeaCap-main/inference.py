# -*- coding: utf-8 -*-
"""
Enhanced Semantic Processor for MeaCap with Full COS-Net Integration
Complete pipeline: Visual Encoder -> Semantic Comprehender -> Semantic Ranker
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from cosnet_visual_adapter import COSNetVisualAdapter
from semantic_comprehender import SemanticComprehender
from semantic_ranker import SemanticRanker, HierarchicalSemanticRanker
from concept_optimizer import ConceptOptimizer

# Optional: original COSNet semantic wrapper (requires local cosnet package)
try:
    from cosnet.cosnet_integration import COSNetSemanticWrapper  # type: ignore
    HAS_COSNET_WRAPPER = True
except Exception:
    HAS_COSNET_WRAPPER = False


class MeaCapSemanticProcessor:
    """
    Complete MeaCap Semantic Processor with full COS-Net pipeline
    """
    def __init__(self, args, device='cuda'):
        self.device = device
        self.confidence_threshold = getattr(args, 'semantic_confidence_threshold', 0.3)
        # 原始最多保留 max_key_concepts 个概念
        self.max_concepts = getattr(args, 'max_key_concepts', 10)
        # 实现时发现过多的概念会削弱多样性，强制截断 6
        self.output_max_concepts = min(self.max_concepts, 6)
        self.use_cosnet_original_semantics = getattr(args, 'use_cosnet_original_semantics', False)

        # Load precomputed COSNet clip-filtered semantic candidates (attr_pred per image_id)
        self.cosnet_anno = {}
        self._load_cosnet_clipfilter_annos([
            "/home/cyp/project/mea_cos/xmodaler/open_source_dataset/mscoco_dataset/cosnet/mscoco_caption_anno_fast_train.pkl",
            "/home/cyp/project/mea_cos/xmodaler/open_source_dataset/mscoco_dataset/cosnet/mscoco_caption_anno_fast_val.pkl",
            "/home/cyp/project/mea_cos/xmodaler/open_source_dataset/mscoco_dataset/cosnet/mscoco_caption_anno_fast_test.pkl",
        ])
        
        # Model parameters
        self.hidden_size = getattr(args, 'semantic_hidden_size', 512)
        self.num_classes = getattr(args, 'semantic_num_classes', 906)
        self.slot_size = getattr(args, 'semantic_slot_size', 6)
        self.num_visual_layers = getattr(args, 'num_visual_layers', 6)
        self.num_semantic_layers = getattr(args, 'num_semcomphder_layers', 3)
        self.max_pos = getattr(args, 'semantic_max_pos', 26)
        
        # Weight configuration
        self.slot_visual_weight = getattr(args, 'slot_visual_weight', 0.8)
        self.word_visual_weight = getattr(args, 'word_visual_weight', 0.6)
        self.frequency_bonus_weight = getattr(args, 'frequency_bonus_weight', 0.3)
        
        # Ranker configuration
        self.use_semantic_ranker = getattr(args, 'use_semantic_ranker', True)
        self.use_hierarchical_ranker = getattr(args, 'use_hierarchical_ranker', True)
        
        # Load vocabulary
        vocab_path = getattr(args, 'vocab_path', 'data/new_vocab_words.txt')
        self.vocab = self._load_vocabulary(vocab_path)
        self.word_to_id = {word: idx for idx, word in enumerate(self.vocab)}
        self.id_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        
        # Initialize modules
        self._init_cosnet_modules(args)
        
        # Concept optimizer
        self.concept_optimizer = ConceptOptimizer(device=device)
        
        print(f"\n{'='*60}")
        print(f"Enhanced MeaCap Semantic Processor with COS-Net")
        print(f"{'='*60}")
        print(f"Architecture Pipeline:")
        print(f"  1. COS-Net Visual Encoder: {self.num_visual_layers} layers")
        print(f"  2. Semantic Comprehender:  {self.num_semantic_layers} layers")
        print(f"  3. Semantic Ranker:        {'Hierarchical' if self.use_hierarchical_ranker else 'Sequential'}")
        print(f"  4. Concept Optimizer:      Enabled")
        print(f"{'='*60}\n")

    def _load_cosnet_clipfilter_annos(self, paths):
        """
        Load COSNet clip-filtered semantic candidate annotations from pickle files.
        Each pickle is expected to be either a list of dicts with an 'image_id' key,
        or a dict mapping image_id to such dicts.
        We index them by integer image_id for fast lookup at inference.
        """
        total = 0
        for path in paths:
            if not path:
                continue
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
            except Exception:
                continue
            if isinstance(data, dict):
                for k, v in data.items():
                    try:
                        key = int(k)
                    except Exception:
                        continue
                    self.cosnet_anno[key] = v
                    total += 1
            elif isinstance(data, list):
                for ent in data:
                    if not isinstance(ent, dict):
                        continue
                    img_id = ent.get("image_id", None)
                    if img_id is None:
                        continue
                    try:
                        key = int(img_id)
                    except Exception:
                        continue
                    self.cosnet_anno[key] = ent
                    total += 1
        if total > 0:
            print(f"[COSNetSemanticWrapper] Loaded {total} COSNet clipfilter entries.")
        else:
            print("[COSNetSemanticWrapper] Warning: no COSNet clipfilter annotations loaded.")
        
    def _load_vocabulary(self, vocab_path):
        """Load vocabulary from file"""
        try:
            # 用 latin-1 可兼容任意 8-bit 字节流，不会抛解码异常
            with open(vocab_path, 'r', encoding='latin-1') as f:
                vocab = [line.strip() for line in f.readlines()]
            print(f"Loaded vocabulary: {len(vocab)} words")
            return vocab
        except Exception as e:
            print(f"[Warning] Vocabulary load failed ({e}), fallback to default.")
            return self._get_default_vocab()
    
    def _get_default_vocab(self):
        """Get default vocabulary"""
        return ['person', 'man', 'woman', 'child', 'boy', 'girl', 'car', 'bicycle', 'dog', 'cat',
                'horse', 'cow', 'sheep', 'bird', 'tree', 'flower', 'grass', 'building', 'house', 'road',
                'street', 'table', 'chair', 'bed', 'desk', 'food', 'water', 'ball', 'bat', 'player',
                'baseball', 'batter', 'pitcher', 'field', 'court', 'game', 'sport', 'team', 'standing',
                'sitting', 'walking', 'running', 'playing', 'holding', 'wearing', 'looking', 'eating',
                'drinking', 'talking', 'white', 'black', 'red', 'blue', 'green', 'yellow', 'brown',
                'young', 'old', 'small', 'large', 'big', 'tall', 'short', 'giraffe', 'elephant',
                'monkey', 'zebra', 'tiger', 'lion', 'bear', 'area', 'yard', 'park', 'room', 'kitchen',
                'bathroom', 'bedroom', 'office', 'next', 'near', 'behind', 'front', 'under', 'over',
                'sunglasses', 'hat', 'shirt', 'dress', 'pants', 'shoes', 'wall', 'ceiling', 'floor']
    
    def _init_cosnet_modules(self, args):
        """Initialize COS-Net modules"""
        # Configuration
        class Config:
            def __init__(self, args):
                for key, value in args.__dict__.items():
                    setattr(self, key, value)
        
        config = Config(args)
        
        # Visual Encoder Adapter
        self.visual_adapter = COSNetVisualAdapter(config).to(self.device)
        self.visual_adapter.eval()

        if self.use_cosnet_original_semantics and HAS_COSNET_WRAPPER:
            # Use original COSNetSemanticModule via wrapper
            self.cosnet_semantic = COSNetSemanticWrapper(args, self.device, self.vocab).to(self.device)
            self.cosnet_semantic.eval()
            print("COS-Net modules initialized with original COSNetSemanticModule (via cosnet wrapper).")
        else:
            # Use MeaCap internal SemanticComprehender + SemanticRanker + classification head
            if self.use_cosnet_original_semantics and not HAS_COSNET_WRAPPER:
                print("[MeaCapSemanticProcessor] COSNet wrapper not available, falling back to internal semantic modules.")

            self.semantic_comprehender = SemanticComprehender(config).to(self.device)
            self.semantic_comprehender.eval()

            if self.use_hierarchical_ranker:
                self.semantic_ranker = HierarchicalSemanticRanker(
                    hidden_size=self.hidden_size,
                    num_classes=self.num_classes,
                    max_seq_len=self.max_pos,
                    dropout=0.1
                ).to(self.device)
            else:
                self.semantic_ranker = SemanticRanker(
                    hidden_size=self.hidden_size,
                    num_classes=self.num_classes,
                    max_seq_len=self.max_pos,
                    dropout=0.1
                ).to(self.device)
            self.semantic_ranker.eval()

            # Category predictor
            self.category_predictor = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size, self.num_classes + 1)
            ).to(self.device)
            self.category_predictor.eval()

            print("COS-Net modules initialized with MeaCap internal semantic components.")
    
    def extract_semantic_words_enhanced(self, memory_captions):
        """Enhanced semantic word extraction with linguistic analysis"""
        word_counts = {}
        word_contexts = {}
        
        for caption in memory_captions:
            words = caption.lower().replace('.', '').replace(',', '').split()
            
            # Extract semantic words with context
            for i, word in enumerate(words):
                if word in self.word_to_id:
                    word_counts[word] = word_counts.get(word, 0) + 1
                    
                    # Store context (previous and next words)
                    if word not in word_contexts:
                        word_contexts[word] = []
                    
                    context = {
                        'prev': words[i-1] if i > 0 else None,
                        'next': words[i+1] if i < len(words)-1 else None
                    }
                    word_contexts[word].append(context)
        
        # Sort by frequency and importance
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Extract top words
        unique_words = []
        word_ids = []
        for word, count in sorted_words[:self.output_max_concepts * 2]:  # Get more candidates
            if word not in unique_words:
                unique_words.append(word)
                word_ids.append(self.word_to_id[word])
        
        return word_ids, unique_words, word_counts, word_contexts
    
    def process_with_cosnet_pipeline(self, visual_features, memory_captions, wte_embeddings=None, image_id=None):
        """
        Complete COS-Net processing pipeline with enhanced features
        """
        with torch.no_grad():
            # Step 1: Get candidate semantic ids
            if image_id is not None and image_id in self.cosnet_anno:
                ent = self.cosnet_anno.get(image_id, {})
                attr_pred = ent.get("attr_pred", [])
                word_ids = []
                words = []
                max_candidates = self.output_max_concepts * 2
                for idx in list(attr_pred)[:max_candidates]:
                    try:
                        cid = int(idx)
                    except Exception:
                        continue
                    if 0 <= cid < len(self.vocab):
                        if cid not in word_ids:
                            word_ids.append(cid)
                            words.append(self.vocab[cid])
                word_counts = {w: 1 for w in words}
                word_contexts = {w: [] for w in words}
            else:
                # Fallback: extract from MeaCap memory captions
                word_ids, words, word_counts, word_contexts = self.extract_semantic_words_enhanced(memory_captions)
            
            if not word_ids:
                return ['person', 'scene'][:self.output_max_concepts], [0.5, 0.5][:self.output_max_concepts], None
            
            # Prepare tensors
            semantic_ids_tensor = torch.tensor(word_ids[:20], device=self.device, dtype=torch.long).unsqueeze(0)
            
            # Step 2: Visual Encoder processing
            encoded_visual, global_features = self.visual_adapter(visual_features)

            # 如果使用原始 COSNet 语义模块，则直接使用其输出
            if self.use_cosnet_original_semantics and hasattr(self, "cosnet_semantic"):
                ordered_concepts, concept_scores, extra = self.cosnet_semantic.infer_ordered(
                    encoded_visual, semantic_ids_tensor
                )

                if not ordered_concepts:
                    # 若原始模块未返回有效结果，则返回默认概念
                    return ['person', 'scene'], [0.5, 0.5], None

                return ordered_concepts[:self.output_max_concepts], concept_scores[:self.output_max_concepts], {
                    'global_features': global_features.cpu().numpy() if global_features is not None else None,
                    'ranking_method': extra.get('ranking_method', 'cosnet_original')
                }

            # 否则使用 MeaCap 内部实现的语义模块（兼容原始逻辑）

            # Step 3: Semantic Comprehender
            comprehended_features, slot_features, word_features = self.semantic_comprehender(
                semantic_ids=semantic_ids_tensor,
                visual_features=encoded_visual
            )

            # Step 4: Category prediction with visual-semantic alignment
            category_logits = self.category_predictor(comprehended_features)

            # Separate slot and word predictions
            slot_predictions = category_logits[0, :self.slot_size, :]
            word_predictions = category_logits[0, self.slot_size:self.slot_size + len(word_ids[:20]), :]

            # Calculate enhanced confidence scores
            slot_probs = F.softmax(slot_predictions, dim=-1)
            word_probs = F.softmax(word_predictions, dim=-1)

            # Enhanced scoring with visual-semantic alignment
            concept_scores_dict = {}

            # Process slot predictions
            for i in range(min(self.slot_size, slot_probs.size(0))):
                top_k = min(5, slot_probs.size(1) - 1)
                top_categories = torch.topk(slot_probs[i, :-1], k=top_k)

                for score, cat_id in zip(top_categories.values, top_categories.indices):
                    if cat_id.item() < len(self.vocab):
                        concept = self.vocab[cat_id.item()]
                        if concept in words:  # Prefer words from memory
                            score_val = score.item() * self.slot_visual_weight * 1.2
                        else:
                            score_val = score.item() * self.slot_visual_weight

                        concept_scores_dict[concept] = max(
                            concept_scores_dict.get(concept, 0.0),
                            score_val
                        )

            # Process word predictions with context
            for i, word in enumerate(words[:word_predictions.size(0)]):
                if word in self.word_to_id:
                    word_id = self.word_to_id[word]
                    if word_id < word_probs.shape[1] - 1:
                        base_score = word_probs[i, word_id].item() if i < word_probs.size(0) else 0.5

                        # Boost score based on frequency
                        freq_boost = min(word_counts[word] / len(memory_captions), 0.3)

                        final_score = base_score * self.word_visual_weight + freq_boost
                        concept_scores_dict[word] = max(
                            concept_scores_dict.get(word, 0.0),
                            final_score
                        )

            # Filter and select top concepts
            sorted_concepts = sorted(concept_scores_dict.items(), key=lambda x: x[1], reverse=True)
            filtered_concepts = [(c, s) for c, s in sorted_concepts if s > self.confidence_threshold]
            final_concepts = filtered_concepts[:self.output_max_concepts]

            if not final_concepts:
                return ['person', 'scene'][:self.output_max_concepts], [0.5, 0.5][:self.output_max_concepts], None

            # Step 5: Semantic Ranking with enhanced features
            concept_words = [c for c, s in final_concepts]
            concept_scores = [s for c, s in final_concepts]
            ranking_word_ids = [self.word_to_id[c] for c in concept_words if c in self.word_to_id]

            if self.use_semantic_ranker and hasattr(self, 'semantic_ranker'):
                # Prepare features for ranking
                num_concepts = len(ranking_word_ids)
                word_features_for_ranker = word_features[:, :num_concepts, :]

                # Use visual-semantic aligned ranking
                if self.use_hierarchical_ranker:
                    ordered_word_ids = self.semantic_ranker(
                        word_features_for_ranker,
                        slot_features,
                        ranking_word_ids,
                        concept_scores,
                        word_contexts={words[i]: word_contexts.get(words[i], [])
                                       for i in range(len(words)) if i < num_concepts}
                    )
                else:
                    ordered_word_ids, _ = self.semantic_ranker(
                        word_features_for_ranker,
                        slot_features,
                        ranking_word_ids,
                        concept_scores,
                        return_scores=True
                    )

                # Convert to words
                ordered_concepts = []
                for wid in ordered_word_ids:
                    if wid < len(self.vocab):
                        ordered_concepts.append(self.vocab[wid])

                # Step 6: Apply concept optimization
                ordered_concepts = self.concept_optimizer.optimize_concept_order(ordered_concepts)
                ordered_scores = [concept_scores_dict.get(c, 0.5) for c in ordered_concepts]

            else:
                ordered_concepts = concept_words
                ordered_scores = concept_scores

            return ordered_concepts, ordered_scores, {
                'global_features': global_features.cpu().numpy() if global_features is not None else None,
                'slot_features': slot_features.cpu().numpy(),
                'ranking_method': 'hierarchical' if self.use_hierarchical_ranker else 'sequential'
            }
    
    def integrate_with_meacap(self, batch_image_embeds, select_memory_captions,
                              wte_model, device, logger, args, image_name=None):
        """
        Integrate full COS-Net semantic pipeline into MeaCap.
        """
        logger.logger.info("="*60)
        logger.logger.info("Processing with COS-Net Enhanced Architecture:")
        logger.logger.info("  Visual Encoder -> Semantic Comprehender -> Semantic Ranker -> Optimizer")
        logger.logger.info("="*60)
        
        try:
            # Get memory text embeddings
            memory_text_embeddings = wte_model.encode(select_memory_captions, convert_to_tensor=True)

            # Parse COCO image_id from image_name (e.g., 'COCO_val2014_000000027440.jpg' -> 27440)
            image_id = None
            if image_name is not None:
                base_name = os.path.splitext(image_name)[0]
                parts = base_name.split("_")
                if len(parts) > 0:
                    try:
                        image_id = int(parts[-1])
                    except Exception:
                        image_id = None
            
            # Process through complete pipeline (COSNet-style semantics)
            ordered_concepts, concept_scores, additional_info = self.process_with_cosnet_pipeline(
                batch_image_embeds.squeeze(0) if batch_image_embeds.dim() > 1 else batch_image_embeds,
                select_memory_captions,
                memory_text_embeddings,
                image_id=image_id
            )

            # 使用原始简单统计提取一份关键概念，作为 MeaCap 原始关键概念（带频率分数）
            basic_concepts, basic_scores = self._extract_basic_concepts_with_scores(select_memory_captions)

            # 将 COSNet 结果与原始关键概念融合，得到"最终的关键概念列表"（基于融合分数）
            final_concepts = self._merge_concept_lists_with_scores(
                cosnet_concepts=ordered_concepts,
                cosnet_scores=concept_scores,
                basic_concepts=basic_concepts,
                basic_scores=basic_scores,
                top_n=self.max_concepts
            )
            
            # Log results
            logger.logger.info(f"COS-Net extracted and ordered concepts: {ordered_concepts}")
            logger.logger.info(f"Basic extracted concepts: {basic_concepts}")
            logger.logger.info(f"Final merged key concepts: {final_concepts}")
            logger.logger.info(f"Concept scores: {[f'{s:.3f}' for s in concept_scores]}")

            if additional_info:
                logger.logger.info(f"Ranking method: {additional_info.get('ranking_method', 'unknown')}")
                if additional_info.get('global_features') is not None:
                    logger.logger.info(f"Global feature dimension: {additional_info['global_features'].shape}")
            
            # 返回“最终的关键概念列表”
            return final_concepts
            
        except Exception as e:
            logger.logger.error(f"COS-Net pipeline error: {e}")
            logger.logger.info("Fallback to basic extraction")
            
            # Fallback mechanism：用原始逻辑（仅高频关键词）
            basic_concepts, _ = self._extract_basic_concepts_with_scores(select_memory_captions)
            return basic_concepts[:self.max_concepts] if basic_concepts else ["person", "scene"]
    
    def _extract_basic_concepts(self, captions):
        """Basic concept extraction as fallback"""
        basic_concepts, _ = self._extract_basic_concepts_with_scores(captions)
        return basic_concepts
    
    def _extract_basic_concepts_with_scores(self, captions):
        """
        Basic concept extraction with frequency scores
        Returns: (concepts_list, scores_list)
        """
        all_words = []
        for caption in captions:
            words = caption.lower().replace('.', '').replace(',', '').split()
            for word in words:
                if word in self.word_to_id:
                    all_words.append(word)
        
        # Get unique words by frequency
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Normalize frequency scores to [0, 1]
        max_count = max(word_counts.values()) if word_counts else 1
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        concepts = [word for word, _ in sorted_words]
        scores = [count / max_count for _, count in sorted_words]
        
        return concepts, scores

    def _merge_concept_lists_with_scores(self, cosnet_concepts, cosnet_scores, basic_concepts, basic_scores, top_n=None):
        """
        基于融合分数将 COSNet 概念与 MeaCap 原始关键概念融合
        
        融合分数计算：
        - COSNet 分数：来自视觉-语义对齐的置信度
        - Basic 分数：来自记忆库中的频率统计
        - 融合分数 = alpha * cosnet_score + beta * basic_score
        
        Args:
            cosnet_concepts: List[str]，COSNet 排序后的概念
            cosnet_scores: List[float]，COSNet 概念分数
            basic_concepts: List[str]，从记忆库统计得到的关键概念
            basic_scores: List[float]，Basic 概念频率分数
            top_n: 截断长度，默认为 self.max_concepts
        """
        if top_n is None:
            top_n = self.max_concepts
        
        # 融合权重：COSNet 分数权重更高（因为经过视觉验证）
        alpha = 0.6  # COSNet 分数权重
        beta = 0.4   # Basic 频率分数权重
        
        # 构建概念到分数的映射
        cosnet_score_dict = {}
        for concept, score in zip(cosnet_concepts, cosnet_scores):
            cosnet_score_dict[concept] = float(score)
        
        basic_score_dict = {}
        for concept, score in zip(basic_concepts, basic_scores):
            basic_score_dict[concept] = float(score)
        
        # 定义同义词映射（去除重复）
        synonym_map = {
            'stands': 'standing', 'sits': 'sitting', 'walks': 'walking',
            'runs': 'running', 'plays': 'playing', 'holds': 'holding',
            'wears': 'wearing', 'looks': 'looking', 'eats': 'eating',
            'drinks': 'drinking', 'talks': 'talking', 'hits': 'hitting',
            'throws': 'throwing', 'catches': 'catching', 'rides': 'riding',
            'flies': 'flying', 'man': 'person', 'woman': 'person',
            'boy': 'person', 'girl': 'person', 'child': 'person',
        }
        
        def normalize_word(word):
            return synonym_map.get(word, word)
        
        # 收集所有唯一概念并计算融合分数
        all_concepts = set(cosnet_concepts) | set(basic_concepts)
        fusion_scores = {}
        concept_mapping = {}  # normalized -> original
        
        for concept in all_concepts:
            normalized = normalize_word(concept)
            
            # 选择更通用的形式作为代表
            if normalized not in concept_mapping:
                concept_mapping[normalized] = concept
            elif concept in basic_concepts and concept_mapping[normalized] not in basic_concepts:
                concept_mapping[normalized] = concept
            
            # 计算融合分数
            cosnet_score = cosnet_score_dict.get(concept, 0.0)
            basic_score = basic_score_dict.get(concept, 0.0)
            
            # 如果概念在 basic_concepts 中，给予额外奖励（来自记忆库）
            if concept in basic_concepts:
                basic_score = max(basic_score, 0.3)  # 至少 0.3
            
            # 融合分数
            fusion_score = alpha * cosnet_score + beta * basic_score
            
            # 如果概念同时出现在两个列表中，额外奖励
            if concept in cosnet_concepts and concept in basic_concepts:
                fusion_score *= 1.2
            
            fusion_scores[normalized] = fusion_score
        
        # 按融合分数排序
        sorted_concepts = sorted(
            fusion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 提取最终概念列表（使用原始形式，去除同义词）
        final_concepts = []
        seen_normalized = set()
        
        for normalized, score in sorted_concepts:
            if normalized not in seen_normalized:
                original = concept_mapping[normalized]
                final_concepts.append(original)
                seen_normalized.add(normalized)
        
        return final_concepts[:top_n]
    
    def _merge_concept_lists(self, cosnet_concepts, basic_concepts, top_n=None):
        """
        旧版本：不使用分数，仅用于向后兼容
        将 MeaCap 原始关键概念与 COSNet 排序结果融合，得到"最终的关键概念列表"
        改进版本：去除重复词、按词性优先级排序、确保语义连贯性

        - cosnet_concepts: List[str]，COSNet 重新排序后的概念
        - basic_concepts:  List[str]，从记忆库中统计得到的关键概念
        - top_n:           截断长度，默认为 self.max_concepts
        """
        if top_n is None:
            top_n = self.max_concepts

        # 定义词性分类（用于优先级排序）
        # 名词（对象、人物等）优先级最高
        noun_words = {
            'person', 'man', 'woman', 'child', 'boy', 'girl', 'people', 'men', 'women', 'children',
            'car', 'bicycle', 'dog', 'cat', 'horse', 'cow', 'sheep', 'bird', 'giraffe', 'elephant',
            'building', 'house', 'room', 'kitchen', 'bathroom', 'bedroom', 'office', 'yard', 'park',
            'table', 'chair', 'bed', 'desk', 'ball', 'bat', 'player', 'batter', 'pitcher',
            'field', 'court', 'game', 'sport', 'team', 'baseball', 'tennis', 'soccer',
            'tree', 'flower', 'grass', 'road', 'street', 'area', 'wall', 'ceiling', 'floor',
            'sunglasses', 'hat', 'shirt', 'dress', 'pants', 'shoes', 'phone', 'computer', 'laptop'
        }
        
        # 形容词优先级次之
        adj_words = {
            'white', 'black', 'red', 'blue', 'green', 'yellow', 'brown', 'gray', 'grey',
            'young', 'old', 'small', 'large', 'big', 'tall', 'short', 'long', 'wide', 'narrow',
            'open', 'closed', 'full', 'empty', 'bright', 'dark', 'clean', 'dirty', 'new', 'old'
        }
        
        # 动词优先级最低（但某些关键动词仍需要）
        verb_words = {
            'standing', 'sitting', 'walking', 'running', 'playing', 'holding', 'wearing', 'looking',
            'eating', 'drinking', 'talking', 'hitting', 'throwing', 'catching', 'riding', 'flying'
        }
        
        # 定义同义词/重复词映射（保留更常用的形式）
        synonym_map = {
            'stands': 'standing',
            'sits': 'sitting',
            'walks': 'walking',
            'runs': 'running',
            'plays': 'playing',
            'holds': 'holding',
            'wears': 'wearing',
            'looks': 'looking',
            'eats': 'eating',
            'drinks': 'drinking',
            'talks': 'talking',
            'hits': 'hitting',
            'throws': 'throwing',
            'catches': 'catching',
            'rides': 'riding',
            'flies': 'flying',
            'man': 'person',  # 如果同时有 person 和 man，优先 person
            'woman': 'person',
            'boy': 'person',
            'girl': 'person',
            'child': 'person',
        }
        
        # 去除同义词，保留更常用的形式
        def normalize_word(word):
            return synonym_map.get(word, word)
        
        # 合并并去重，处理同义词
        all_concepts = list(cosnet_concepts) + list(basic_concepts)
        concept_map = {}  # normalized -> original word
        
        for word in all_concepts:
            normalized = normalize_word(word)
            # 如果 normalized 已经存在，保留更早出现的（优先 basic_concepts）
            if normalized not in concept_map:
                concept_map[normalized] = word
            # 如果当前词在 basic_concepts 中，优先使用它
            elif word in basic_concepts and concept_map[normalized] not in basic_concepts:
                concept_map[normalized] = word
        
        # 按优先级分类
        nouns = []
        adjectives = []
        verbs = []
        others = []
        
        unique_words = list(concept_map.values())
        for word in unique_words:
            if word in noun_words:
                nouns.append(word)
            elif word in adj_words:
                adjectives.append(word)
            elif word in verb_words:
                verbs.append(word)
            else:
                others.append(word)
        
        # 优先保留在 basic_concepts 中的词（保持其顺序）
        priority_order = []
        seen_in_priority = set()
        
        # 1. 先添加 basic_concepts 中的名词（按原始顺序）
        for word in basic_concepts:
            normalized = normalize_word(word)
            if normalized in noun_words and normalized not in seen_in_priority:
                if normalized in concept_map:
                    priority_order.append(concept_map[normalized])
                    seen_in_priority.add(normalized)
        
        # 2. 添加其他名词（来自 cosnet_concepts，但不在 basic 中）
        for word in nouns:
            if word not in priority_order:
                priority_order.append(word)
        
        # 3. 添加形容词（优先 basic_concepts 中的）
        for word in basic_concepts:
            normalized = normalize_word(word)
            if normalized in adj_words and normalized not in seen_in_priority:
                if normalized in concept_map:
                    priority_order.append(concept_map[normalized])
                    seen_in_priority.add(normalized)
        
        for word in adjectives:
            if word not in priority_order:
                priority_order.append(word)
        
        # 4. 最后添加关键动词（限制数量，避免语法混乱）
        verb_count = 0
        max_verbs = min(2, max(1, top_n // 3))  # 最多2个动词，至少1个
        
        for word in basic_concepts:
            normalized = normalize_word(word)
            if normalized in verb_words and normalized not in seen_in_priority and verb_count < max_verbs:
                if normalized in concept_map:
                    priority_order.append(concept_map[normalized])
                    seen_in_priority.add(normalized)
                    verb_count += 1
        
        for word in verbs:
            if word not in priority_order and verb_count < max_verbs:
                priority_order.append(word)
                verb_count += 1
        
        # 5. 添加其他词（限制数量）
        for word in others:
            if word not in priority_order and len(priority_order) < top_n:
                priority_order.append(word)
        
        # 截断到 top_n，确保不超过限制
        return priority_order[:top_n]