#!/usr/bin/env python3
"""
Complex Consensus Test Script

This script tests the multi-LLM round-table discussion with challenging scenarios:
1. Ambiguous cell types that could be multiple things
2. Transitional cell states
3. Mixed marker gene profiles
4. Subtype-level distinctions

Using latest models from each provider.
"""

import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from mllmcelltype import interactive_consensus_annotation
from mllmcelltype.utils import load_api_key

# Load environment variables
load_dotenv("/Users/apple/Research/mLLMCelltype/.env")


def test_complex_consensus():
    """Test consensus with challenging, ambiguous marker gene profiles."""

    # Complex scenario: Human bone marrow with ambiguous cell populations
    # These markers are designed to create disagreement between models

    complex_marker_genes = {
        # Cluster 0: T cell vs NK cell boundary
        # GZMB, PRF1, NKG7 are shared between cytotoxic T cells and NK cells
        # CD3D is T cell specific, but NCAM1 (CD56) is NK specific
        # Without clear CD4/CD8, it's ambiguous
        "0": [
            "CD3D",
            "GZMB",
            "PRF1",
            "NKG7",
            "GNLY",
            "KLRD1",
            "CCL5",
            "CTSW",
            "GZMA",
            "IFNG",
        ],
        # Cluster 1: Monocyte vs Macrophage vs Dendritic cell
        # CD14 is monocyte, CD68 is macrophage, CD1C is DC marker
        # Mixed expression creates classification challenges
        "1": [
            "CD14",
            "CD68",
            "CSF1R",
            "FCGR3A",
            "CD1C",
            "HLA-DRA",
            "LYZ",
            "S100A9",
            "VCAN",
            "FCN1",
        ],
        # Cluster 2: B cell vs Plasma cell transitional
        # MS4A1 (CD20) is B cell, SDC1 (CD138) and IGHG1 are plasma cell
        # This suggests a transitioning plasmablast
        "2": [
            "MS4A1",
            "CD79A",
            "IGHG1",
            "IGHG4",
            "SDC1",
            "XBP1",
            "MZB1",
            "JCHAIN",
            "CD38",
            "PRDM1",
        ],
        # Cluster 3: Mesenchymal stem cell vs Fibroblast vs Myofibroblast
        # Overlapping markers make this a challenging classification
        "3": [
            "COL1A1",
            "COL3A1",
            "VIM",
            "ACTA2",
            "THY1",
            "PDGFRA",
            "ENG",
            "NT5E",
            "MCAM",
            "DCN",
        ],
        # Cluster 4: cDC1 vs cDC2 vs pDC
        # Dendritic cell subtype classification is notoriously difficult
        "4": [
            "CLEC9A",
            "XCR1",
            "CD1C",
            "FCER1A",
            "LILRA4",
            "TCF4",
            "IRF8",
            "BATF3",
            "IDO1",
            "CD83",
        ],
        # Cluster 5: Erythroid progenitor vs Reticulocyte
        # Early vs late erythroid differentiation
        "5": [
            "HBB",
            "HBA1",
            "GYPA",
            "TFRC",
            "KLF1",
            "GATA1",
            "EPOR",
            "SLC4A1",
            "ALAS2",
            "CA1",
        ],
        # Cluster 6: Megakaryocyte vs Platelet progenitor
        # Different stages of megakaryopoiesis
        "6": [
            "PPBP",
            "PF4",
            "GP9",
            "ITGA2B",
            "GP1BA",
            "TUBB1",
            "TREML1",
            "MYL9",
            "SPARC",
            "CD9",
        ],
        # Cluster 7: Regulatory T cell vs Activated T cell vs Exhausted T cell
        # All express high CD4 and activation/regulatory markers
        "7": [
            "CD4",
            "FOXP3",
            "IL2RA",
            "CTLA4",
            "TIGIT",
            "LAG3",
            "PDCD1",
            "ICOS",
            "TNFRSF18",
            "IL10",
        ],
    }

    # Define the models to use (latest versions)
    # Note: Adjust based on which API keys are available
    models_to_use = []

    # Check available API keys and select latest models
    api_keys = {}

    # OpenAI - gpt-5.2
    openai_key = load_api_key("openai")
    if openai_key:
        api_keys["openai"] = openai_key
        models_to_use.append("gpt-5.2")  # Latest GPT model

    # Anthropic - claude-opus-4-5
    anthropic_key = load_api_key("anthropic")
    if anthropic_key:
        api_keys["anthropic"] = anthropic_key
        models_to_use.append("claude-opus-4-5-20251101")  # Latest Opus

    # DeepSeek - V3.2
    deepseek_key = load_api_key("deepseek")
    if deepseek_key:
        api_keys["deepseek"] = deepseek_key
        models_to_use.append("deepseek-chat")  # V3.2

    # Qwen - qwen3-max
    qwen_key = load_api_key("qwen")
    if qwen_key:
        api_keys["qwen"] = qwen_key
        models_to_use.append("qwen3-max")  # Latest Qwen3

    # Zhipu - GLM-4-plus (glm-4.7 has rate limits)
    zhipu_key = load_api_key("zhipu")
    if zhipu_key:
        api_keys["zhipu"] = zhipu_key
        models_to_use.append("glm-4-plus")

    print("=" * 80)
    print("COMPLEX CONSENSUS TEST - Multi-LLM Round-Table Discussion")
    print("=" * 80)
    print("\nTissue: Human Bone Marrow (complex hematopoietic environment)")
    print("Species: Human")
    print(f"\nModels to use ({len(models_to_use)}): {', '.join(models_to_use)}")
    print(f"Number of clusters: {len(complex_marker_genes)}")
    print("\n" + "=" * 80)
    print("CLUSTER MARKER GENE PROFILES (Designed for ambiguity):")
    print("=" * 80)

    cluster_descriptions = {
        "0": "T cell vs NK cell boundary (cytotoxic phenotype)",
        "1": "Monocyte vs Macrophage vs DC (myeloid lineage)",
        "2": "B cell vs Plasma cell transition (plasmablast?)",
        "3": "MSC vs Fibroblast vs Myofibroblast (mesenchymal)",
        "4": "cDC1 vs cDC2 vs pDC (DC subtypes)",
        "5": "Erythroid progenitor vs Reticulocyte",
        "6": "Megakaryocyte vs Platelet progenitor",
        "7": "Treg vs Activated vs Exhausted T cell",
    }

    for cluster, genes in complex_marker_genes.items():
        desc = cluster_descriptions.get(cluster, "Unknown")
        print(f"\nCluster {cluster} - {desc}:")
        print(f"  Markers: {', '.join(genes[:5])}...")

    print("\n" + "=" * 80)
    print("STARTING CONSENSUS ANNOTATION...")
    print("=" * 80 + "\n")

    if len(models_to_use) < 3:
        print(f"WARNING: Only {len(models_to_use)} models available.")
        print("At least 3 models recommended for meaningful consensus.")
        print("Proceeding with available models...")

    if len(models_to_use) < 1:
        print("ERROR: No API keys available. Cannot run test.")
        return

    # Run the interactive consensus annotation
    results = interactive_consensus_annotation(
        marker_genes=complex_marker_genes,
        species="human",
        tissue="bone marrow",
        models=models_to_use,
        api_keys=api_keys,
        consensus_threshold=0.7,  # 70% agreement required
        entropy_threshold=1.0,  # Allow some diversity before requiring discussion
        max_discussion_rounds=3,  # Allow up to 3 rounds of discussion
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    if results:
        # Extract the nested structure
        consensus = results.get("consensus", {})
        consensus_proportions = results.get("consensus_proportion", {})
        entropies = results.get("entropy", {})
        controversial = results.get("controversial_clusters", [])
        resolved = results.get("resolved", {})
        model_annotations = results.get("model_annotations", {})

        print(f"\nControversial clusters requiring discussion: {len(controversial)}")
        if controversial:
            print(f"  {controversial}")

        print("\n" + "-" * 60)
        print("PER-CLUSTER RESULTS:")
        print("-" * 60)

        for cluster_id in sorted(complex_marker_genes.keys(), key=lambda x: int(x)):
            print(f"\n>>> Cluster {cluster_id}: {cluster_descriptions.get(cluster_id, 'N/A')}")

            # Get final annotation
            annotation = consensus.get(cluster_id, "N/A")
            cp = consensus_proportions.get(cluster_id, "N/A")
            h = entropies.get(cluster_id, "N/A")

            print(f"    Final annotation: {annotation}")
            print(f"    Consensus proportion: {cp}")
            print(f"    Entropy: {h}")

            # Show individual model predictions if available
            if model_annotations:
                print("    Individual model predictions:")
                for model, preds in model_annotations.items():
                    if cluster_id in preds:
                        print(f"      - {model}: {preds[cluster_id]}")

            # Check if this cluster was controversial and resolved
            if cluster_id in controversial:
                print("    [CONTROVERSIAL - Required discussion]")
                if cluster_id in resolved:
                    print(f"    Resolved annotation: {resolved.get(cluster_id)}")

        print("\n" + "=" * 80)
        print("METADATA:")
        print("=" * 80)
        metadata = results.get("metadata", {})
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    else:
        print("No results returned!")


if __name__ == "__main__":
    test_complex_consensus()
