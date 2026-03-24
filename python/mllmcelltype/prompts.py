"""Prompt generation module for LLMCellType."""

from __future__ import annotations

from .logger import write_log
from .utils import cluster_sort_key


def _format_marker_genes_for_prompt(
    marker_genes: dict[str, list[str]], cluster_format: str = "Cluster {}: {}"
) -> str:
    """Format marker genes consistently for prompts.

    Clusters are sorted in natural numerical order to match the
    "IN NUMERICAL ORDER" instruction in the prompt template.

    Args:
        marker_genes: Dictionary mapping cluster names to marker gene lists
        cluster_format: Format string for cluster entries

    Returns:
        str: Formatted marker genes text
    """
    marker_lines = []
    for cluster in sorted(marker_genes.keys(), key=cluster_sort_key):
        genes_str = ", ".join(str(g) for g in marker_genes[cluster])
        marker_lines.append(cluster_format.format(cluster, genes_str))
    return "\n".join(marker_lines)


# Default prompt template for single dataset annotation
DEFAULT_PROMPT_TEMPLATE = """You are an expert single-cell RNA-seq analyst specializing in cell type annotation.
I need you to identify cell types of {species} cells from {tissue}.
Below is a list of marker genes for each cluster.
Please assign the most likely cell type to each cluster based on the marker genes.

IMPORTANT: Provide your answers in the EXACT format below, with one cluster per line:
Cluster 0: [cell type]
Cluster 1: [cell type]
...and so on, IN NUMERICAL ORDER.

Only provide the cell type name for each cluster. Be concise but specific.
Some clusters can be a mixture of multiple cell types.

Here are the marker genes for each cluster:
{markers}
"""

SUPPORTED_PROMPT_PLACEHOLDERS = ("species", "tissue", "markers")


def _render_prompt_template(
    *,
    prompt_template: str,
    species: str,
    tissue_text: str,
    marker_text: str,
) -> str:
    """Render prompt template with clear errors for invalid placeholders/format."""
    try:
        return prompt_template.format(species=species, tissue=tissue_text, markers=marker_text)
    except KeyError as e:
        placeholder = e.args[0] if e.args else "unknown"
        raise ValueError(
            "Invalid prompt_template placeholder "
            f"'{placeholder}'. Supported placeholders: {', '.join(SUPPORTED_PROMPT_PLACEHOLDERS)}"
        ) from e
    except ValueError as e:
        raise ValueError(f"Invalid prompt_template format: {e!s}") from e


def create_consensus_check_prompt(annotations: list[str]) -> str:
    """Create a prompt for checking consensus among different annotations.

    Args:
        annotations: List of cell type annotations from different models

    Returns:
        str: Formatted prompt for LLM to check consensus

    """
    prompt = """You are an expert in single-cell RNA-seq analysis and cell type annotation.

I need you to analyze the following cell type annotations from different models for the same cluster and determine if there is a consensus.

The annotations are:
{annotations}

Please analyze these annotations and determine:
1. If there is a consensus (1 for yes, 0 for no)
2. The consensus proportion (between 0 and 1)
3. An entropy value measuring the diversity of opinions (higher means more diverse)
4. The best consensus annotation

Respond with exactly 4 lines:
Line 1: 0 or 1 (consensus reached?)
Line 2: Consensus proportion (e.g., 0.75)
Line 3: Entropy value (e.g., 0.85)
Line 4: The consensus cell type (or most likely if no clear consensus)

Only output these 4 lines, nothing else."""

    # Format the annotations
    formatted_annotations = "\n".join([f"- {anno}" for anno in annotations])

    # Replace the placeholder
    return prompt.replace("{annotations}", formatted_annotations)


def create_prompt(
    marker_genes: dict[str, list[str]],
    species: str,
    tissue: str | None = None,
    additional_context: str | None = None,
    prompt_template: str | None = None,
) -> str:
    """Create a prompt for cell type annotation.

    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes
        species: Species name (e.g., 'human', 'mouse')
        tissue: Tissue name (e.g., 'brain', 'liver')
        additional_context: Additional context to include in the prompt
        prompt_template: Custom prompt template

    Returns:
        str: The generated prompt

    """
    write_log(f"Creating prompt for {len(marker_genes)} clusters")

    # Validate custom template type and use default template when empty.
    if prompt_template is not None and not isinstance(prompt_template, str):
        raise ValueError(
            f"prompt_template must be a string or None, got {type(prompt_template).__name__}"
        )

    # Use default template if not provided
    if not prompt_template or not prompt_template.strip():
        prompt_template = DEFAULT_PROMPT_TEMPLATE

    # Default tissue if none provided
    tissue_text = tissue if tissue else "unknown tissue"

    # Format marker genes text using helper function
    marker_text = _format_marker_genes_for_prompt(marker_genes)

    # Add additional context if provided
    context_text = f"\nAdditional context: {additional_context}\n" if additional_context else ""

    # Fill in the template with clear validation errors.
    prompt = _render_prompt_template(
        prompt_template=prompt_template,
        species=species,
        tissue_text=tissue_text,
        marker_text=marker_text,
    )

    # Add context
    if context_text:
        sections = prompt.split("Here are the marker genes for each cluster:")
        if len(sections) == 2:
            prompt = f"{sections[0]}{context_text}Here are the marker genes for each cluster:{sections[1]}"
        else:
            prompt = f"{prompt}{context_text}"

    write_log(f"Generated prompt with {len(prompt)} characters")
    return prompt


def create_initial_discussion_prompt(
    cluster_id: str,
    marker_genes: list[str],
    initial_predictions: dict[str, str],
    species: str,
    tissue: str | None = None,
) -> str:
    """Create a prompt for the initial round of multi-model discussion.

    This prompt is used when multiple models participate in discussing
    a controversial cluster annotation.

    Args:
        cluster_id: ID of the cluster
        marker_genes: List of marker genes for the cluster
        initial_predictions: Dictionary mapping model names to their initial predictions
        species: Species name (e.g., 'human', 'mouse')
        tissue: Tissue name (e.g., 'brain', 'blood')

    Returns:
        str: The generated prompt for initial discussion round
    """
    write_log(f"Creating initial discussion prompt for cluster {cluster_id}")

    tissue_text = tissue if tissue else "unknown tissue"
    marker_genes_text = ", ".join(marker_genes)

    # Format initial predictions
    predictions_text = "\n".join(
        f"- {model}: {prediction}" for model, prediction in initial_predictions.items()
    )

    prompt = f"""We are analyzing cluster {cluster_id} with the following marker genes: {marker_genes_text}
Species: {species}
Tissue: {tissue_text}

Different models have made different predictions:
{predictions_text}

Please provide your cell type prediction using the Toulmin argumentation model:

1. CLAIM: State your clear cell type prediction
2. GROUNDS: Present specific marker genes that support your claim
3. WARRANT: Explain why these genes indicate this cell type
4. BACKING: Provide references or established knowledge
5. QUALIFIER: Indicate your certainty level (definite, probable, possible)
6. REBUTTAL: Address counter-arguments or other models' predictions

Format your response as:
CELL TYPE: [your predicted cell type]
GROUNDS: [specific marker genes supporting your claim]
WARRANT: [logical connection between evidence and claim]
BACKING: [additional support for your reasoning]
QUALIFIER: [degree of certainty]
REBUTTAL: [addressing counter-arguments]"""

    write_log(f"Generated initial discussion prompt with {len(prompt)} characters")
    return prompt


def create_discussion_prompt(
    cluster_id: str,
    marker_genes: list[str],
    previous_rounds: list[dict[str, str]],
    round_number: int,
    species: str,
    tissue: str | None = None,
) -> str:
    """Create a prompt for subsequent rounds of multi-model discussion.

    This prompt includes the discussion history from all previous rounds,
    allowing each model to see and respond to other models' arguments.

    Args:
        cluster_id: ID of the cluster
        marker_genes: List of marker genes for the cluster
        previous_rounds: List of dicts, each containing model responses for a round
            Example: [{"gpt-5": "response1", "claude": "response2"}, ...]
        round_number: Current round number (2, 3, ...)
        species: Species name (e.g., 'human', 'mouse')
        tissue: Tissue name (e.g., 'brain', 'blood')

    Returns:
        str: The generated prompt for the discussion round
    """
    write_log(f"Creating discussion prompt for cluster {cluster_id}, round {round_number}")

    tissue_text = tissue if tissue else "unknown tissue"
    marker_genes_text = ", ".join(marker_genes)

    # Format previous discussion history
    discussion_history_parts = []
    for round_idx, round_responses in enumerate(previous_rounds, start=1):
        round_text = f"Round {round_idx}:\n"
        for model_name, response in round_responses.items():
            round_text += f"\n{model_name}:\n{response}\n"
        discussion_history_parts.append(round_text)

    discussion_history = "\n".join(discussion_history_parts)

    prompt = f"""We are continuing the discussion for cluster {cluster_id}.
Marker genes: {marker_genes_text}
Species: {species}
Tissue: {tissue_text}

Previous discussion:
{discussion_history}

This is round {round_number} of the discussion.

Using the Toulmin argumentation model, please structure your response:

1. CLAIM: State your clear cell type prediction
2. GROUNDS: Present specific marker genes that support your claim
3. WARRANT: Explain why these genes indicate this cell type
4. BACKING: Provide references or established knowledge
5. QUALIFIER: Indicate your certainty level
6. REBUTTAL: Address counter-arguments or other models' predictions

Based on previous discussion, also indicate:
- Whether you agree or disagree with any emerging consensus
- If you've revised your previous position, explain why

Format your response as:
CELL TYPE: [your current prediction]
GROUNDS: [specific marker genes supporting your claim]
WARRANT: [logical connection between evidence and claim]
BACKING: [additional support for your reasoning]
QUALIFIER: [degree of certainty]
REBUTTAL: [addressing counter-arguments]
CONSENSUS STATUS: [Agree/Disagree with emerging consensus]"""

    write_log(f"Generated discussion prompt with {len(prompt)} characters")
    return prompt


def create_cell_type_extraction_prompt(text: str) -> str:
    """Create a prompt to extract a concise cell type label from free-text.

    When model responses contain natural-language sentences instead of concise
    cell type labels (e.g., "This cluster is likely CD4+ T cells"), this prompt
    asks an LLM to produce a clean label (e.g., "CD4+ T cells").

    Args:
        text: Free-text containing a cell type reference

    Returns:
        str: Prompt for LLM to extract the cell type label
    """
    return (
        "Extract ONLY the cell type name from the text below.\n"
        'Reply with just the cell type (e.g., "CD4+ T cells", "NK cells", '
        '"B lymphocytes").\n'
        "Do not include any explanation or extra words.\n\n"
        f"Text: {text}"
    )


def create_discussion_consensus_prompt(
    round_responses: dict[str, str],
    consensus_threshold: float = 2 / 3,
    entropy_threshold: float = 1.0,
) -> str:
    """Create a prompt for LLM to extract cell types and check consensus.

    Unlike create_consensus_check_prompt which takes pre-extracted annotations,
    this function takes raw discussion responses and asks the LLM to perform
    both cell-type extraction and consensus checking in a single call.

    This mirrors the R implementation (check_consensus.R:488) where the LLM
    receives full discussion responses and produces clean labels + metrics.

    Args:
        round_responses: Dictionary mapping model identifiers to raw responses
        consensus_threshold: Agreement threshold for consensus determination
        entropy_threshold: Entropy threshold for consensus determination

    Returns:
        str: Formatted prompt for LLM consensus check
    """
    # Use anonymous model IDs (Model 1, 2, ...) to prevent provider-name bias
    responses_text = "\n\n".join(
        f"Model {i + 1}:\n{response}"
        for i, (_, response) in enumerate(round_responses.items())
    )

    return (
        "You are a cell type annotation expert. Below are different models' "
        "discussion responses for the same cell cluster.\n\n"
        "Your task:\n"
        "1. Extract the predicted cell type from each model's response\n"
        "2. Determine if the predictions reach consensus\n\n"
        f"RESPONSES:\n{responses_text}\n\n"
        "IMPORTANT GUIDELINES:\n"
        "1. Extract ONLY the cell type name from each response "
        '(e.g., "CD4+ T cells", not full sentences)\n'
        "2. Responses may contain structured predictions (with "
        '"CELL TYPE:" prefix) or free-text predictions\n'
        "3. Consider predictions as matching if they refer to the same "
        "cell type, ignoring:\n"
        '   - Formatting (e.g., "NK cells" vs "Natural Killer cells")\n'
        "   - Capitalization\n"
        '   - Additional qualifiers (e.g., "activated", "mature")\n'
        '4. If any prediction is "Unknown" or "Unclear", treat it as '
        "a separate group\n\n"
        "CALCULATE:\n"
        "1. Consensus Proportion = Number of models supporting the majority "
        "prediction / Total number of models\n"
        "2. Shannon Entropy = -sum(p_i * log2(p_i)) where p_i is the "
        "proportion of each unique prediction\n"
        f"3. Consensus is reached if Consensus Proportion >= "
        f"{consensus_threshold:.2f} AND Entropy <= "
        f"{entropy_threshold:.2f}\n\n"
        "RESPOND WITH EXACTLY 4 LINES:\n"
        "Line 1: 1 if consensus is reached, 0 if not\n"
        "Line 2: Consensus Proportion (a decimal between 0 and 1)\n"
        "Line 3: Shannon Entropy (a decimal number)\n"
        "Line 4: The majority cell type (concise label only, "
        'e.g., "CD4+ T cells")\n\n'
        "RESPOND WITH EXACTLY FOUR LINES AS SPECIFIED ABOVE."
    )
