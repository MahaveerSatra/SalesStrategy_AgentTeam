"""
Seller product taxonomy loader.

Each seller can provide a JSON taxonomy file at:
  src/data_sources/{SellerName}_Complete_Taxonomy_With_Toolbox_Mappings.json

The code loads whatever file exists for the current seller and formats compact
prompt sections for injection into Identifier, Validator, and Coordinator prompts.
No seller-specific product names are hardcoded in agent code.
"""
import json
from pathlib import Path


def load_seller_taxonomy(seller_name: str) -> dict | None:
    """Load seller-specific product taxonomy from JSON file, if available.

    Args:
        seller_name: Seller company name (e.g. "MathWorks")

    Returns:
        Parsed taxonomy dict, or None if no file exists for this seller.
    """
    if not isinstance(seller_name, str):
        return None
    normalized = seller_name.replace(" ", "").replace("-", "")
    path = Path(__file__).parent / f"{normalized}_Complete_Taxonomy_With_Toolbox_Mappings.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def format_taxonomy_for_identifier(taxonomy: dict) -> str:
    """Format taxonomy as compact REQUIREMENT → PRIMARY TOOLBOX block for Identifier prompts.

    Uses cross_reference_matrix.requirement_to_toolbox_mapping (~15 entries).
    Typical output: ~20 lines.

    Args:
        taxonomy: Parsed taxonomy dict from load_seller_taxonomy()

    Returns:
        Formatted string for prompt injection, or empty string if data is missing.
    """
    req_map = (
        taxonomy
        .get("cross_reference_matrix", {})
        .get("requirement_to_toolbox_mapping", {})
    )
    if not req_map:
        return ""

    lines = ["SELLER PRODUCT TAXONOMY — REQUIREMENT TO PRIMARY TOOLBOX:"]
    for req, mapping in req_map.items():
        primary = mapping.get("primary_toolbox", "")
        supporting = mapping.get("supporting_toolboxes", [])
        sup_str = f" [supporting: {', '.join(supporting[:3])}]" if supporting else ""
        lines.append(f"  {req} → {primary}{sup_str}")
    lines.append("Prioritize the PRIMARY toolbox when signals match a requirement type above.")
    return "\n".join(lines)


def format_taxonomy_for_validator(taxonomy: dict) -> str:
    """Format taxonomy for Validator prompts: domain categories + requirement mappings.

    Combines:
    - quick_reference_toolbox_catalog (21 domain categories → toolbox lists)
      Used for peripheral product detection.
    - cross_reference_matrix.requirement_to_toolbox_mapping
      Used for objective gap detection.

    Args:
        taxonomy: Parsed taxonomy dict from load_seller_taxonomy()

    Returns:
        Formatted string for prompt injection, or empty string if data is missing.
    """
    catalog = taxonomy.get("quick_reference_toolbox_catalog", {})
    req_map = (
        taxonomy
        .get("cross_reference_matrix", {})
        .get("requirement_to_toolbox_mapping", {})
    )

    parts = []
    if catalog:
        parts.append("DOMAIN CATEGORIES (for identifying peripheral products):")
        for domain, toolboxes in catalog.items():
            parts.append(f"  {domain}: {', '.join(toolboxes)}")

    if req_map:
        parts.append("")
        parts.append("REQUIREMENT → PRIMARY TOOLBOX (for gap detection):")
        for req, mapping in req_map.items():
            parts.append(f"  {req} → {mapping.get('primary_toolbox', '')}")

    return "\n".join(parts)
